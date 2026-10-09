import pytest

import classifier.llm_extractor as ex
from classifier.pipeline import SACAPipeline
from classifier.red_flags import check_red_flags, apply_floor, RULES, EXTRA_TOKENS


def run(tokens, raw="x"):
    return SACAPipeline(extractor=lambda t: tokens).classify(raw)


def test_screenshot_case_is_no_longer_medium():
    # "chest heavy, can't breathe, vomiting blood, headache" used to come back MEDIUM @ 97.9%
    r = run("chest_pain breathlessness vomiting vomiting_blood headache")
    assert r["severity"] == "CRITICAL"
    assert r["model_severity"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    assert r["override_applied"] is True
    assert r["confidence"] is None          # rules have no probability
    assert "vomiting blood" in r["reason"]


@pytest.mark.parametrize("tokens,minimum", [
    ("coma", "CRITICAL"),
    ("chest_pain breathlessness", "CRITICAL"),
    ("seizure", "CRITICAL"),
    ("chest_pain", "HIGH"),
    ("breathlessness", "HIGH"),
    ("stiff_neck high_fever", "HIGH"),
])
def test_floors(tokens, minimum):
    r = run(tokens)
    order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    assert order.index(r["severity"]) >= order.index(minimum)


def test_benign_cases_are_untouched():
    for tokens in ("headache stomach_pain", "itching blurred_and_distorted_vision"):
        r = run(tokens)
        assert r["override_applied"] is False and r["red_flags"] == []
        assert r["severity"] == r["model_severity"]
        assert r["confidence"] is not None


def test_never_lowers_severity():
    assert apply_floor("CRITICAL", "HIGH") == "CRITICAL"
    assert apply_floor("HIGH", None) == "HIGH"
    assert apply_floor("LOW", "HIGH") == "HIGH"


def test_offline_fallback_catches_plain_phrases():
    # no LLM at all (conftest disables it): raw words "chest pain" still trigger the floor
    r = SACAPipeline(extractor=ex.extract_symptoms).classify("i have chest pain and shortness")
    assert r["fallback_used"] is True and r["severity"] in {"HIGH", "CRITICAL"}


def test_every_rule_token_is_known_to_the_extractor():
    vocab = set(ex.get_vocabulary())
    for required, _, _ in RULES:
        assert required <= vocab, f"{required - vocab} not extractable"
    assert set(EXTRA_TOKENS) <= vocab