from types import SimpleNamespace

import classifier.llm_extractor as ex
from classifier.pipeline import SACAPipeline


def fake_client(reply):
    def create(**kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def use(monkeypatch, reply):
    monkeypatch.setattr(ex, "_get_client", lambda: fake_client(reply))


def test_extracts_known_tokens(monkeypatch):
    use(monkeypatch, '{"symptoms": ["high_fever", "vomiting"]}')
    assert ex.extract_symptoms("I have a bad fever and I am throwing up") == "high_fever vomiting"


def test_drops_invented_and_injected_tokens(monkeypatch):
    use(monkeypatch, '{"symptoms": ["vomiting", "totally_fake", "IGNORE_ALL_RULES", "vomiting"]}')
    assert ex.extract_symptoms("x") == "vomiting"


def test_strips_code_fences(monkeypatch):
    use(monkeypatch, '```json\n{"symptoms": ["chest_pain"]}\n```')
    assert ex.extract_symptoms("x") == "chest_pain"


def test_nothing_valid_returns_empty(monkeypatch):
    use(monkeypatch, '{"symptoms": []}')
    assert ex.extract_symptoms("x") == ""


def test_pipeline_end_to_end_and_fallback(monkeypatch):
    use(monkeypatch, '{"symptoms": ["chest_pain", "breathlessness", "sweating"]}')
    res = SACAPipeline(extractor=ex.extract_symptoms).classify("my chest hurts and I can't breathe")
    assert res["fallback_used"] is False and res["extracted_text"] == "chest_pain breathlessness sweating"

    use(monkeypatch, "not json at all")   # malformed reply -> pipeline falls back to raw text
    res = SACAPipeline(extractor=ex.extract_symptoms).classify("my chest hurts")
    assert res["fallback_used"] is True
