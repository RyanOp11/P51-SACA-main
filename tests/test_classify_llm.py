"""
/classify endpoint with the LLM extraction step wired in (main.py -> SACAPipeline).
The LLM is always faked here; the real Groq API is never called.
"""
import pytest
from fastapi.testclient import TestClient

import database
import main


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "test_saca.db"))
    database.init_db()
    return TestClient(main.app)


def test_classify_uses_llm_extraction_when_available(client, monkeypatch):
    monkeypatch.setattr(main.pipeline, "extractor", lambda text: "chest_pain breathlessness sweating")
    r = client.post("/classify", json={"symptom_text": "my chest hurts and I can't breathe"})
    body = r.json()
    assert r.status_code == 200
    assert body["input_source"] == "LLM extraction"
    assert body["fallback_used"] is False
    assert body["extracted_text"] == "chest_pain breathlessness sweating"
    assert body["severity"] in {"CRITICAL", "HIGH", "MEDIUM", "LOW"}
    assert body["symptoms"] == ["chest_pain", "breathlessness", "sweating"]


def test_classify_falls_back_to_raw_text_when_llm_unavailable(client, monkeypatch):
    # conftest.py makes the Groq client unavailable, exactly like a missing API key
    monkeypatch.setattr(main.pipeline, "extractor", None)
    r = client.post("/classify", json={"symptom_text": "fever headache nausea"})
    body = r.json()
    assert r.status_code == 200
    assert body["fallback_used"] is True
    assert body["input_source"] == "Raw text fallback"
    assert body["severity"] in {"CRITICAL", "HIGH", "MEDIUM", "LOW"}


def test_classify_falls_back_when_llm_times_out(client, monkeypatch):
    def slow(text):
        raise TimeoutError("timed out")
    monkeypatch.setattr(main.pipeline, "extractor", slow)
    body = client.post("/classify", json={"symptom_text": "fever headache"}).json()
    assert body["fallback_used"] is True


def test_session_is_saved_with_extracted_symptoms(client, monkeypatch):
    monkeypatch.setattr(main.pipeline, "extractor", lambda text: "high_fever vomiting")
    client.post("/classify", json={"symptom_text": "very hot and throwing up"})
    sessions = client.get("/sessions").json()
    assert len(sessions) == 1
    assert sessions[0]["symptom_text"] == "very hot and throwing up"   # original words kept


def test_app_imports_without_groq_key(monkeypatch):
    # Regression test for the import-time Groq() crash in ammu_safety.py.
    # The current code uses lazy client init, so importing must never raise.
    # We clear the env var, reload ammu_safety, then clear any key that
    # load_dotenv() may have re-populated before testing the lazy failure.
    import importlib
    import ammu_safety
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(ammu_safety, "_client", None)
    importlib.reload(ammu_safety)   # must not raise on import

    # load_dotenv() runs on reload, so clear it again before testing
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(ammu_safety, "_client", None)

    with pytest.raises(Exception):
        ammu_safety.make_llm_reply("LOW", ["cough"], "r", "f")