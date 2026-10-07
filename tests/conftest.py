import pytest

import classifier.llm_extractor as llm_extractor


@pytest.fixture(autouse=True)
def no_real_llm_calls(monkeypatch):
    """
    Keep the test suite offline and deterministic: by default the Groq client is
    unavailable, so the pipeline takes its raw-text fallback path. Tests that want
    the LLM path patch `llm_extractor._get_client` (or `main.pipeline.extractor`)
    themselves. This also stops a developer's real GROQ_API_KEY in .env from
    triggering network calls during pytest.
    """
    def unavailable():
        raise RuntimeError("LLM disabled in tests")

    monkeypatch.setattr(llm_extractor, "_get_client", unavailable)

    # Also block the pipeline's cached reference to the extractor if main
    # has already been imported (main.pipeline.extractor is captured at
    # import time, so patching llm_extractor alone isn't enough).
    try:
        import main
        if hasattr(main, "pipeline") and hasattr(main.pipeline, "extractor"):
            def blocked(text):
                raise RuntimeError("LLM disabled in tests")
            main.pipeline.extractor = blocked
    except ImportError:
        pass