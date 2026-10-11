import pytest


@pytest.fixture(autouse=True)
def _no_network_warmup(monkeypatch):
    # The catalog warm-up fetches ollama.com and Hugging Face; tests must not.
    monkeypatch.setenv("AMY_CATALOG_WARMUP", "false")
