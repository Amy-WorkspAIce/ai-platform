import pytest


@pytest.fixture(autouse=True)
def _no_network_warmup(monkeypatch):
    # The catalog warm-up fetches ollama.com and Hugging Face; tests must not.
    monkeypatch.setenv("AMY_CATALOG_WARMUP", "false")


@pytest.fixture(autouse=True)
def _plenty_of_disk(monkeypatch):
    # Installs check free space; tests mustn't depend on the machine they run on.
    import shutil
    from collections import namedtuple

    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda _: usage(10**13, 0, 10**13))
