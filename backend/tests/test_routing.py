import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app import routing
from app.config import ProviderConfig, Settings
from app.main import create_app

AUTH = {"Authorization": "Bearer k"}
MODELS = ["llama3.1:8b", "qwen2.5-coder:7b"]


def sse(text: str) -> bytes:
    chunk = json.dumps({"choices": [{"delta": {"content": text}}]})
    return f"data: {chunk}\n\ndata: [DONE]\n\n".encode()


class Server:
    """A provider with a general model and a coding model; records which one each chat used."""

    def __init__(self):
        self.used: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": m} for m in MODELS]})
        body = json.loads(request.content)
        if not body.get("stream"):  # memory extraction
            empty = {"add": [], "update": [], "remove": []}
            return httpx.Response(
                200, json={"choices": [{"message": {"content": json.dumps(empty)}}]}
            )
        self.used.append(body["model"])
        return httpx.Response(200, content=sse("ok"), headers={"content-type": "text/event-stream"})


@pytest.fixture
def client(tmp_path):
    settings = Settings(
        data_dir=tmp_path,
        api_key="k",
        ollama_url="",
        static_dir=tmp_path,
        providers=[ProviderConfig(name="fake", base_url="http://fake/v1")],
    )
    with TestClient(create_app(settings)) as c:
        c.server = Server()
        c.app.state.http = httpx.AsyncClient(transport=httpx.MockTransport(c.server))
        yield c


def chat(client, message, **extra):
    body = {"model": "auto", "message": message, **extra}
    with client.stream("POST", "/api/agent/chat", headers=AUTH, json=body) as resp:
        assert resp.status_code == 200, resp.read()
        return [json.loads(line[6:]) for line in resp.iter_lines() if line.startswith("data: ")]


def test_code_is_recognised():
    assert routing.looks_like_code("why does this fail?\n```py\nprint(x)\n```")
    assert routing.looks_like_code("Traceback (most recent call last):\n  File ...")
    assert routing.looks_like_code("How do I write a regex in python for emails?")
    assert not routing.looks_like_code("What's a good dinner with chicken and rice?")
    assert not routing.looks_like_code("Remember that my sister lives in Portland")
    assert routing.is_coder("ollama/qwen2.5-coder:7b") and routing.is_coder("devstral:24b")
    assert not routing.is_coder("ollama/llama3.1:8b")


def test_ranking_prefers_fit_then_tools_then_kind_then_size():
    models = [
        {"id": "big", "coder": False, "tools": True, "size_gb": 40, "fit": "too_large"},
        {"id": "general", "coder": False, "tools": True, "size_gb": 5, "fit": "gpu"},
        {"id": "coder", "coder": True, "tools": True, "size_gb": 4, "fit": "gpu"},
        {"id": "no-tools", "coder": True, "tools": False, "size_gb": 9, "fit": "gpu"},
    ]
    assert [m["id"] for m in routing.rank(models, "chat")][:2] == ["general", "coder"]
    assert [m["id"] for m in routing.rank(models, "coding")][:2] == ["coder", "general"]


def test_auto_sends_code_to_the_coding_model_and_stays_there(client):
    events = chat(client, "What should I cook tonight?")
    assert client.server.used[-1] == "llama3.1:8b"
    assert {"type": "model", "model": "fake/llama3.1:8b", "purpose": "chat"} in events

    conv = chat(client, "Fix this:\n```js\nconst x = ;\n```")[0]["conversation_id"]
    assert client.server.used[-1] == "qwen2.5-coder:7b"
    chat(client, "thanks, now make it faster", conversation_id=conv)
    assert client.server.used[-1] == "qwen2.5-coder:7b"  # follow-ups stay with the coder
    messages = client.get(f"/api/conversations/{conv}", headers=AUTH).json()["messages"]
    assert messages[-1]["model"] == "fake/qwen2.5-coder:7b"  # which model actually answered


def test_pinned_defaults_win_and_show_in_settings(client):
    models = client.get("/api/settings", headers=AUTH).json()["models"]
    assert models["chat"]["best"] == "fake/llama3.1:8b"
    assert models["coding"]["best"] == "fake/qwen2.5-coder:7b"
    r = client.put(
        "/api/settings/models",
        headers=AUTH,
        json={"capability": "chat", "model": "fake/qwen2.5-coder:7b"},
    )
    assert r.json()["chat"]["pinned"] == "fake/qwen2.5-coder:7b"
    chat(client, "What should I cook tonight?")
    assert client.server.used[-1] == "qwen2.5-coder:7b"
    bad = client.put(
        "/api/settings/models", headers=AUTH, json={"capability": "coding", "model": "fake/nope"}
    )
    assert bad.status_code == 400


def test_a_named_model_is_used_as_is(client):
    body = {"model": "fake/llama3.1:8b", "message": "```py\nx = 1\n```"}
    with client.stream("POST", "/api/agent/chat", headers=AUTH, json=body) as resp:
        events = [json.loads(line[6:]) for line in resp.iter_lines() if line.startswith("data: ")]
    assert client.server.used[-1] == "llama3.1:8b"
    assert not any(e["type"] == "model" for e in events)
