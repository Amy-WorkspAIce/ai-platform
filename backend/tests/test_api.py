import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import ProviderConfig, Settings
from app.main import create_app
from app.providers import ProviderRegistry

KEY = "test-key"


@pytest.fixture
def client(tmp_path):
    settings = Settings(data_dir=tmp_path, api_key=KEY, ollama_url="", static_dir=tmp_path)
    with TestClient(create_app(settings)) as c:
        yield c


def auth():
    return {"Authorization": f"Bearer {KEY}"}


def test_health_is_public(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.parametrize("path", ["/v1/models", "/api/system"])
def test_requires_api_key(client, path):
    assert client.get(path).status_code == 401
    assert client.get(path, headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get(path, headers=auth()).status_code == 200


def test_unknown_model_is_404(client):
    resp = client.post("/v1/chat/completions", headers=auth(), json={"model": "nope/x"})
    assert resp.status_code == 404


def test_generates_and_persists_api_key(tmp_path):
    settings = Settings(data_dir=tmp_path, ollama_url="", static_dir=tmp_path)
    first = create_app(settings).state.api_key
    assert first.startswith("amy-")
    assert create_app(settings).state.api_key == first


def test_resolve_keeps_slashes_in_upstream_name():
    registry = ProviderRegistry.from_settings(
        Settings(ollama_url="http://x:11434", providers=[ProviderConfig(name="a", base_url="u")])
    )
    provider, upstream = registry.resolve("ollama/hf.co/org/model:Q4")
    assert provider.name == "ollama"
    assert upstream == "hf.co/org/model:Q4"
    assert registry.resolve("ollama") is None


def test_proxies_chat_to_provider(tmp_path):
    def upstream(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "llama3"}]})
        body = request.read().decode()
        assert '"model":"llama3"' in body.replace(" ", "")
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})

    settings = Settings(
        data_dir=tmp_path,
        api_key=KEY,
        ollama_url="",
        static_dir=tmp_path,
        providers=[ProviderConfig(name="local", base_url="http://local/v1")],
    )
    with TestClient(create_app(settings)) as c:
        c.app.state.http = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
        models = c.get("/v1/models", headers=auth()).json()["data"]
        assert [m["id"] for m in models] == ["local/llama3"]
        resp = c.post(
            "/v1/chat/completions",
            headers=auth(),
            json={"model": "local/llama3", "messages": [{"role": "user", "content": "yo"}]},
        )
        assert resp.status_code == 200
        assert resp.json()["choices"][0]["message"]["content"] == "hi"
