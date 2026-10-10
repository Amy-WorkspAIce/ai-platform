import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

AUTH = {"Authorization": "Bearer k"}


@pytest.fixture
def client(tmp_path):
    settings = Settings(data_dir=tmp_path, api_key="k", ollama_url="", static_dir=tmp_path)
    with TestClient(create_app(settings)) as c:
        yield c


def test_conversation_lifecycle(client):
    conv = client.post(
        "/api/conversations", headers=AUTH, json={"title": "Hi", "model": "m"}
    ).json()
    cid = conv["id"]

    r = client.post(
        f"/api/conversations/{cid}/messages", headers=AUTH, json={"role": "user", "content": "q"}
    )
    assert r.status_code == 201
    client.post(
        f"/api/conversations/{cid}/messages",
        headers=AUTH,
        json={"role": "assistant", "content": "a", "reasoning": "thought", "model": "m"},
    )

    full = client.get(f"/api/conversations/{cid}", headers=AUTH).json()
    assert [m["role"] for m in full["messages"]] == ["user", "assistant"]
    assert full["messages"][1]["reasoning"] == "thought"

    renamed = client.patch(f"/api/conversations/{cid}", headers=AUTH, json={"title": "New"}).json()
    assert renamed["title"] == "New"

    assert client.get("/api/conversations", headers=AUTH).json()[0]["id"] == cid
    assert client.delete(f"/api/conversations/{cid}", headers=AUTH).status_code == 204
    assert client.get(f"/api/conversations/{cid}", headers=AUTH).status_code == 404


def test_most_recent_first(client):
    a = client.post("/api/conversations", headers=AUTH, json={"title": "a"}).json()["id"]
    b = client.post("/api/conversations", headers=AUTH, json={"title": "b"}).json()["id"]
    client.post(
        f"/api/conversations/{a}/messages", headers=AUTH, json={"role": "user", "content": "x"}
    )
    assert [c["id"] for c in client.get("/api/conversations", headers=AUTH).json()] == [a, b]


def test_message_to_missing_conversation(client):
    r = client.post(
        "/api/conversations/nope/messages", headers=AUTH, json={"role": "user", "content": "x"}
    )
    assert r.status_code == 404


def test_requires_auth(client):
    assert client.get("/api/conversations").status_code == 401


def test_persists_across_restart(tmp_path):
    settings = Settings(data_dir=tmp_path, api_key="k", ollama_url="", static_dir=tmp_path)
    with TestClient(create_app(settings)) as c:
        c.post("/api/conversations", headers=AUTH, json={"title": "keep"})
    with TestClient(create_app(settings)) as c:
        assert c.get("/api/conversations", headers=AUTH).json()[0]["title"] == "keep"
