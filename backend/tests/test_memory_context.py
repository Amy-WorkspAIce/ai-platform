import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app import context, memory
from app.agent import history_with_ids
from app.config import ProviderConfig, Settings
from app.main import create_app

AUTH = {"Authorization": "Bearer k"}


def sse(text: str) -> bytes:
    return f"data: {json.dumps({'choices': [{'delta': {'content': text}}]})}\n\ndata: [DONE]\n\n".encode()


class FakeLLM:
    """Streams chat replies; answers memory extraction and summary requests (non-streaming)."""

    def __init__(self, reply="Sure!", extraction=None, summary="SUMMARY: user is planning a trip"):
        self.reply = reply
        self.extraction = (
            extraction if extraction is not None else {"add": [], "update": [], "remove": []}
        )
        self.summary = summary
        self.chat_requests: list[dict] = []
        self.extract_requests: list[dict] = []
        self.summary_requests: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if not body.get("stream"):
            system = body["messages"][0]["content"]
            if system.startswith("You maintain a long-term memory"):
                self.extract_requests.append(body)
                return httpx.Response(
                    200, json={"choices": [{"message": {"content": json.dumps(self.extraction)}}]}
                )
            self.summary_requests.append(body)
            return httpx.Response(200, json={"choices": [{"message": {"content": self.summary}}]})
        self.chat_requests.append(body)
        return httpx.Response(
            200, content=sse(self.reply), headers={"content-type": "text/event-stream"}
        )


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
        yield c


def chat(client, llm: FakeLLM, message: str, conversation_id=None, folder_id=None):
    client.app.state.http = httpx.AsyncClient(transport=httpx.MockTransport(llm))
    body = {
        "model": "fake/m",
        "message": message,
        "conversation_id": conversation_id,
        "folder_id": folder_id,
    }
    with client.stream("POST", "/api/agent/chat", headers=AUTH, json=body) as resp:
        return [json.loads(line[6:]) for line in resp.iter_lines() if line.startswith("data: ")]


# ---------------------------------------------------------------- folders


def test_folders(client):
    folder = client.post("/api/folders", headers=AUTH, json={"name": "Work"}).json()
    conv = client.post("/api/conversations", headers=AUTH, json={"title": "a"}).json()
    moved = client.patch(
        f"/api/conversations/{conv['id']}", headers=AUTH, json={"folder_id": folder["id"]}
    ).json()
    assert moved["folder_id"] == folder["id"]
    assert (
        client.patch(f"/api/folders/{folder['id']}", headers=AUTH, json={"collapsed": True}).json()[
            "collapsed"
        ]
        is True
    )
    unfiled = client.patch(
        f"/api/conversations/{conv['id']}", headers=AUTH, json={"folder_id": ""}
    ).json()
    assert unfiled["folder_id"] is None
    client.patch(f"/api/conversations/{conv['id']}", headers=AUTH, json={"folder_id": folder["id"]})
    assert client.delete(f"/api/folders/{folder['id']}", headers=AUTH).status_code == 204
    kept = client.get(f"/api/conversations/{conv['id']}", headers=AUTH).json()
    assert kept["folder_id"] is None  # chats survive their folder
    assert (
        client.patch(
            f"/api/conversations/{conv['id']}", headers=AUTH, json={"folder_id": "nope"}
        ).status_code
        == 404
    )


def test_new_chat_inside_folder(client):
    folder = client.post("/api/folders", headers=AUTH, json={"name": "Trips"}).json()
    events = chat(client, FakeLLM(), "plan a trip", folder_id=folder["id"])
    conv = client.get(f"/api/conversations/{events[0]['conversation_id']}", headers=AUTH).json()
    assert conv["folder_id"] == folder["id"]


# ---------------------------------------------------------------- memory


def test_learns_facts_after_the_answer(client):
    llm = FakeLLM(
        extraction={
            "add": [{"text": "Has an RTX 2080 Ti", "category": "setup"}],
            "update": [],
            "remove": [],
        }
    )
    events = chat(client, llm, "I have an RTX 2080 Ti in my Unraid box")
    types = [e["type"] for e in events]
    assert types.index("answer_done") < types.index("memory") < types.index("done")
    note = next(e for e in events if e["type"] == "memory")
    assert note == {"type": "memory", "action": "remembered", "text": "Has an RTX 2080 Ti"}
    facts = client.get("/api/memory", headers=AUTH).json()["facts"]
    assert facts[0]["text"] == "Has an RTX 2080 Ti" and facts[0]["category"] == "setup"
    assert facts[0]["source_conversation_id"] == events[0]["conversation_id"]
    assert "embedding" not in facts[0]

    # Next chat: the fact is recalled into the system prompt.
    llm2 = FakeLLM()
    chat(client, llm2, "what should I run?")
    assert "Has an RTX 2080 Ti" in llm2.chat_requests[0]["messages"][0]["content"]


def test_remember_that_refers_back_to_earlier_messages(client):
    llm = FakeLLM()
    first = chat(client, llm, "I have two cats, Miso and Tofu")
    chat(client, llm, "Please remember that", conversation_id=first[0]["conversation_id"])
    asked = llm.extract_requests[-1]
    assert "Earlier in this chat:" in asked["messages"][1]["content"]
    assert "Miso and Tofu" in asked["messages"][1]["content"]
    assert asked["reasoning_effort"] == "none"  # no minutes of thinking for a background job
    # Amy is told memory exists, so it doesn't claim it can't remember.
    assert "You have long-term memory" in llm.chat_requests[-1]["messages"][0]["content"]


def test_a_failed_memory_update_is_shown(client):
    class Rambler(FakeLLM):
        def __call__(self, request):
            body = json.loads(request.content)
            if not body.get("stream"):
                return httpx.Response(200, json={"choices": [{"message": {"content": "Hmm."}}]})
            return super().__call__(request)

    events = chat(client, Rambler(), "I live in Redmond")
    note = next(e for e in events if e["type"] == "memory")
    assert note["action"] == "failed"


def test_background_replies_fall_back_without_the_reasoning_option():
    from app.providers.base import Provider

    bodies = []

    def server(request):
        body = json.loads(request.content)
        bodies.append(body)
        if "reasoning_effort" in body:
            return httpx.Response(400, json={"error": "unknown field"})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(server)) as c:
            return await Provider("p", "http://x/v1").quick_reply(c, {"model": "m", "messages": []})

    assert asyncio.run(go()) == "ok"
    assert len(bodies) == 2 and "reasoning_effort" not in bodies[1]


def test_updates_removes_and_filters_secrets(client):
    db = client.app.state.db
    old = db.add_memory("Lives in Seattle", "personal")
    wrong = db.add_memory("Has a cat named Tom", "personal")
    llm = FakeLLM(
        extraction={
            "add": [{"text": "My API key is sk-abcdefghijklmnop"}, {"text": "Lives in Seattle"}],
            "update": [{"id": old["id"], "text": "Lives in Redmond"}],
            "remove": [wrong["id"], "not-a-real-id"],
        }
    )
    events = chat(client, llm, "I moved to Redmond. I don't have a cat, forget that.")
    actions = sorted((e["action"], e["text"]) for e in events if e["type"] == "memory")
    assert actions == [("removed", "Has a cat named Tom"), ("updated", "Lives in Redmond")]
    texts = [f["text"] for f in db.list_memories()]
    assert texts == ["Lives in Redmond"]  # no duplicate, no secret
    # Extraction saw the known facts with their ids.
    assert old["id"] in llm.extract_requests[0]["messages"][0]["content"]


def test_memory_switches(client):
    client.app.state.db.add_memory("Prefers concise answers", "preferences")
    client.put("/api/memory/settings", headers=AUTH, json={"auto": False})
    llm = FakeLLM(extraction={"add": [{"text": "Likes tea"}], "update": [], "remove": []})
    chat(client, llm, "I like tea")
    assert llm.extract_requests == []  # not learning
    assert (
        "Prefers concise answers" in llm.chat_requests[0]["messages"][0]["content"]
    )  # still recalling
    client.put("/api/memory/settings", headers=AUTH, json={"enabled": False})
    llm = FakeLLM()
    chat(client, llm, "hi")
    assert "Prefers concise answers" not in llm.chat_requests[0]["messages"][0]["content"]


def test_memory_api(client):
    added = client.post(
        "/api/memory", headers=AUTH, json={"text": "Works on Amy", "category": "projects"}
    ).json()
    assert (
        client.post("/api/memory", headers=AUTH, json={"text": "password is hunter22"}).status_code
        == 400
    )
    edited = client.patch(
        f"/api/memory/{added['id']}", headers=AUTH, json={"text": "Builds Amy WorkspAIce"}
    ).json()
    assert edited["text"] == "Builds Amy WorkspAIce"
    assert [
        f["id"] for f in client.get("/api/memory?q=workspaice", headers=AUTH).json()["facts"]
    ] == [added["id"]]
    assert client.delete("/api/memory", headers=AUTH).status_code == 400  # needs confirm
    assert client.delete("/api/memory?confirm=true", headers=AUTH).json() == {"deleted": 1}


def test_recall_ranks_by_keywords_when_there_are_many_facts(client):
    db = client.app.state.db
    for i in range(memory.INCLUDE_ALL_BELOW + 5):
        db.add_memory(f"Filler fact number {i} about gardening", "general")
    db.add_memory("Owns an RTX 2080 Ti graphics card", "setup")
    facts = asyncio.run(memory.recall(client.app, "which graphics card do I have?"))
    assert len(facts) == memory.RECALL_LIMIT
    assert facts[0]["text"] == "Owns an RTX 2080 Ti graphics card"


def test_embedding_helpers():
    v = [0.1, 0.2, 0.3]
    assert memory.unpack(memory.pack(v)) == pytest.approx(v)
    assert memory.cosine([1, 0], [1, 0]) == pytest.approx(1)
    assert memory.cosine([1, 0], [0, 1]) == pytest.approx(0)
    assert memory.clean_fact("  Prefers   dark mode ") == "Prefers dark mode"
    assert memory.clean_fact("token ghp_abcdefghijklmnopqrstu") is None


# ---------------------------------------------------------------- context


def test_trim_tool_results_keeps_recent():
    msgs = [{"role": "tool", "content": "x" * 5000} for _ in range(5)]
    context.trim_tool_results(msgs, keep_recent=2)
    assert all("trimmed" in m["content"] for m in msgs[:3])
    assert all(len(m["content"]) == 5000 for m in msgs[3:])


def test_history_with_ids_skips_summarized_part():
    stored = [
        {"id": "1", "role": "user", "content": "a", "meta": None},
        {"id": "2", "role": "assistant", "content": "b", "meta": None},
        {"id": "3", "role": "user", "content": "c", "meta": None},
    ]
    msgs, ids = history_with_ids(stored, after_id="2")
    assert ids == ["3"] and msgs == [{"role": "user", "content": "c"}]


def test_long_chat_is_condensed_into_a_summary(client):
    client.app.state.settings.context_tokens = 4096
    llm = FakeLLM()
    conv = None
    notices = []
    for i in range(12):
        events = chat(client, llm, f"Message {i}: " + "trip planning details " * 60, conv)
        conv = events[0]["conversation_id"]
        notices += [e["message"] for e in events if e["type"] == "notice"]
    assert llm.summary_requests, "expected the history to be summarized"
    assert any("Condensing" in n for n in notices)  # the user is told
    stored = client.get(f"/api/conversations/{conv}", headers=AUTH).json()
    assert stored["summary"].startswith("SUMMARY") and stored["summary_upto"]
    last = llm.chat_requests[-1]["messages"]
    assert any(
        m["role"] == "system" and "Summary of the earlier part" in m["content"] for m in last
    )
    assert context.message_tokens(last) < 4096
