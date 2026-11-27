"""Connectors: apps' MCP tools in chat."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import ProviderConfig, Settings
from app.main import create_app
from tests.test_agent import sse_body

AUTH = {"Authorization": "Bearer k"}
DOCS = "http://docs:8500/mcp"

TOOLS = [
    {
        "name": "search_documents",
        "description": "Find documents by title or text.",
        "inputSchema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "delete_document",
        "description": "Delete a document.",
        "inputSchema": {"type": "object", "properties": {"id": {"type": "string"}}},
    },
    {
        "name": "append_text",
        "description": "Add text to the end of a document.",
        "inputSchema": {"type": "object", "properties": {"id": {"type": "string"}}},
        "annotations": {"destructiveHint": False},
    },
]


class FakeDocs:
    """An MCP server (Streamable HTTP) that answers with JSON or, if asked, event streams."""

    def __init__(self, events: bool = False):
        self.events = events
        self.calls: list[dict] = []
        self.keys: list[str | None] = []
        self.up = True

    def handle(self, request: httpx.Request) -> httpx.Response:
        if not self.up:
            raise httpx.ConnectError("down")
        self.keys.append(request.headers.get("authorization"))
        msg = json.loads(request.content)
        if "id" not in msg:
            return httpx.Response(202)
        method = msg["method"]
        headers = {"mcp-session-id": "s1"}
        if method == "initialize":
            result = {
                "protocolVersion": "2025-06-18",
                "serverInfo": {"name": "Docs", "version": "0.1"},
                "instructions": "The user's documents.",
                "capabilities": {"tools": {}},
            }
        elif method == "tools/list":
            assert request.headers.get("mcp-session-id") == "s1"
            if msg["params"].get("cursor"):
                result = {"tools": TOOLS[1:]}
            else:
                result = {"tools": TOOLS[:1], "nextCursor": "p2"}
        elif method == "tools/call":
            self.calls.append(msg["params"])
            if msg["params"]["name"] == "delete_document":
                result = {
                    "content": [{"type": "text", "text": "No such document"}],
                    "isError": True,
                }
            else:
                result = {"content": [{"type": "text", "text": "Found: Q3 plan (id d1)"}]}
        else:
            return httpx.Response(
                200, json={"jsonrpc": "2.0", "id": msg["id"], "error": {"message": "nope"}}
            )
        body = {"jsonrpc": "2.0", "id": msg["id"], "result": result}
        if self.events:
            text = f"event: message\ndata: {json.dumps({'jsonrpc': '2.0', 'method': 'notifications/progress'})}\n\n"
            text += f"event: message\ndata: {json.dumps(body)}\n\n"
            return httpx.Response(
                200, text=text, headers={**headers, "content-type": "text/event-stream"}
            )
        return httpx.Response(200, json=body, headers=headers)


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


def connect(client, docs: FakeDocs, model_turns: list | None = None, seen: list | None = None):
    turns = model_turns or []
    seen = seen if seen is not None else []

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.host == "docs":
            return docs.handle(request)
        seen.append(json.loads(request.content))
        turn = turns[len(seen) - 1] if len(seen) <= len(turns) else [{"content": "[]"}]
        return httpx.Response(
            200, content=sse_body(turn), headers={"content-type": "text/event-stream"}
        )

    client.app.state.http = httpx.AsyncClient(transport=httpx.MockTransport(handle))


@pytest.mark.parametrize("events", [False, True])
def test_hub_registers_an_app_and_its_tools_are_listed(client, events):
    docs = FakeDocs(events)
    connect(client, docs)
    r = client.put(
        "/api/connectors/docs",
        headers=AUTH,
        json={"url": DOCS, "api_key": "docs-key", "source": "hub"},
    )
    assert r.status_code == 200, r.text
    c = r.json()
    assert c["name"] == "Docs" and c["has_key"] and "api_key" not in c and c["error"] is None
    assert [(t["name"], t["changes"]) for t in c["tools"]] == [
        ("search_documents", False),
        ("delete_document", True),
        ("append_text", False),
    ]
    assert set(docs.keys) == {"Bearer docs-key"}

    # Updating keeps what isn't sent; a bad id or a new connector without a URL is refused.
    assert client.put("/api/connectors/docs", headers=AUTH, json={"enabled": False}).json()[
        "has_key"
    ]
    assert client.put("/api/connectors/Bad_Id", headers=AUTH, json={"url": DOCS}).status_code == 400
    assert client.put("/api/connectors/other", headers=AUTH, json={"name": "x"}).status_code == 400
    assert client.delete("/api/connectors/docs", headers=AUTH).status_code == 204
    assert client.get("/api/connectors", headers=AUTH).json() == []


def test_an_app_that_is_down_shows_why(client):
    docs = FakeDocs()
    docs.up = False
    connect(client, docs)
    c = client.put("/api/connectors/docs", headers=AUTH, json={"url": DOCS}).json()
    assert "Can't reach" in c["error"] and c["tools"] == []


def test_chat_uses_an_apps_tools(client):
    docs = FakeDocs()
    seen: list[dict] = []
    call = [{"tool_calls": [{"index": 0, "id": "c1", "function": {
        "name": "docs_search_documents", "arguments": '{"query": "Q3"}'}}]}]  # fmt: skip
    connect(client, docs, [call, [{"content": "Your Q3 plan is there."}]], seen)
    client.put("/api/connectors/docs", headers=AUTH, json={"url": DOCS, "api_key": "docs-key"})

    with client.stream(
        "POST",
        "/api/agent/chat",
        headers=AUTH,
        json={"model": "fake/m", "message": "find my Q3 plan", "web": False},
    ) as resp:
        events = [json.loads(line[6:]) for line in resp.iter_lines() if line.startswith("data: ")]

    first = seen[0]
    names = [t["function"]["name"] for t in first["tools"]]
    assert {"docs_search_documents", "docs_delete_document", "docs_append_text"} <= set(names)
    assert "Connected apps" in first["messages"][0]["content"]
    assert "- Docs: The user's documents." in first["messages"][0]["content"]
    call_event = next(e for e in events if e["type"] == "tool_call")
    assert call_event["summary"] == "Q3" and not call_event["needs_approval"]
    result = next(e for e in events if e["type"] == "tool_result")
    assert result["ok"] and "Q3 plan" in result["output"]
    assert docs.calls == [{"name": "search_documents", "arguments": {"query": "Q3"}}]
