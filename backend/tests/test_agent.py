import asyncio
import json
import os
import sys

import httpx
import pytest
from fastapi.testclient import TestClient

from app.agent import AgentRun, history_to_messages, run_agent
from app.config import ProviderConfig, Settings
from app.main import create_app
from app.providers import Provider
from app.tools import Toolset, workspace_tools
from app.tools.web import WebError, _check_url, parse_duckduckgo
from app.tools.workspace import ToolError, Workspace, clip

AUTH = {"Authorization": "Bearer k"}
posix_only = pytest.mark.skipif(os.name == "nt", reason="bash/process groups")


# ---- workspace tools ----


def test_paths_are_confined(tmp_path):
    ws = Workspace(tmp_path / "ws")
    ws.root.mkdir()
    for bad in ("../outside.txt", "/etc/passwd", "a/../../x"):
        with pytest.raises(ToolError):
            ws.path(bad)
    outside = tmp_path / "secret"
    outside.write_text("s")
    (ws.root / "link").symlink_to(outside) if os.name != "nt" else None
    if os.name != "nt":
        with pytest.raises(ToolError):
            ws.read_file("link")


def test_file_tools(tmp_path):
    ws = Workspace(tmp_path)
    assert "Created" in ws.write_file("src/app.py", "a = 1\nb = 2\n")
    assert "    1  a = 1" in ws.read_file("src/app.py")
    assert ws.edit_file("src/app.py", "b = 2", "b = 3") == "Edited src/app.py"
    with pytest.raises(ToolError, match="not found"):
        ws.edit_file("src/app.py", "zzz", "y")
    ws.write_file("dup.txt", "x\nx\n")
    with pytest.raises(ToolError, match="2 times"):
        ws.edit_file("dup.txt", "x", "y")
    listing = ws.list_files()
    assert "src/" in listing and "app.py" in listing


def test_search_files(tmp_path):
    ws = Workspace(tmp_path)
    ws.write_file("a.py", "def hello():\n    pass\n")
    out = asyncio.run(ws.search("def hel+o"))
    assert "a.py:1:" in out


def test_env_strips_secrets_and_platform_venv(tmp_path, monkeypatch):
    monkeypatch.setenv("AMY_API_KEY", "secret")
    monkeypatch.setenv("MY_SERVICE_TOKEN", "secret")
    monkeypatch.setenv("VIRTUAL_ENV", "/opt/venv")
    env = Workspace(tmp_path, github_token="ghp_x").env()
    assert "AMY_API_KEY" not in env and "MY_SERVICE_TOKEN" not in env and "VIRTUAL_ENV" not in env
    assert env["GH_TOKEN"] == "ghp_x" and env["GIT_CONFIG_KEY_0"].endswith(".helper")
    if sys.prefix != sys.base_prefix:
        assert sys.prefix not in env["PATH"]


@posix_only
def test_run_command(tmp_path):
    ws = Workspace(tmp_path)
    out, ok = asyncio.run(ws.execute("echo hi && pwd && exit 3"))
    assert "hi" in out and str(tmp_path.resolve()) in out and "[exit code 3]" in out and not ok
    out, ok = asyncio.run(ws.execute("sleep 5", timeout=0.5))
    assert "[Timed out" in out and not ok
    assert "[exit code 0]" in asyncio.run(ws.run("true"))
    with pytest.raises(ToolError, match="exit code 1"):
        asyncio.run(ws.run("false"))


def test_workspace_venv_is_auto_activated(tmp_path):
    bin_dir = tmp_path / ".venv" / ("Scripts" if os.name == "nt" else "bin")
    bin_dir.mkdir(parents=True)
    env = Workspace(tmp_path).env()
    assert env["PATH"].startswith(str(bin_dir)) and env["VIRTUAL_ENV"] == str(bin_dir.parent)


def test_default_git_identity(tmp_path):
    home = tmp_path / "home"
    Workspace(tmp_path, home=home).env()
    assert "name = Amy Agent" in (home / ".gitconfig").read_text()
    (home / ".gitconfig").write_text("[user]\n\tname = Me\n")
    Workspace(tmp_path, home=home).env()
    assert "Me" in (home / ".gitconfig").read_text()  # never overwritten


def test_clip_keeps_head_and_tail():
    text = "START" + "x" * 50_000 + "END"
    out = clip(text, 1000)
    assert out.startswith("START") and out.endswith("END") and "omitted" in out


# ---- web ----

DDG_HTML = """
<div class="result results_links web-result"><div class="links_main result__body">
  <h2 class="result__title"><a rel="nofollow" class="result__a" href="https://example.com/a">Example <b>A</b></a></h2>
  <a class="result__snippet" href="https://example.com/a">First <b>snippet</b> here</a>
</div></div>
<div class="result"><div class="result__body">
  <h2 class="result__title"><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fb&rut=x">B</a></h2>
</div></div>
"""


def test_parse_duckduckgo():
    results = parse_duckduckgo(DDG_HTML)
    assert [(r.title, r.url) for r in results] == [
        ("Example A", "https://example.com/a"),
        ("B", "https://example.org/b"),
    ]
    assert results[0].snippet == "First snippet here"


@pytest.mark.parametrize(
    "url",
    ["http://127.0.0.1:8484/", "http://localhost/", "http://192.168.1.1/", "file:///etc/passwd"],
)
def test_fetch_blocks_private_addresses(url):
    with pytest.raises(WebError):
        asyncio.run(_check_url(url, Settings(web_allow_private=False)))


def test_fetch_private_allowed_when_configured():
    asyncio.run(_check_url("http://127.0.0.1/", Settings(web_allow_private=True)))


# ---- agent loop ----


def sse_body(chunks: list[dict]) -> bytes:
    lines = [f"data: {json.dumps({'choices': [{'delta': c}]})}\n\n" for c in chunks]
    return ("".join(lines) + "data: [DONE]\n\n").encode()


def scripted_model(turns: list[list[dict]], seen: list[dict]):
    """A fake OpenAI-compatible server that plays back one scripted turn per request."""

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            content=sse_body(turns[len(seen) - 1]),
            headers={"content-type": "text/event-stream"},
        )

    return handler


WRITE_CALL = [
    {"content": "I'll create it."},
    {
        "tool_calls": [
            {
                "index": 0,
                "id": "call_1",
                "function": {"name": "write_file", "arguments": '{"path": "hello.txt", '},
            }
        ]
    },
    {"tool_calls": [{"index": 0, "function": {"arguments": '"content": "hi"}'}}]},
]
DONE_TURN = [{"content": "Done: created hello.txt."}]


async def drive(run: AgentRun, toolset, turns, approve: bool | None):
    seen: list[dict] = []
    saved: list[dict] = []
    client = httpx.AsyncClient(transport=httpx.MockTransport(scripted_model(turns, seen)))
    task = asyncio.create_task(
        run_agent(
            run,
            provider=Provider("fake", "http://fake/v1"),
            model="m",
            client=client,
            messages=[{"role": "user", "content": "make hello.txt"}],
            toolset=toolset,
            persist=saved.append,
        )
    )
    events = []
    while not (task.done() and run.queue.empty()):
        try:
            ev = await asyncio.wait_for(run.queue.get(), 0.1)
        except TimeoutError:
            continue
        events.append(ev)
        if ev["type"] == "tool_call" and ev["needs_approval"] and approve is not None:
            await asyncio.sleep(0)
            run.resolve(ev["id"], approve)
    await task
    return events, saved, seen


def test_agent_runs_tool_after_approval(tmp_path):
    ws = Workspace(tmp_path)
    toolset = Toolset()
    for t in workspace_tools(ws, 60):
        toolset.add(t)
    events, saved, seen = asyncio.run(
        drive(AgentRun(), toolset, [WRITE_CALL, DONE_TURN], approve=True)
    )

    assert (tmp_path / "hello.txt").read_text() == "hi"
    call = next(e for e in events if e["type"] == "tool_call")
    assert call["needs_approval"] and call["summary"] == "hello.txt"
    assert next(e for e in events if e["type"] == "tool_result")["ok"]
    # The model saw the tool result on its second turn.
    assert seen[1]["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "Created hello.txt (1 lines)",
    }
    assert "tools" in seen[0]
    assert [m["role"] for m in saved] == ["assistant", "tool", "assistant"]


def test_agent_denied(tmp_path):
    toolset = Toolset()
    for t in workspace_tools(Workspace(tmp_path), 60):
        toolset.add(t)
    events, saved, seen = asyncio.run(
        drive(AgentRun(), toolset, [WRITE_CALL, DONE_TURN], approve=False)
    )
    assert not (tmp_path / "hello.txt").exists()
    assert "denied" in seen[1]["messages"][-1]["content"]


def test_repeated_failures_are_stopped(tmp_path):
    bad_edit = [
        {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "c",
                    "function": {"name": "read_file", "arguments": '{"path": "missing.txt"}'},
                }
            ]
        }
    ]
    toolset = Toolset()
    for t in workspace_tools(Workspace(tmp_path), 60):
        toolset.add(t)
    events, saved, seen = asyncio.run(drive(AgentRun(), toolset, [bad_edit] * 10, approve=None))
    assert len(seen) == 4  # stopped after the 4th identical failure
    assert "failed 2 times" in seen[2]["messages"][-1]["content"]
    assert events[-1]["type"] == "error" and "repeating" in events[-1]["message"]


def test_history_round_trip():
    stored = [
        {"role": "user", "content": "q", "meta": None},
        {
            "role": "assistant",
            "content": "",
            "meta": {
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "x", "arguments": "{}"}}
                ]
            },
        },
        {"role": "tool", "content": "out", "meta": {"tool_call_id": "c1", "name": "x", "ok": True}},
        {"role": "assistant", "content": "a", "meta": None},
    ]
    msgs = history_to_messages(stored)
    assert msgs[1]["tool_calls"][0]["id"] == "c1"
    assert msgs[2] == {"role": "tool", "tool_call_id": "c1", "content": "out"}


# ---- API ----


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


def test_workspaces_crud(client):
    assert client.post("/api/workspaces", headers=AUTH, json={"name": "proj"}).status_code == 201
    assert client.post("/api/workspaces", headers=AUTH, json={"name": "proj"}).status_code == 409
    assert client.post("/api/workspaces", headers=AUTH, json={"name": "../x"}).status_code == 400
    assert [w["name"] for w in client.get("/api/workspaces", headers=AUTH).json()] == ["proj"]
    assert client.delete("/api/workspaces/proj", headers=AUTH).status_code == 204


def test_agent_chat_streams_and_saves(client):
    seen: list[dict] = []
    client.app.state.http = httpx.AsyncClient(
        transport=httpx.MockTransport(scripted_model([WRITE_CALL, DONE_TURN], seen))
    )
    client.post("/api/workspaces", headers=AUTH, json={"name": "proj"})
    with client.stream(
        "POST",
        "/api/agent/chat",
        headers=AUTH,
        json={
            "model": "fake/m",
            "message": "make hello.txt",
            "mode": "agent",
            "workspace": "proj",
            "auto_approve": True,
        },
    ) as resp:
        events = [json.loads(line[6:]) for line in resp.iter_lines() if line.startswith("data: ")]

    assert events[0]["type"] == "start" and events[-1]["type"] == "done"
    assert any(e["type"] == "tool_result" and e["ok"] for e in events)
    ws_file = client.app.state.settings.data_dir / "workspaces" / "proj" / "hello.txt"
    assert ws_file.read_text() == "hi"
    system = seen[0]["messages"][0]["content"]
    assert "coding agent" in system and 'Do NOT prefix paths with "proj/"' in system

    conv = client.get(f"/api/conversations/{events[0]['conversation_id']}", headers=AUTH).json()
    assert [m["role"] for m in conv["messages"]] == ["user", "assistant", "tool", "assistant"]
    assert conv["settings"]["mode"] == "agent"


def test_agent_mode_requires_workspace(client):
    r = client.post(
        "/api/agent/chat", headers=AUTH, json={"model": "fake/m", "message": "x", "mode": "agent"}
    )
    assert r.status_code == 400


def test_model_without_tool_support(client):
    client.app.state.http = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(
                400, json={"error": {"message": "gemma3 does not support tools"}}
            )
        )
    )
    with client.stream(
        "POST",
        "/api/agent/chat",
        headers=AUTH,
        json={"model": "fake/m", "message": "x", "web": True},
    ) as resp:
        events = [json.loads(line[6:]) for line in resp.iter_lines() if line.startswith("data: ")]
    err = next(e for e in events if e["type"] == "error")
    assert "can't use tools" in err["message"]


def test_search_includes_page_excerpts(monkeypatch):
    from app.tools import web

    async def fake_search(query, settings, client, limit=8):
        return [web.SearchResult(f"T{i}", f"https://e.com/{i}", "snip") for i in range(5)]

    async def fake_fetch(url, settings, client):
        if url.endswith("/1"):
            raise WebError("blocked")
        return "PAGE " + url + " " + "x" * 5000

    monkeypatch.setattr(web, "search", fake_search)
    monkeypatch.setattr(web, "fetch", fake_fetch)
    out = asyncio.run(web.search_with_excerpts("q", Settings(), None))
    assert out.count("--- page excerpt ---") == 2  # top 3 read, one failed
    assert "PAGE https://e.com/0" in out and "PAGE https://e.com/1" not in out
    assert "[5] T4" in out and len(out) < 9000


def test_search_falls_back_down_the_chain():
    from app.tools import web

    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.host)
        if request.url.host == "searx.local":
            # SearXNG up, but every engine failed.
            return httpx.Response(
                200, json={"results": [], "unresponsive_engines": [["google", "CAPTCHA"]]}
            )
        html = '<a class="result__a" href="https://example.com/x">X</a>'
        return httpx.Response(200, text=html)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    settings = Settings(searxng_url="http://searx.local:8888")
    results = asyncio.run(web.search("q", settings, client))
    assert [r.url for r in results] == ["https://example.com/x"]
    assert calls == ["searx.local", "html.duckduckgo.com"]


def test_search_unavailable_message_is_actionable_for_the_model():
    from app.tools import web

    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(503)))
    with pytest.raises(WebError) as exc:
        asyncio.run(web.search("q", Settings(searxng_url="http://s:1"), client))
    msg = str(exc.value)
    assert "Don't retry" in msg and "fetch_url" in msg and "searxng" in msg and "duckduckgo" in msg


def test_provider_chain():
    from app.tools.web import provider_chain

    assert provider_chain(Settings()) == ["duckduckgo"]
    assert provider_chain(Settings(bundled_searxng=True, searxng_url="")) == [
        "searxng",
        "duckduckgo",
    ]
    assert provider_chain(Settings(searxng_url="http://s", brave_api_key="b")) == [
        "searxng",
        "brave",
        "duckduckgo",
    ]
    assert provider_chain(Settings(search_provider="brave", brave_api_key="b")) == [
        "brave",
        "duckduckgo",
    ]


def test_system_prompt_explains_date_and_web_access():
    from app.api.agent import AgentChatRequest, system_prompt

    no_web = system_prompt(AgentChatRequest(model="m", message="x"))
    assert "not a trick" in no_web and "do NOT have web access" in no_web
    with_web = system_prompt(AgentChatRequest(model="m", message="x", web=True))
    assert "You have web access" in with_web and "do NOT" not in with_web
