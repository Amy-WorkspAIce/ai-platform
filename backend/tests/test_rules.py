import json

import httpx

from tests.test_routing import AUTH, client  # noqa: F401 - fixture


def add(c, name, text, scope="global", applies="always"):
    body = {"name": name, "text": text, "scope": scope, "applies": applies}
    r = c.post("/api/rules", headers=AUTH, json=body)
    assert r.status_code == 201, r.text
    return r.json()


def system_prompt_for(c, **extra):
    """Send one chat and return the system prompt the model got."""
    seen = []
    inner = c.server

    def spy(request):
        if request.content and json.loads(request.content).get("stream"):
            seen.append(json.loads(request.content))
        return inner(request)

    c.app.state.http = httpx.AsyncClient(transport=httpx.MockTransport(spy))
    body = {"model": "fake/llama3.1:8b", "message": "hi", **extra}
    with c.stream("POST", "/api/agent/chat", headers=AUTH, json=body) as resp:
        resp.read()
    return seen[-1]["messages"][0]["content"], {
        t["function"]["name"] for t in seen[-1].get("tools") or []
    }


def test_rules_follow_scope_and_can_be_turned_off_per_folder(client):  # noqa: F811
    panel = add(client, "Deliberate", "For every task, ask three models with ask_models first.")
    work = client.post("/api/folders", headers=AUTH, json={"name": "Work"}).json()
    home = client.post("/api/folders", headers=AUTH, json={"name": "Home"}).json()
    add(
        client,
        "No force-push",
        "When creating PRs, never force-push.",
        scope=f"folder:{work['id']}",
    )

    prompt, tools = system_prompt_for(client, folder_id=home["id"])
    assert "Deliberate: For every task" in prompt and "force-push" not in prompt
    assert "ask_models" in tools  # two models installed, so there's a panel to ask

    prompt, _ = system_prompt_for(client, folder_id=work["id"])
    assert "Deliberate" in prompt and "No force-push: When creating PRs" in prompt
    assert prompt.index("Deliberate") < prompt.index("No force-push")  # specific ones last

    # The Work folder opts out of the global rule; other folders keep it.
    scope = f"folder:{work['id']}"
    r = client.put(
        f"/api/rules/{panel['id']}/excluded", headers=AUTH, json={"scope": scope, "excluded": True}
    )
    assert r.status_code == 200
    listing = client.get("/api/rules", headers=AUTH, params={"scope": scope}).json()
    assert [r["name"] for r in listing["rules"]] == ["No force-push"]
    assert listing["inherited"][0]["excluded"] is True
    prompt, _ = system_prompt_for(client, folder_id=work["id"])
    assert "Deliberate" not in prompt and "No force-push" in prompt
    assert "Deliberate" in system_prompt_for(client, folder_id=home["id"])[0]

    # A switched-off rule applies nowhere; a deleted folder takes its rules along.
    client.patch(f"/api/rules/{panel['id']}", headers=AUTH, json={"enabled": False})
    prompt, tools = system_prompt_for(client)
    assert "Deliberate" not in prompt
    assert "ask_models" not in tools  # no rule asks for a panel, so none is offered
    client.delete(f"/api/folders/{work['id']}", headers=AUTH)
    assert client.get("/api/rules", headers=AUTH, params={"scope": scope}).status_code == 404
    assert [r["name"] for r in client.get("/api/rules", headers=AUTH).json()["rules"]] == [
        "Deliberate"
    ]


def test_rules_can_target_chats_or_workspace_tasks(client):  # noqa: F811
    add(client, "Brief", "Keep chat answers short.", applies="chat")
    add(client, "Tests", "Run the tests before saying you're done.", applies="agent")
    assert client.post("/api/workspaces", headers=AUTH, json={"name": "proj"}).status_code == 201
    add(client, "Style", "Use tabs.", scope="workspace:proj")

    prompt, _ = system_prompt_for(client)
    assert "Brief" in prompt and "Tests" not in prompt and "Style" not in prompt
    prompt, _ = system_prompt_for(client, mode="agent", workspace="proj")
    assert "Brief" not in prompt and "Tests" in prompt and "Style: Use tabs." in prompt


def test_workspace_rule_files_are_read(client, tmp_path):  # noqa: F811
    assert client.post("/api/workspaces", headers=AUTH, json={"name": "proj"}).status_code == 201
    root = tmp_path / "workspaces" / "proj"
    (root / "AGENTS.md").write_text("Always use pnpm, never npm.")
    (root / ".cursor" / "rules").mkdir(parents=True)
    (root / ".cursor" / "rules" / "db.mdc").write_text("Migrations go in db/migrations.")

    listing = client.get("/api/rules", headers=AUTH, params={"scope": "workspace:proj"}).json()
    assert [f["name"] for f in listing["project"]] == ["AGENTS.md", ".cursor/rules/db.mdc"]
    prompt, _ = system_prompt_for(client, mode="agent", workspace="proj")
    assert "never npm" in prompt and "db/migrations" in prompt


def test_bad_scopes_and_exclusions_are_refused(client):  # noqa: F811
    body = {"name": "x", "text": "y", "scope": "folder:nope"}
    assert client.post("/api/rules", headers=AUTH, json=body).status_code == 404
    rule = add(client, "x", "y")
    folder = client.post("/api/folders", headers=AUTH, json={"name": "F"}).json()
    local = add(client, "z", "w", scope=f"folder:{folder['id']}")
    off = {"scope": f"folder:{folder['id']}", "excluded": True}
    assert (
        client.put(f"/api/rules/{local['id']}/excluded", headers=AUTH, json=off).status_code == 400
    )
    glob = {"scope": "global", "excluded": True}
    assert (
        client.put(f"/api/rules/{rule['id']}/excluded", headers=AUTH, json=glob).status_code == 400
    )


def test_ask_models_puts_one_question_to_several_models(client):  # noqa: F811
    import asyncio

    from app.rules import ask_models_tool

    asked = []

    def server(request):
        body = json.loads(request.content)
        asked.append(body["model"])
        reply = f"{body['model']} says yes"
        return httpx.Response(200, json={"choices": [{"message": {"content": reply}}]})

    client.app.state.http = httpx.AsyncClient(transport=httpx.MockTransport(server))
    tool = ask_models_tool(client.app, ["fake/llama3.1:8b", "fake/qwen2.5-coder:7b"])
    out = asyncio.run(
        tool.run({"question": "Ship it?", "models": ["llama3.1:8b", "qwen2.5-coder:7b", "gpt-9"]})
    )
    assert sorted(asked) == ["llama3.1:8b", "qwen2.5-coder:7b"]
    assert "## fake/llama3.1:8b\nllama3.1:8b says yes" in out
    assert "Not installed: gpt-9" in out


def test_ask_models_logs_each_model_as_a_sub_agent(client):  # noqa: F811
    import asyncio

    from app.rules import ask_models_tool
    from app.subagents import Tracker

    def server(request):
        body = json.loads(request.content)
        if body["model"] == "broken:1b":
            return httpx.Response(500, json={"error": "model crashed"})
        return httpx.Response(200, json={"choices": [{"message": {"content": "Yes, ship it"}}]})

    client.app.state.http = httpx.AsyncClient(transport=httpx.MockTransport(server))
    db = client.app.state.db
    conv = db.create_conversation("Ship?", "fake/m", {})
    tracker, events = Tracker(db), []
    tracker.conversation_id = conv["id"]

    async def emit(event):
        events.append(event)

    tracker.emit = emit
    tool = ask_models_tool(client.app, ["fake/llama3.1:8b", "fake/broken:1b"], tracker)
    out = asyncio.run(tool.run({"question": "Ship it?", "models": ["llama3.1:8b", "broken:1b"]}))
    assert "Yes, ship it" in out and "(failed:" in out
    rows = {r["model"]: r for r in db.list_subagents(conv["id"])}
    assert rows["fake/llama3.1:8b"]["status"] == "done"
    assert rows["fake/llama3.1:8b"]["result"] == "Yes, ship it"
    assert (
        rows["fake/broken:1b"]["status"] == "failed" and rows["fake/llama3.1:8b"]["kind"] == "ask"
    )
    assert all(r["task"] == "Ship it?" and r["finished_at"] for r in rows.values())
    # Each is streamed when it starts and when it finishes.
    assert [e["status"] for e in events].count("running") == 2 and len(events) == 4
    assert client.get(f"/api/conversations/{conv['id']}/details", headers=AUTH).json()["subagents"]
