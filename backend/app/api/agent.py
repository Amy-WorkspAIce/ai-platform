"""Tool-using chat: web access and agent mode (``/api/agent``), plus workspaces.

``POST /api/agent/chat`` streams server-sent events (see ``app.agent``) and saves the
conversation as it goes, including tool calls and results.
"""

import asyncio
import json
import re
import shutil
import time
from datetime import date
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app import context, memory, routing
from app import rules as rulesets
from app.agent import ModelError, history_with_ids, run_agent
from app.db import Database
from app.security import require_api_key
from app.tools import Toolset, Workspace, web_tools, workspace_tools
from app.tools.creative import creative_prompt, creative_tools

router = APIRouter(prefix="/api", dependencies=[Depends(require_api_key)])

# Memory updates still running after their chat stream closed (kept so they aren't collected).
_learning: set[asyncio.Task] = set()

WORKSPACE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

# Every chat gets this. Without it, models treat the date as a puzzle ("my training ends in
# 2024, is the user testing me?") and spend their thinking arguing about it.
BASE_PROMPT = """You are Amy, an AI assistant running on the user's own server.

The current date is {today}. Your training data ends before this date; that's expected and \
not a trick. Trust this date, and for anything that may have changed since your training \
(news, prices, software versions, weather, schedules), rely on your tools when you have them.

Answer simple questions directly. Use markdown when it helps readability."""

WEB_PROMPT = """You have web access in this chat:
- web_search searches the web and includes excerpts from the top pages.
- fetch_url reads one page in full.
- get_weather gives current weather and the forecast for a place (use it for any weather \
question; a city and state or country is enough, never ask for more).
Use them for live or recent information instead of guessing, then answer from what you found \
and cite the pages as markdown links. Prefer official sources; if sources disagree, say so.
If a tool fails, don't keep retrying it. Try one other approach (e.g. fetch_url on a site you \
know) or tell the user what didn't work."""

MEMORY_PROMPT = """You have long-term memory. After each reply, facts the user shares about \
themselves (who they are, people and pets in their life, preferences, projects, anything they \
ask you to remember) are saved automatically, and relevant ones are given to you in later \
chats. The user can see and edit them on the Memory page. So never say you can't remember \
things between chats; when asked to remember something, say you'll remember it."""

NO_WEB_PROMPT = """You do NOT have web access (it's turned off in Settings). If the user needs \
live or recent information, say so briefly and help with what you do know."""

AGENT_PROMPT = """You are working as an autonomous coding agent in the workspace "{name}" \
on the user's server.

The workspace root is the current directory ("."). All paths are relative to it. Do NOT prefix \
paths with "{name}/". Current top-level contents:
{listing}

How to work:
- Look before you change: list and read the relevant files first. Never invent file contents.
- Make focused changes that match the existing code style. Prefer edit_file for small changes \
and write_file for new files.
- Verify your work by running the tests, build or program, and fix what fails.
- Each run_command starts a NEW non-interactive bash shell in the workspace root, so `cd`, \
`source` and exported variables do not carry over. Chain steps with && in one command.
- Python: run `python3 -m venv .venv` once. After that, every command automatically uses \
.venv, so plain `pip install ...`, `python ...` and `pytest` just work (no need to activate). \
Also available: git, node/npm/npx, ripgrep, curl, wget, jq, unzip, make, gcc/g++.
- Environment: commands run inside the platform's container as an unprivileged user (no root, \
no sudo, no apt-get), with internet access. To get other tools: pip (into .venv), npm or \
`npm install -g` (installs to ~/.local), or download a prebuilt binary with curl into \
~/.local/bin (on PATH, kept between runs). The workspace, your home directory and /tmp are \
writable. If something needs a system package, tell the user which one rather than guessing. \
Never conclude there's no network without testing it (e.g. `curl -sSI https://example.com`).
- A failed command shows its exit code. Read the error and change your approach; don't repeat \
the same command hoping for a different result.
- Never claim something works (tests pass, build succeeds) unless a command's output shows it.
- If the request is ambiguous or risky (deleting data, force-pushing), ask first instead \
of guessing. Some actions need the user's approval; if one is denied, ask how to proceed.
- When done, summarize what you changed and how you verified it."""


class AgentChatRequest(BaseModel):
    model: str
    message: str = Field(min_length=1, max_length=100_000)
    conversation_id: str | None = None
    mode: Literal["chat", "agent"] = "chat"
    # None = follow the "web access" setting (on by default); False turns it off for this chat.
    web: bool | None = None
    workspace: str | None = None
    auto_approve: bool = False
    folder_id: str | None = None  # for a new chat: create it inside this folder


class ApprovalRequest(BaseModel):
    call_id: str
    approve: bool
    always: bool = False


class WorkspaceCreate(BaseModel):
    name: str
    git_url: str | None = None


def workspaces_root(request: Request) -> Path:
    root = request.app.state.settings.data_dir / "workspaces"
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_workspace(request: Request, name: str) -> Workspace:
    if not WORKSPACE_NAME.match(name):
        raise HTTPException(400, "Invalid workspace name")
    path = workspaces_root(request) / name
    if not path.is_dir():
        raise HTTPException(404, f"Workspace '{name}' not found")
    settings = request.app.state.settings
    home = settings.data_dir / "agent-home"
    home.mkdir(parents=True, exist_ok=True)
    return Workspace(path, github_token=settings.github_token, home=home)


def system_prompt(
    req: AgentChatRequest,
    ws: Workspace | None = None,
    web: bool | None = None,
    creative: str | None = None,
    memories: str | None = None,
    memory_on: bool = False,
    rules: str | None = None,
) -> str:
    parts = [BASE_PROMPT.format(today=f"{date.today():%A, %B %d, %Y}")]
    if req.mode == "agent" and ws is not None:
        listing = ws.list_files(".", depth=1)
        if len(listing) > 2000:
            listing = listing[:2000] + "\n..."
        parts.append(AGENT_PROMPT.format(name=req.workspace, listing=listing))
    web = req.web if web is None else web
    parts.append(WEB_PROMPT if web else NO_WEB_PROMPT)
    if creative:
        parts.append(creative)
    if rules:
        parts.append(rules)
    if memory_on:
        parts.append(MEMORY_PROMPT)
    if memories:
        parts.append(memories)
    return "\n\n".join(parts)


@router.post("/agent/chat")
async def agent_chat(request: Request, req: AgentChatRequest) -> StreamingResponse:
    state = request.app.state
    db: Database = state.db
    model, purpose = req.model, None
    if req.model == routing.AUTO:
        # Auto: coding work goes to the coding model, everything else to the chat model. A chat
        # that turned to code stays with the coding model for its follow-ups.
        prior = db.get_conversation(req.conversation_id) if req.conversation_id else None
        was_coding = (((prior or {}).get("settings") or {}).get("purpose")) == "coding"
        coding = req.mode == "agent" or was_coding or routing.looks_like_code(req.message)
        purpose = "coding" if coding else "chat"
        model = await routing.pick(request.app, purpose)
        if model is None:
            raise HTTPException(404, "No chat model is installed yet")
    resolved = state.providers.resolve(model)
    if resolved is None:
        raise HTTPException(404, f"Unknown model '{model}'")
    provider, upstream = resolved

    toolset = Toolset()
    ws: Workspace | None = None
    if req.mode == "agent":
        if not req.workspace:
            raise HTTPException(400, "Agent mode needs a workspace")
        ws = get_workspace(request, req.workspace)
        for tool in workspace_tools(ws, state.settings.agent_command_timeout):
            toolset.add(tool)
    web = state.user_settings.web_access and req.web is not False
    if web:
        for tool in web_tools(state.settings, state.http):
            toolset.add(tool)
    # Delegation: image/video/speech tools for whatever media models are installed.
    media_tools = creative_tools(request.app)
    for tool in media_tools:
        toolset.add(tool)

    settings = {"mode": req.mode, "web": web, "workspace": req.workspace, "purpose": purpose}
    summary: str | None = None
    summary_upto: str | None = None
    if req.conversation_id:
        conv = db.get_conversation(req.conversation_id)
        if conv is None:
            raise HTTPException(404, "Conversation not found")
        db.update_conversation(conv["id"], model=req.model, settings=settings)
        summary, summary_upto = conv.get("summary"), conv.get("summary_upto")
        history, history_ids = history_with_ids(conv["messages"], summary_upto)
    else:
        conv = db.create_conversation(title_from(req.message), req.model, settings)
        if req.folder_id and req.folder_id in {f["id"] for f in db.list_folders()}:
            db.update_conversation(conv["id"], folder_id=req.folder_id)
            conv["folder_id"] = req.folder_id
        history, history_ids = [], []
    conv_id = conv["id"]
    db.add_message(conv_id, "user", req.message)

    us = state.user_settings
    rules = rulesets.effective(db, conv.get("folder_id"), ws and req.workspace, req.mode)
    project = rulesets.project_rules(ws.root) if ws else []
    # A panel of other models, only when a rule asks for one: offered unasked, small models
    # consult it for everything.
    if rulesets.wants_panel(rules, project):
        others = await rulesets.chat_model_ids(request.app)
        if len(others) > 1:
            toolset.add(rulesets.ask_models_tool(request.app, others))
    recalled = await memory.recall(request.app, req.message) if us.memory_enabled else []
    system = system_prompt(
        req,
        ws,
        web,
        creative_prompt(media_tools),
        memory.recall_prompt(recalled),
        memory_on=us.memory_enabled and us.memory_auto,
        rules=rulesets.prompt(rules, project),
    )
    # What the user said earlier in this chat, so "remember that" can refer back to it.
    earlier = [m["content"] for m in history if m.get("role") == "user" and m.get("content")]

    # Context budget for the conversation itself: window minus system prompt, tool
    # definitions, the new message and room for the reply.
    window = context.context_size(state.settings)
    fixed = context.estimate_tokens(system) + context.estimate_tokens(toolset.schemas())
    budget = window - context.RESERVE_FOR_REPLY - fixed - context.estimate_tokens(req.message)
    context.trim_tool_results(history)

    run = state.agent_runs.create(auto_approve=req.auto_approve)
    replies: list[str] = []

    def persist(msg: dict) -> None:
        if msg["role"] == "assistant" and msg.get("content"):
            replies.append(msg["content"])
        db.add_message(
            conv_id, msg["role"], msg["content"], msg.get("reasoning"), model, msg.get("meta")
        )

    async def condense() -> None:
        """Fold the oldest messages into the running summary when history doesn't fit."""
        nonlocal history, summary
        if context.message_tokens(history) + context.estimate_tokens(summary or "") <= budget:
            return
        cut = context.split_point(history, int(budget * context.SUMMARY_TARGET))
        if cut <= 0:
            return
        await run.emit(
            {
                "type": "notice",
                "message": "Condensing earlier messages to stay within the model's context…",
            }
        )
        new_summary = await context.summarize(
            state.http, provider, upstream, history[:cut], summary, window
        )
        if new_summary is None:
            return  # fall back to trimming (below) and the server's own truncation
        summary = new_summary
        db.update_conversation(conv_id, summary=summary, summary_upto=history_ids[cut - 1])
        history = history[cut:]

    async def learn() -> None:
        """Update long-term memory from this exchange (after the answer is shown)."""
        if not (us.memory_enabled and us.memory_auto and replies):
            return
        # Coding models follow the memory instructions poorly (they "remember" facts about the
        # code), so a general chat model does this when one is installed.
        mem_provider, mem_model = provider, upstream
        if routing.is_coder(model):
            general = await routing.pick(request.app, "chat")
            found = state.providers.resolve(general) if general else None
            if found and not routing.is_coder(general):
                mem_provider, mem_model = found
        changes = await memory.extract(
            request.app,
            provider=mem_provider,
            model=mem_model,
            user_message=req.message,
            assistant_reply=replies[-1],
            conversation_id=conv_id,
            earlier=earlier[-memory.EARLIER_TURNS :],
        )
        for change in changes:
            await run.emit({"type": "memory", **change})

    async def work() -> None:
        try:
            await condense()
            messages = [{"role": "system", "content": system}]
            if summary:
                messages.append(context.summary_message(summary))
            messages += history
            messages.append({"role": "user", "content": req.message})
            context.fit(
                messages,
                window - context.RESERVE_FOR_REPLY - context.estimate_tokens(toolset.schemas()),
            )
            await run_agent(
                run,
                provider=provider,
                model=upstream,
                client=state.http,
                messages=messages,
                toolset=toolset,
                persist=persist,
                max_steps=state.settings.agent_max_steps,
                context_budget=window
                - context.RESERVE_FOR_REPLY
                - context.estimate_tokens(toolset.schemas()),
            )
            await run.emit({"type": "answer_done"})
            # Leaving the chat closes the stream and cancels this run; the memory update (slow
            # with reasoning models) must survive that, so it runs as its own task.
            task = asyncio.create_task(learn())
            _learning.add(task)
            task.add_done_callback(_learning.discard)
            await asyncio.shield(task)
        except ModelError as exc:
            await run.emit({"type": "error", "message": str(exc)})
        except Exception as exc:  # surface anything unexpected to the user
            await run.emit({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
            raise
        finally:
            await run.emit({"type": "done"})

    async def stream():
        task = asyncio.create_task(work())
        try:
            yield sse({"type": "start", "run_id": run.id, "conversation_id": conv_id})
            if purpose:
                yield sse({"type": "model", "model": model, "purpose": purpose})
            while True:
                event = await run.queue.get()
                yield sse(event)
                if event["type"] == "done":
                    break
        finally:
            # Client disconnected (Stop button) or run finished: stop any running tool.
            task.cancel()
            state.agent_runs.remove(run.id)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/agent/runs/{run_id}/approve")
async def approve(request: Request, run_id: str, body: ApprovalRequest) -> dict:
    run = request.app.state.agent_runs.get(run_id)
    if run is None:
        raise HTTPException(404, "Run not found (it may have finished)")
    if not run.resolve(body.call_id, body.approve, body.always):
        raise HTTPException(409, "Nothing is waiting for approval with that id")
    return {"ok": True, "auto_approve": run.auto_approve}


def sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


def title_from(text: str) -> str:
    line = text.strip().split("\n")[0].strip()
    return line if len(line) <= 60 else line[:57] + "…"


# ---- workspaces ----


async def _git(args: list[str], cwd: Path, ws_env: dict, timeout: float = 15) -> str | None:
    proc = await asyncio.create_subprocess_exec(
        "git",
        *args,
        cwd=cwd,
        env=ws_env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout)
    except TimeoutError:
        proc.kill()
        return None
    return out.decode(errors="replace").strip() if proc.returncode == 0 else None


@router.get("/workspaces")
async def list_workspaces(request: Request) -> list[dict]:
    root = workspaces_root(request)
    out = []
    for d in sorted(p for p in root.iterdir() if p.is_dir() and WORKSPACE_NAME.match(p.name)):
        ws = get_workspace(request, d.name)
        info: dict = {"name": d.name, "modified": d.stat().st_mtime, "git": None}
        if (d / ".git").exists():
            env = ws.env()
            branch = await _git(["rev-parse", "--abbrev-ref", "HEAD"], d, env)
            remote = await _git(["config", "--get", "remote.origin.url"], d, env)
            dirty = await _git(["status", "--porcelain"], d, env)
            info["git"] = {
                "branch": branch,
                "remote": remote,
                "changes": len((dirty or "").splitlines()),
            }
        out.append(info)
    return out


@router.post("/workspaces", status_code=201)
async def create_workspace(request: Request, body: WorkspaceCreate) -> dict:
    if not WORKSPACE_NAME.match(body.name):
        raise HTTPException(400, "Names may contain letters, numbers, '.', '_' and '-'")
    path = workspaces_root(request) / body.name
    if path.exists():
        raise HTTPException(409, f"Workspace '{body.name}' already exists")
    if not body.git_url:
        path.mkdir()
        return {"name": body.name}
    if not re.match(r"^(https://|git@)[\w.@:/~+-]+$", body.git_url):
        raise HTTPException(400, "Use an https:// or git@ repository URL")
    settings = request.app.state.settings
    env = Workspace(
        workspaces_root(request), settings.github_token, settings.data_dir / "agent-home"
    ).env()
    started = time.time()
    proc = await asyncio.create_subprocess_exec(
        "git",
        "clone",
        "--",
        body.git_url,
        str(path),
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), 900)
    except TimeoutError:
        proc.kill()
        shutil.rmtree(path, ignore_errors=True)
        raise HTTPException(504, "git clone timed out") from None
    if proc.returncode != 0:
        shutil.rmtree(path, ignore_errors=True)
        lines = out.decode(errors="replace").strip().splitlines()
        raise HTTPException(400, f"git clone failed: {lines[-1] if lines else 'unknown error'}")
    return {"name": body.name, "seconds": round(time.time() - started, 1)}


@router.delete("/workspaces/{name}", status_code=204)
async def delete_workspace(request: Request, name: str) -> None:
    ws = get_workspace(request, name)
    await asyncio.to_thread(shutil.rmtree, ws.root)
    request.app.state.db.delete_scope(rulesets.workspace_scope(name))
