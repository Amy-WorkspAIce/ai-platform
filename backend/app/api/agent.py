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

from app.agent import ModelError, history_to_messages, run_agent
from app.db import Database
from app.security import require_api_key
from app.tools import Toolset, Workspace, web_tools, workspace_tools

router = APIRouter(prefix="/api", dependencies=[Depends(require_api_key)])

WORKSPACE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

WEB_PROMPT = (
    "You can search the web with web_search and read pages with fetch_url. Use them for "
    "current events, prices, documentation, or anything you're unsure about, instead of "
    "guessing. Search results are only short snippets and are often outdated: for specific "
    "facts (versions, dates, numbers, quotes), open the most relevant results with fetch_url, "
    "preferring official sources, and answer from the page content. If sources disagree, say "
    "so. Cite the pages you used as markdown links."
)

AGENT_PROMPT = """You are Amy, an autonomous coding agent working in the workspace "{name}" \
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
Also available: git, node/npm/npx, ripgrep, curl.
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
    web: bool = False
    workspace: str | None = None
    auto_approve: bool = False


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


def system_prompt(req: AgentChatRequest, ws: Workspace | None = None) -> str:
    parts = [f"Today is {date.today():%A, %B %d, %Y}."]
    if req.mode == "agent" and ws is not None:
        listing = ws.list_files(".", depth=1)
        if len(listing) > 2000:
            listing = listing[:2000] + "\n..."
        parts.insert(0, AGENT_PROMPT.format(name=req.workspace, listing=listing))
    if req.web:
        parts.append(WEB_PROMPT)
    return "\n\n".join(parts)


@router.post("/agent/chat")
async def agent_chat(request: Request, req: AgentChatRequest) -> StreamingResponse:
    state = request.app.state
    resolved = state.providers.resolve(req.model)
    if resolved is None:
        raise HTTPException(404, f"Unknown model '{req.model}'")
    provider, upstream = resolved

    toolset = Toolset()
    ws: Workspace | None = None
    if req.mode == "agent":
        if not req.workspace:
            raise HTTPException(400, "Agent mode needs a workspace")
        ws = get_workspace(request, req.workspace)
        for tool in workspace_tools(ws, state.settings.agent_command_timeout):
            toolset.add(tool)
    if req.web:
        for tool in web_tools(state.settings, state.http):
            toolset.add(tool)

    db: Database = state.db
    settings = {"mode": req.mode, "web": req.web, "workspace": req.workspace}
    if req.conversation_id:
        conv = db.get_conversation(req.conversation_id)
        if conv is None:
            raise HTTPException(404, "Conversation not found")
        db.update_conversation(conv["id"], model=req.model, settings=settings)
        history = history_to_messages(conv["messages"])
    else:
        conv = db.create_conversation(title_from(req.message), req.model, settings)
        history = []
    conv_id = conv["id"]
    db.add_message(conv_id, "user", req.message)

    messages = [{"role": "system", "content": system_prompt(req, ws)}, *history]
    messages.append({"role": "user", "content": req.message})

    run = state.agent_runs.create(auto_approve=req.auto_approve)

    def persist(msg: dict) -> None:
        db.add_message(
            conv_id, msg["role"], msg["content"], msg.get("reasoning"), req.model, msg.get("meta")
        )

    async def work() -> None:
        try:
            await run_agent(
                run,
                provider=provider,
                model=upstream,
                client=state.http,
                messages=messages,
                toolset=toolset,
                persist=persist,
                max_steps=state.settings.agent_max_steps,
            )
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
