"""The agent loop: stream a model turn, run the tools it calls, repeat.

Used both for chat with web access (web tools only) and agent mode (workspace tools,
optionally web). Tool calls use the OpenAI function-calling format, which Ollama and most
OpenAI-compatible servers support.

Events are pushed to an asyncio queue and streamed to the browser as server-sent events:

    {"type": "start", "run_id", "conversation_id"}
    {"type": "text" | "reasoning", "delta"}
    {"type": "tool_call", "id", "name", "arguments", "summary", "needs_approval"}
    {"type": "tool_result", "id", "ok", "output", "attachments"?}
    {"type": "error", "message"}
    {"type": "done"}
"""

import asyncio
import json
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import httpx

from app.context import fit
from app.providers import Provider
from app.tools import Toolset
from app.tools.web import WebError
from app.tools.workspace import ToolError, clip

log = logging.getLogger(__name__)

APPROVAL_TIMEOUT = 30 * 60
REPEAT_WARN = 2
REPEAT_STOP = 4
UI_OUTPUT_CHARS = 6000

Emit = Callable[[dict], Awaitable[None]]
Persist = Callable[[dict], None]


@dataclass
class AgentRun:
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    auto_approve: bool = False
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    _pending: dict[str, asyncio.Future] = field(default_factory=dict)

    async def emit(self, event: dict) -> None:
        await self.queue.put(event)

    async def approval(self, call_id: str) -> bool:
        if self.auto_approve:
            return True
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[call_id] = fut
        try:
            return await asyncio.wait_for(fut, timeout=APPROVAL_TIMEOUT)
        except TimeoutError:
            return False
        finally:
            self._pending.pop(call_id, None)

    def resolve(self, call_id: str, approve: bool, always: bool = False) -> bool:
        if always and approve:
            self.auto_approve = True
        fut = self._pending.get(call_id)
        if fut is None or fut.done():
            return False
        fut.set_result(approve)
        return True


class RunRegistry:
    def __init__(self) -> None:
        self.runs: dict[str, AgentRun] = {}

    def create(self, auto_approve: bool) -> AgentRun:
        run = AgentRun(auto_approve=auto_approve)
        self.runs[run.id] = run
        return run

    def get(self, run_id: str) -> AgentRun | None:
        return self.runs.get(run_id)

    def remove(self, run_id: str) -> None:
        self.runs.pop(run_id, None)


class ModelError(Exception):
    pass


class ToolsUnsupported(ModelError):
    """The model rejected a request because it can't call tools."""


async def run_agent(
    run: AgentRun,
    *,
    provider: Provider,
    model: str,
    client: httpx.AsyncClient,
    messages: list[dict],
    toolset: Toolset,
    persist: Persist,
    max_steps: int = 40,
    context_budget: int | None = None,
) -> None:
    """Run until the model answers without calling tools (or max_steps is reached)."""
    failures: dict[tuple[str, str], int] = {}  # identical failing calls, to break loops
    for _step in range(max_steps):
        if context_budget:
            # Tool results pile up between steps; trim old ones if we're near the limit.
            fit(messages, context_budget)
        try:
            content, reasoning, calls = await _stream_turn(
                run, provider, model, client, messages, toolset
            )
        except ToolsUnsupported:
            if not toolset:
                raise
            # Plain chat still works; just without web search or media creation.
            toolset = Toolset()
            await run.emit(
                {
                    "type": "notice",
                    "message": "This model can't use tools, so web search and image/video "
                    "creation are off. Pick a model tagged 'Tools' for those.",
                }
            )
            continue
        assistant: dict = {"role": "assistant", "content": content}
        if calls:
            assistant["tool_calls"] = [
                {
                    "id": c["id"],
                    "type": "function",
                    "function": {"name": c["name"], "arguments": c["arguments"]},
                }
                for c in calls
            ]
        messages.append(assistant)
        persist(
            {
                "role": "assistant",
                "content": content,
                "reasoning": reasoning or None,
                "meta": {"tool_calls": assistant["tool_calls"]} if calls else None,
            }
        )
        if not calls:
            return
        for call in calls:
            result = await _run_tool(run, toolset, call)
            if not result["ok"]:
                key = (call["name"], call["arguments"])
                failures[key] = failures.get(key, 0) + 1
                if failures[key] >= REPEAT_STOP:
                    await run.emit(
                        {
                            "type": "error",
                            "message": "Stopped: the agent kept repeating a failing action. "
                            "Try rephrasing the task or giving it a hint.",
                        }
                    )
                    return
                if failures[key] >= REPEAT_WARN:
                    result["content"] += (
                        f"\n\n[This exact call has now failed {failures[key]} times. Do not "
                        "repeat it: try a different approach, or explain the problem to the user.]"
                    )
            messages.append(
                {"role": "tool", "tool_call_id": call["id"], "content": result["content"]}
            )
            meta = {"tool_call_id": call["id"], "name": call["name"], "ok": result["ok"]}
            if result.get("attachments"):
                meta["attachments"] = result["attachments"]
            persist({"role": "tool", "content": result["content"], "meta": meta})
    await run.emit(
        {
            "type": "error",
            "message": f"Stopped after {max_steps} steps. Say 'continue' to keep going.",
        }
    )


async def _stream_turn(
    run: AgentRun,
    provider: Provider,
    model: str,
    client: httpx.AsyncClient,
    messages: list[dict],
    toolset: Toolset,
) -> tuple[str, str, list[dict]]:
    payload: dict = {"model": model, "messages": messages, "stream": True}
    if toolset:
        payload["tools"] = toolset.schemas()
    content, reasoning = "", ""
    calls: dict[int, dict] = {}
    req = client.build_request(
        "POST", provider.url("chat/completions"), json=payload, headers=provider.headers()
    )
    try:
        resp = await client.send(req, stream=True)
    except httpx.HTTPError as exc:
        raise ModelError(f"Model server unreachable: {exc}") from exc
    try:
        if resp.status_code >= 400:
            await resp.aread()
            message = _error_message(resp)
            if "can't use tools" in message:
                raise ToolsUnsupported(message)
            raise ModelError(message)
        async for line in resp.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            choice = (json.loads(data).get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            if delta.get("content"):
                content += delta["content"]
                await run.emit({"type": "text", "delta": delta["content"]})
            think = delta.get("reasoning") or delta.get("reasoning_content")
            if think:
                reasoning += think
                await run.emit({"type": "reasoning", "delta": think})
            for tc in delta.get("tool_calls") or []:
                entry = calls.setdefault(
                    tc.get("index", len(calls)), {"id": "", "name": "", "arguments": ""}
                )
                entry["id"] = tc.get("id") or entry["id"]
                fn = tc.get("function") or {}
                if fn.get("name") and not entry["name"]:
                    entry["name"] = fn["name"]
                args = fn.get("arguments")
                if isinstance(args, dict):  # some servers send parsed objects
                    args = json.dumps(args)
                entry["arguments"] += args or ""
    finally:
        await resp.aclose()
    ordered = [calls[i] for i in sorted(calls)]
    for n, c in enumerate(ordered):
        c["id"] = c["id"] or f"call_{uuid.uuid4().hex[:12]}_{n}"
    return content, reasoning, ordered


def _error_message(resp: httpx.Response) -> str:
    try:
        body = resp.json()
        err = body.get("error")
        msg = err.get("message") if isinstance(err, dict) else err or body.get("detail")
    except ValueError:
        msg = resp.text
    msg = str(msg or f"HTTP {resp.status_code}")
    if "does not support tools" in msg:
        return (
            "This model can't use tools (web access, agent mode). Pick a model tagged "
            "'Tools' on the Models page, e.g. Qwen 3, gpt-oss or Qwen3-Coder."
        )
    return msg


async def _run_tool(run: AgentRun, toolset: Toolset, call: dict) -> dict:
    tool = toolset.tools.get(call["name"])
    try:
        args = json.loads(call["arguments"] or "{}")
        if not isinstance(args, dict):
            raise ValueError("arguments must be an object")
    except ValueError as exc:
        args = None
        error = f"Invalid JSON arguments: {exc}. Call the tool again with valid JSON."
    needs_approval = bool(tool and tool.mutating and not run.auto_approve)
    await run.emit(
        {
            "type": "tool_call",
            "id": call["id"],
            "name": call["name"],
            "arguments": args if args is not None else call["arguments"],
            "summary": str((args or {}).get(tool.summary_arg, ""))
            if tool and tool.summary_arg
            else "",
            "needs_approval": needs_approval and args is not None,
        }
    )
    if tool is None:
        return await _result(run, call, False, f"Unknown tool '{call['name']}'.")
    if args is None:
        return await _result(run, call, False, error)
    if needs_approval and not await run.approval(call["id"]):
        return await _result(
            run, call, False, "The user denied this action. Ask how they'd like to proceed."
        )
    started = time.time()
    attachments: list[dict] = []
    try:
        output = await tool.run(args)
        if isinstance(output, dict):  # text for the model + media for the user
            attachments = output.get("attachments") or []
            output = output["content"]
        ok = True
    except KeyError as exc:
        output, ok = f"Error: missing required argument {exc}. Call the tool again with it.", False
    except (ToolError, WebError, TypeError, ValueError, OSError) as exc:
        output, ok = f"Error: {exc}", False
    log.info("tool %s ok=%s %.1fs", call["name"], ok, time.time() - started)
    return await _result(run, call, ok, output, attachments)


async def _result(
    run: AgentRun, call: dict, ok: bool, content: str, attachments: list[dict] | None = None
) -> dict:
    event = {
        "type": "tool_result",
        "id": call["id"],
        "ok": ok,
        "output": clip(content, UI_OUTPUT_CHARS),
    }
    if attachments:
        event["attachments"] = attachments
    await run.emit(event)
    return {"ok": ok, "content": content, "attachments": attachments or []}


def history_with_ids(
    stored: list[dict], after_id: str | None = None
) -> tuple[list[dict], list[str]]:
    """Like history_to_messages, starting after ``after_id`` (the summarized part), and also
    returning the stored message id behind each rebuilt message."""
    if after_id:
        idx = next((i for i, m in enumerate(stored) if m["id"] == after_id), None)
        if idx is not None:
            stored = stored[idx + 1 :]
    out: list[dict] = []
    ids: list[str] = []
    for m in stored:
        meta = m.get("meta") or {}
        if m["role"] == "tool":
            if out and (out[-1].get("tool_calls") or out[-1].get("role") == "tool"):
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": meta.get("tool_call_id", ""),
                        "content": m["content"],
                    }
                )
                ids.append(m["id"])
            continue
        msg: dict = {"role": m["role"], "content": m["content"]}
        if meta.get("tool_calls"):
            msg["tool_calls"] = meta["tool_calls"]
        out.append(msg)
        ids.append(m["id"])
    # A leading tool result without its call would be rejected by the model server.
    while out and out[0]["role"] == "tool":
        out.pop(0)
        ids.pop(0)
    return out, ids


def history_to_messages(stored: list[dict]) -> list[dict]:
    """Rebuild OpenAI-format messages (including tool calls) from saved messages."""
    out: list[dict] = []
    for m in stored:
        meta = m.get("meta") or {}
        if m["role"] == "tool":
            if out and out[-1].get("tool_calls"):
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": meta.get("tool_call_id", ""),
                        "content": m["content"],
                    }
                )
            continue
        msg: dict = {"role": m["role"], "content": m["content"]}
        if meta.get("tool_calls"):
            msg["tool_calls"] = meta["tool_calls"]
        out.append(msg)
    return out
