"""Sub-agents: other models the chat hands work to.

That's the models ``ask_models`` consults, and the image, video and voice models behind the
creative tools. Each one is logged per chat (the ``subagents`` table) and streamed as a
``subagent`` event, so the chat's Details panel can list what each was asked and how it went.
"""

import asyncio
import dataclasses
from collections.abc import Awaitable, Callable
from typing import Any

from app import delegation
from app.tools import Tool

# Creative tools, by the kind of model doing the work (also the delegation capability).
MEDIA_KINDS = {
    "generate_image": "image",
    "edit_image": "image",
    "generate_video": "video",
    "speak": "speech",
}
TASK_CHARS = 2000
RESULT_CHARS = 20000


class Tracker:
    """Logs sub-agents for one chat. ``conversation_id`` and ``emit`` are set once known."""

    def __init__(self, db) -> None:
        self.db = db
        self.conversation_id: str | None = None
        self.emit: Callable[[dict], Awaitable[None]] | None = None

    async def run(
        self,
        kind: str,
        model: str,
        task: str,
        work: Callable[[], Awaitable[Any]],
        text: Callable[[Any], str] = str,
    ) -> Any:
        """Run ``work`` as a sub-agent; ``text`` turns its result into what the panel shows."""
        if not self.conversation_id:
            return await work()
        row = self.db.add_subagent(self.conversation_id, kind, model, task[:TASK_CHARS])
        await self._send(row)
        try:
            result = await work()
        except BaseException as exc:  # also the Stop button (cancellation)
            stopped = isinstance(exc, asyncio.CancelledError)
            reason = "Stopped" if stopped else str(exc) or type(exc).__name__
            row = self.db.finish_subagent(row["id"], "stopped" if stopped else "failed", reason)
            await self._send(row)
            raise
        row = self.db.finish_subagent(row["id"], "done", text(result)[:RESULT_CHARS])
        await self._send(row)
        return result

    async def _send(self, row: dict) -> None:
        if self.emit:
            await self.emit({"type": "subagent", **row})


def _content(result: Any) -> str:
    return str(result.get("content", "")) if isinstance(result, dict) else str(result)


def track_media(app, tool: Tool, tracker: Tracker) -> Tool:
    """The creative tool, logging the model it delegates to as a sub-agent."""
    kind = MEDIA_KINDS.get(tool.name)
    if kind is None:
        return tool
    inner = tool.run

    async def run(args: dict) -> Any:
        model = delegation.choose(app, kind, args.get("model"))
        if model is None:
            return await inner(args)  # it explains that no such model is installed
        task = str(args.get(tool.summary_arg or "") or tool.name)
        return await tracker.run(kind, model["name"], task, lambda: inner(args), _content)

    return dataclasses.replace(tool, run=run)
