"""Connectors: the suite's apps (and any MCP server) whose tools Amy can use in chat.

A connector is an MCP endpoint with an API key. The hub registers each app it installs that
serves one (``PUT /api/connectors/{id}``); people can add others in Settings. Each chat loads
the tools of every enabled connector (cached briefly), named ``<connector>_<tool>`` so two apps
can't clash, and calling one forwards the call to the app.
"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field

import httpx

from app.mcp_client import McpClient, McpError, result_text
from app.tools import Tool
from app.tools.workspace import ToolError

log = logging.getLogger(__name__)

# How long a connector's tool list is reused before asking the app again.
CACHE_SECONDS = 60
# A connector that doesn't answer this fast is left out of the chat (it's retried next time).
LIST_TIMEOUT = 8
CALL_TIMEOUT = 120
ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,29}$")
_UNSAFE = re.compile(r"[^a-zA-Z0-9_-]")


@dataclass
class Status:
    """What a connector offered the last time it was asked."""

    checked: float = 0.0
    name: str = ""
    instructions: str = ""
    tools: list[dict] = field(default_factory=list)
    error: str | None = None


class Connectors:
    def __init__(self, state):
        self.state = state  # the app's state: its database and HTTP client
        self.status: dict[str, Status] = {}

    @property
    def db(self):
        return self.state.db

    @property
    def http(self) -> httpx.AsyncClient:
        return self.state.http

    def client(self, connector: dict, timeout: float = CALL_TIMEOUT) -> McpClient:
        return McpClient(connector["url"], connector.get("api_key"), self.http, timeout)

    async def refresh(self, connector: dict, force: bool = False) -> Status:
        """The connector's tools, from the cache unless it's stale (or ``force``)."""
        cid = connector["id"]
        status = self.status.get(cid)
        if status and not force and time.time() - status.checked < CACHE_SECONDS:
            return status
        try:
            info = await asyncio.wait_for(
                self.client(connector, LIST_TIMEOUT).describe(), LIST_TIMEOUT * 2
            )
            status = Status(time.time(), info["name"], info["instructions"], info["tools"])
        except (McpError, TimeoutError) as exc:
            log.info("Connector %s unavailable: %s", cid, exc or "timed out")
            status = Status(time.time(), error=str(exc) or "Timed out")
        self.status[cid] = status
        return status

    def forget(self, cid: str) -> None:
        self.status.pop(cid, None)

    async def for_chat(self) -> tuple[list[Tool], list[tuple[str, str]]]:
        """Tools from every enabled connector, and (name, instructions) per connected app."""
        enabled = [c for c in self.db.list_connectors() if c["enabled"]]
        statuses = await asyncio.gather(*(self.refresh(c) for c in enabled))
        tools: list[Tool] = []
        apps: list[tuple[str, str]] = []
        for connector, status in zip(enabled, statuses, strict=True):
            if status.error:
                continue
            apps.append((connector["name"], status.instructions))
            tools += [self.wrap(connector, t) for t in status.tools]
        return tools, apps

    def wrap(self, connector: dict, spec: dict) -> Tool:
        """An app's MCP tool as a chat tool."""
        remote = spec["name"]
        name = _UNSAFE.sub("_", f"{connector['id']}_{remote}")[:64]
        about = spec.get("description") or spec.get("title") or remote
        hints = spec.get("annotations") or {}
        # MCP's defaults: a tool may change things, and changes may be destructive.
        mutating = not hints.get("readOnlyHint", False) and hints.get("destructiveHint", True)
        schema = spec.get("inputSchema") or {"type": "object", "properties": {}}

        async def run(args: dict) -> str:
            try:
                result = await self.client(connector).call_tool(remote, args)
            except McpError as exc:
                raise ToolError(f"{connector['name']}: {exc}") from exc
            text = result_text(result)
            if result.get("isError"):
                raise ToolError(text)
            return text

        return Tool(
            name,
            f"{connector['name']}: {about}",
            schema,
            run,
            mutating=bool(mutating),
            summary_arg=_summary_arg(schema),
        )


def _summary_arg(schema: dict) -> str | None:
    """The argument that best says what a call is about: the first required string."""
    props = schema.get("properties") or {}
    strings = [k for k, v in props.items() if isinstance(v, dict) and v.get("type") == "string"]
    required = [k for k in schema.get("required") or [] if k in strings]
    return (required or strings or [None])[0]


def prompt(apps: list[tuple[str, str]]) -> str | None:
    """The system prompt's note on connected apps."""
    if not apps:
        return None
    lines = [
        "Connected apps: you can read and change the user's things in these apps with their "
        "tools (named after the app). Use them whenever the user refers to their documents, "
        "sheets, slides or other content in them, instead of guessing or asking them to paste it."
    ]
    for name, instructions in apps:
        lines.append(f"- {name}" + (f": {instructions.strip()}" if instructions.strip() else ""))
    return "\n".join(lines)
