"""A small MCP client: tool listing and tool calls over the Streamable HTTP transport.

The suite's apps (Docs, Sheets, Slides...) each serve an MCP endpoint; the AI Platform uses
their tools in chat. Only what that needs is here: JSON-RPC over HTTP POST, answered as JSON or
as a server-sent event stream, with the optional ``Mcp-Session-Id``. Each operation opens its
own short session, so nothing needs to be kept alive between chats.
"""

import itertools
import json

import httpx

PROTOCOL_VERSION = "2025-06-18"
CLIENT_INFO = {"name": "amy-ai-platform", "version": "1"}


class McpError(Exception):
    """The server couldn't be reached, or answered with an error."""


class McpClient:
    def __init__(self, url: str, key: str | None, http: httpx.AsyncClient, timeout: float = 60):
        self.url = url
        self.key = key
        self.http = http
        self.timeout = timeout
        self._ids = itertools.count(1)

    def _headers(self, session: str | None, initialized: bool) -> dict[str, str]:
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        }
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        if session:
            headers["Mcp-Session-Id"] = session
        if initialized:
            headers["MCP-Protocol-Version"] = PROTOCOL_VERSION
        return headers

    async def _post(
        self, body: dict, session: str | None = None, initialized: bool = True
    ) -> tuple[dict | None, str | None]:
        """Send one JSON-RPC message; the response (None for a notification) and session id."""
        try:
            resp = await self.http.post(
                self.url,
                json=body,
                headers=self._headers(session, initialized),
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            raise McpError(f"Can't reach {self.url}: {exc}") from exc
        session = resp.headers.get("mcp-session-id") or session
        if resp.status_code == 401:
            raise McpError("The app rejected the API key.")
        if resp.status_code >= 400:
            raise McpError(f"The app returned an error ({resp.status_code}): {resp.text[:300]}")
        if "id" not in body or resp.status_code == 202:
            return None, session
        if resp.headers.get("content-type", "").startswith("text/event-stream"):
            message = _from_events(resp.text, body["id"])
        else:
            try:
                message = resp.json()
            except ValueError as exc:
                raise McpError(f"Not an MCP response: {resp.text[:200]}") from exc
        if not isinstance(message, dict):
            raise McpError("Not an MCP response.")
        if message.get("error"):
            error = message["error"]
            raise McpError(str(error.get("message", error)) if isinstance(error, dict) else error)
        return message.get("result") or {}, session

    async def _request(self, method: str, params: dict, session: str | None) -> dict:
        body = {"jsonrpc": "2.0", "id": next(self._ids), "method": method, "params": params}
        result, _ = await self._post(body, session)
        return result or {}

    async def _open(self) -> tuple[str | None, dict]:
        """Initialize a session; its id (if the server keeps sessions) and the server's info."""
        init = {
            "jsonrpc": "2.0",
            "id": next(self._ids),
            "method": "initialize",
            "params": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": CLIENT_INFO,
            },
        }
        info, session = await self._post(init, initialized=False)
        await self._post({"jsonrpc": "2.0", "method": "notifications/initialized"}, session)
        return session, info or {}

    async def describe(self) -> dict:
        """The server's info and instructions, and all its tools."""
        session, info = await self._open()
        tools: list[dict] = []
        cursor = None
        for _ in range(20):  # pages
            params = {"cursor": cursor} if cursor else {}
            result = await self._request("tools/list", params, session)
            tools += [t for t in result.get("tools") or [] if isinstance(t, dict) and t.get("name")]
            cursor = result.get("nextCursor")
            if not cursor:
                break
        return {
            "name": (info.get("serverInfo") or {}).get("name", ""),
            "instructions": info.get("instructions") or "",
            "tools": tools,
        }

    async def call_tool(self, name: str, arguments: dict) -> dict:
        """The tool's result: {"content": [...], "isError": bool, ...}."""
        session, _ = await self._open()
        return await self._request("tools/call", {"name": name, "arguments": arguments}, session)


def _from_events(text: str, request_id: int) -> dict | None:
    """The JSON-RPC response for ``request_id`` in a server-sent event stream."""
    data: list[str] = []
    for line in [*text.splitlines(), ""]:
        if line.startswith("data:"):
            data.append(line[5:].strip())
        elif not line.strip() and data:
            try:
                message = json.loads("\n".join(data))
            except ValueError:
                message = None
            data = []
            if isinstance(message, dict) and message.get("id") == request_id:
                return message
    return None


def result_text(result: dict) -> str:
    """A tool result as text for the model."""
    parts = []
    for item in result.get("content") or []:
        kind = item.get("type") if isinstance(item, dict) else None
        if kind == "text":
            parts.append(str(item.get("text", "")))
        elif kind == "resource" and isinstance(item.get("resource"), dict):
            parts.append(str(item["resource"].get("text") or item["resource"].get("uri", "")))
        elif kind == "resource_link":
            parts.append(f"{item.get('name', 'Link')}: {item.get('uri', '')}")
        elif kind:
            parts.append(f"[{kind}]")
    if not parts and result.get("structuredContent") is not None:
        parts.append(json.dumps(result["structuredContent"], ensure_ascii=False))
    return "\n".join(parts).strip() or "(no output)"
