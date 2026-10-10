"""Tools the model can call during a chat (web access) or an agent run (workspace)."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import httpx

from app.config import Settings
from app.tools import web
from app.tools.workspace import ToolError, Workspace


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    run: Callable[[dict], Awaitable[str]]
    # Changes state (files, commands): needs approval unless auto-approve is on.
    mutating: bool = False
    # For the UI: which argument best summarizes the call.
    summary_arg: str | None = None

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _obj(required: list[str], **props) -> dict:
    return {"type": "object", "properties": props, "required": required}


STR = {"type": "string"}
INT = {"type": "integer"}


@dataclass
class Toolset:
    tools: dict[str, Tool] = field(default_factory=dict)

    def add(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self.tools.values()]

    def __bool__(self) -> bool:
        return bool(self.tools)


def web_tools(settings: Settings, client: httpx.AsyncClient) -> list[Tool]:
    async def search(args: dict) -> str:
        return await web.search_with_excerpts(str(args["query"]), settings, client)

    async def fetch(args: dict) -> str:
        return await web.fetch(str(args["url"]), settings, client)

    return [
        Tool(
            "web_search",
            "Search the web. Returns titles, URLs, snippets, and excerpts of the top pages. "
            "Use fetch_url to read a page in full.",
            _obj(["query"], query=STR | {"description": "Search query"}),
            search,
            summary_arg="query",
        ),
        Tool(
            "fetch_url",
            "Fetch a web page (or PDF) and return its main text as markdown.",
            _obj(["url"], url=STR | {"description": "Absolute http(s) URL"}),
            fetch,
            summary_arg="url",
        ),
    ]


def workspace_tools(ws: Workspace, command_timeout: float) -> list[Tool]:
    async def run_command(args: dict) -> str:
        timeout = min(float(args.get("timeout") or 120), command_timeout)
        return await ws.run(str(args["command"]), timeout)

    async def wrap(fn, *a):
        return fn(*a)

    return [
        Tool(
            "list_files",
            "List files and folders (skips .git, node_modules, venvs).",
            _obj(
                [],
                path=STR | {"description": "Folder, relative to the workspace (default '.')"},
                depth=INT | {"description": "How many levels deep (default 2)"},
            ),
            lambda a: wrap(ws.list_files, a.get("path") or ".", int(a.get("depth") or 2)),
            summary_arg="path",
        ),
        Tool(
            "read_file",
            "Read a text file with line numbers. Use offset/limit for long files.",
            _obj(
                ["path"],
                path=STR,
                offset=INT | {"description": "First line (1-based)"},
                limit=INT | {"description": "Number of lines (default 400)"},
            ),
            lambda a: wrap(
                ws.read_file, a["path"], int(a.get("offset") or 1), int(a.get("limit") or 400)
            ),
            summary_arg="path",
        ),
        Tool(
            "search_files",
            "Search file contents with a regular expression (ripgrep).",
            _obj(
                ["pattern"],
                pattern=STR,
                path=STR | {"description": "Folder to search (default '.')"},
                glob=STR | {"description": "Optional file glob, e.g. '*.py'"},
            ),
            lambda a: ws.search(a["pattern"], a.get("path") or ".", a.get("glob")),
            summary_arg="pattern",
        ),
        Tool(
            "write_file",
            "Create or overwrite a file with the given content.",
            _obj(["path", "content"], path=STR, content=STR),
            lambda a: wrap(ws.write_file, a["path"], a["content"]),
            mutating=True,
            summary_arg="path",
        ),
        Tool(
            "edit_file",
            "Replace one exact occurrence of old_text with new_text in a file. Read the file "
            "first and copy old_text exactly, including indentation.",
            _obj(["path", "old_text", "new_text"], path=STR, old_text=STR, new_text=STR),
            lambda a: wrap(ws.edit_file, a["path"], a["old_text"], a["new_text"]),
            mutating=True,
            summary_arg="path",
        ),
        Tool(
            "run_command",
            "Run a bash command in the workspace (tests, builds, git, package managers). "
            "Non-interactive; output is truncated.",
            _obj(["command"], command=STR, timeout=INT | {"description": "Seconds (default 120)"}),
            run_command,
            mutating=True,
            summary_arg="command",
        ),
    ]


__all__ = ["Tool", "ToolError", "Toolset", "Workspace", "web_tools", "workspace_tools"]
