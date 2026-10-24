"""Rules: standing instructions the user gives Amy, like Cursor's rules.

- **Scopes**: ``global`` (everywhere), ``folder:<id>`` (chats in one sidebar folder) and
  ``workspace:<name>`` (agent tasks in one workspace). A chat gets the global rules plus those
  of its folder and workspace; a folder or workspace can turn individual global rules off.
- **Applies**: ``always``, ``chat`` (plain chats only) or ``agent`` (workspace tasks only).
- **Project files**: in agent mode, rule files already in the workspace (AGENTS.md, CLAUDE.md,
  .cursorrules, .cursor/rules/, .amy/rules/) count as workspace rules too.

``ask_models`` lets Amy follow rules such as "have three models deliberate on every task": it
puts one question to several installed models at once and returns what each said.
"""

import asyncio
import logging
import re
from pathlib import Path

import httpx

from app import routing
from app.subagents import Tracker
from app.tools import Tool

log = logging.getLogger(__name__)

APPLIES = ("always", "chat", "agent")
SCOPE_ORDER = {"global": 0, "folder": 1, "workspace": 2}
PROJECT_FILES = ("AGENTS.md", "CLAUDE.md", ".cursorrules")
PROJECT_DIRS = (".cursor/rules", ".amy/rules")
MAX_PROJECT_CHARS = 6000  # per file: enough for real rule files, not a whole README

RULES_HEADER = """Rules the user set. Follow them in this chat; when two conflict, the more \
specific one (workspace, then folder, then everywhere) wins:"""


def folder_scope(folder_id: str) -> str:
    return f"folder:{folder_id}"


def workspace_scope(name: str) -> str:
    return f"workspace:{name}"


def effective(db, folder_id: str | None, workspace: str | None, mode: str) -> list[dict]:
    """The rules for one chat, widest scope first, each with the scope it came from."""
    narrow = []
    if folder_id:
        narrow.append(folder_scope(folder_id))
    if workspace:
        narrow.append(workspace_scope(workspace))
    excluded = db.rule_exclusions(narrow)
    out = []
    for rule in db.list_rules(["global", *narrow]):
        if not rule["enabled"] or rule["applies"] not in ("always", mode):
            continue
        if rule["scope"] == "global" and rule["id"] in excluded:
            continue
        out.append(rule)
    return sorted(out, key=lambda r: SCOPE_ORDER[r["scope"].split(":", 1)[0]])


def project_rules(root: Path) -> list[dict]:
    """Rule files a workspace already has (from Cursor, Claude Code or other agents)."""
    paths = [root / name for name in PROJECT_FILES]
    for d in PROJECT_DIRS:
        folder = root / d
        if folder.is_dir():
            paths += sorted(p for p in folder.iterdir() if p.suffix in (".md", ".mdc", ".txt"))
    out = []
    for p in paths:
        if not p.is_file():
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue
        if text:
            out.append({"name": p.relative_to(root).as_posix(), "text": text[:MAX_PROJECT_CHARS]})
    return out


def prompt(rules: list[dict], project: list[dict]) -> str | None:
    if not rules and not project:
        return None
    lines = [RULES_HEADER]
    for r in rules:
        lines.append(f"- {r['name']}: {r['text']}")
    for f in project:
        lines.append(f"- From the workspace's {f['name']}:\n{f['text']}")
    return "\n".join(lines)


# ---------------------------------------------------------------- ask_models

PANEL = re.compile(
    r"ask_models|other models?|several models|\b(two|three|four|five|\d) models|sub-?agents?|"
    r"deliberat|second opinions?|panel of",
    re.I,
)


def wants_panel(rules: list[dict], project: list[dict]) -> bool:
    """Whether a rule asks Amy to consult other models (so ask_models should be offered)."""
    return any(PANEL.search(r["text"]) for r in [*rules, *project])


ASK_SYSTEM = """You're one of several models asked the same question so the user gets more than \
one view. Answer it directly and on your own merits; say where you're unsure."""


def ask_models_tool(app, models: list[str], tracker: Tracker | None = None) -> Tool:
    async def run(args: dict) -> str:
        question = str(args.get("question") or "").strip()
        wanted = [m for m in (args.get("models") or []) if isinstance(m, str)][:5]
        if not question:
            return "Error: give a question."
        chosen, unknown = [], []
        for name in wanted:
            match = next((m for m in models if m == name or m.split("/", 1)[-1] == name), None)
            (chosen if match else unknown).append(match or name)
        if not chosen:
            return f"Error: none of those are installed. Installed chat models: {', '.join(models)}"

        async def one(model: str) -> str:
            resolved = app.state.providers.resolve(model)
            if resolved is None:
                return f"## {model}\n(not available)"
            provider, upstream = resolved
            payload = {
                "model": upstream,
                "messages": [
                    {"role": "system", "content": ASK_SYSTEM},
                    {"role": "user", "content": question},
                ],
            }

            async def ask() -> str:
                return (await provider.quick_reply(app.state.http, payload, timeout=600)).strip()

            try:
                answer = await (tracker.run("ask", model, question, ask) if tracker else ask())
            except (httpx.HTTPError, ValueError, KeyError, IndexError) as exc:
                answer = f"(failed: {str(exc) or type(exc).__name__})"
            return f"## {model}\n{answer.strip()}"

        answers = await asyncio.gather(*(one(m) for m in chosen))
        note = f"\n\n(Not installed: {', '.join(unknown)})" if unknown else ""
        return "\n\n".join(answers) + note

    return Tool(
        name="ask_models",
        description=(
            "Ask other installed models the same question at once and get each one's answer, "
            "so they can deliberate. Use it only when a rule or the user asks for other "
            "models' views. Give the full question with any context they need: they don't see "
            f"this chat. Installed chat models: {', '.join(models)}."
        ),
        parameters={
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "The question, with context."},
                "models": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Which models to ask (up to 5), by name.",
                },
            },
            "required": ["question", "models"],
        },
        run=run,
        summary_arg="question",
    )


async def chat_model_ids(app) -> list[str]:
    return [m["id"] for m in await routing.chat_models(app)]
