"""Workspace tools for agent mode: files and shell commands inside one directory.

Every path is resolved inside the workspace root (symlinks included), so the agent can't
read or write outside it. Commands run with the workspace as the working directory, a
timeout, and an environment stripped of the platform's secrets.

This confines the agent's *tools*, not arbitrary programs it runs: a shell command can still
reach anything the container user can. That's why commands and file changes require
approval by default.
"""

import asyncio
import contextlib
import os
import re
import shutil
import signal
import sys
from pathlib import Path

MAX_READ_CHARS = 40_000
MAX_OUTPUT_CHARS = 16_000
SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", "dist", "build", ".next"}

# Environment variables never passed to agent commands.
SECRET_ENV = re.compile(r"^(AMY_|OLLAMA_|HF_TOKEN|.*(_KEY|_TOKEN|_SECRET|PASSWORD).*$)")


class ToolError(Exception):
    pass


class Workspace:
    def __init__(self, root: Path, github_token: str | None = None, home: Path | None = None):
        self.root = root.resolve()
        self.github_token = github_token
        self.home = home or self.root

    def path(self, rel: str) -> Path:
        p = (self.root / (rel or ".")).resolve()
        if p != self.root and not p.is_relative_to(self.root):
            raise ToolError(f"'{rel}' is outside the workspace")
        return p

    def rel(self, p: Path) -> str:
        return p.relative_to(self.root).as_posix() or "."

    # ---- files ----

    def list_files(self, path: str = ".", depth: int = 2) -> str:
        base = self.path(path)
        if not base.is_dir():
            raise ToolError(f"'{path}' is not a directory")
        lines: list[str] = []

        def walk(d: Path, level: int) -> None:
            try:
                entries = sorted(d.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
            except OSError:
                return
            for e in entries:
                if len(lines) >= 500:
                    return
                indent = "  " * level
                if e.is_dir():
                    lines.append(f"{indent}{e.name}/")
                    if e.name not in SKIP_DIRS and level + 1 < depth:
                        walk(e, level + 1)
                else:
                    lines.append(f"{indent}{e.name}")

        walk(base, 0)
        if len(lines) >= 500:
            lines.append("... (truncated)")
        return "\n".join(lines) or "(empty directory)"

    def read_file(self, path: str, offset: int = 1, limit: int = 400) -> str:
        p = self.path(path)
        if not p.is_file():
            raise ToolError(f"'{path}' is not a file")
        text = p.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        start = max(1, offset)
        chunk = lines[start - 1 : start - 1 + max(1, limit)]
        out = "\n".join(f"{i:>5}  {line}" for i, line in enumerate(chunk, start))
        if len(out) > MAX_READ_CHARS:
            out = out[:MAX_READ_CHARS] + "\n... (truncated)"
        remaining = len(lines) - (start - 1 + len(chunk))
        if remaining > 0:
            out += f"\n... ({remaining} more lines; use offset={start + len(chunk)})"
        return out or "(empty file)"

    def write_file(self, path: str, content: str) -> str:
        p = self.path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        existed = p.exists()
        p.write_text(content, encoding="utf-8")
        verb = "Updated" if existed else "Created"
        return f"{verb} {self.rel(p)} ({len(content.splitlines())} lines)"

    def edit_file(self, path: str, old: str, new: str) -> str:
        p = self.path(path)
        if not p.is_file():
            raise ToolError(f"'{path}' is not a file")
        text = p.read_text(encoding="utf-8")
        count = text.count(old)
        if count == 0:
            raise ToolError("old_text not found; read the file again and copy the text exactly")
        if count > 1:
            raise ToolError(f"old_text appears {count} times; include more surrounding lines")
        p.write_text(text.replace(old, new, 1), encoding="utf-8")
        return f"Edited {self.rel(p)}"

    async def search(self, pattern: str, path: str = ".", glob: str | None = None) -> str:
        base = self.path(path)
        rg = shutil.which("rg")
        if rg:
            cmd = [rg, "-n", "--no-heading", "--max-count", "50", "-e", pattern]
            if glob:
                cmd += ["-g", glob]
            proc = await asyncio.create_subprocess_exec(
                *cmd, str(base), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
            )
            out, _ = await proc.communicate()
            text = out.decode(errors="replace").replace(str(self.root) + os.sep, "")
            return clip(text) or "No matches."
        return await asyncio.to_thread(self._py_search, pattern, base, glob)

    def _py_search(self, pattern: str, base: Path, glob: str | None) -> str:
        try:
            rx = re.compile(pattern)
        except re.error as exc:
            raise ToolError(f"Invalid regex: {exc}") from exc
        hits: list[str] = []
        for f in base.rglob(glob or "*"):
            if not f.is_file() or any(part in SKIP_DIRS for part in f.parts):
                continue
            try:
                for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
                    if rx.search(line):
                        hits.append(f"{self.rel(f)}:{n}:{line[:300]}")
                        if len(hits) >= 200:
                            return "\n".join(hits)
            except (UnicodeDecodeError, OSError):
                continue
        return "\n".join(hits) or "No matches."

    # ---- commands ----

    def env(self) -> dict[str, str]:
        env = {k: v for k, v in os.environ.items() if not SECRET_ENV.match(k)}
        # Hide the platform's own virtualenv so `python`/`pip` in a workspace can't touch it.
        env.pop("VIRTUAL_ENV", None)
        if sys.prefix != sys.base_prefix:
            own_bin = str(Path(sys.prefix) / ("Scripts" if os.name == "nt" else "bin"))
            env["PATH"] = os.pathsep.join(
                p for p in env.get("PATH", "").split(os.pathsep) if p and p != own_bin
            )
        # A workspace virtualenv is always active, so plain `pip`/`python`/`pytest` use it.
        # (Each command is a fresh shell; models often expect `source activate` to persist.)
        for name in (".venv", "venv"):
            venv_bin = self.root / name / ("Scripts" if os.name == "nt" else "bin")
            if venv_bin.is_dir():
                env["VIRTUAL_ENV"] = str(venv_bin.parent)
                env["PATH"] = os.pathsep.join([str(venv_bin), env.get("PATH", "")])
                break
        env.update(HOME=str(self.home), CI="1", PAGER="cat", GIT_TERMINAL_PROMPT="0")
        self._ensure_gitconfig()
        if self.github_token:
            # Credentials for github.com without writing them to disk.
            env.update(
                GH_TOKEN=self.github_token,
                AMY_GIT_TOKEN=self.github_token,
                GIT_CONFIG_COUNT="1",
                GIT_CONFIG_KEY_0="credential.https://github.com.helper",
                GIT_CONFIG_VALUE_0=(
                    "!f() { echo username=x-access-token; echo password=$AMY_GIT_TOKEN; }; f"
                ),
            )
        return env

    def _ensure_gitconfig(self) -> None:
        """Default commit identity in the agent's *global* git config, so a repository's own
        `git config user.*` still takes precedence (environment variables would override it)."""
        gitconfig = self.home / ".gitconfig"
        if not gitconfig.exists():
            self.home.mkdir(parents=True, exist_ok=True)
            gitconfig.write_text(
                "[user]\n\tname = Amy Agent\n\temail = amy-agent@localhost\n"
                "[init]\n\tdefaultBranch = main\n"
            )

    async def run(self, command: str, timeout: float = 120) -> str:
        """Run a command; raises ToolError (with the output) if it fails or times out."""
        text, ok = await self.execute(command, timeout)
        if not ok:
            raise ToolError(text)
        return text

    async def execute(self, command: str, timeout: float = 120) -> tuple[str, bool]:
        proc = await asyncio.create_subprocess_exec(
            "bash",
            "-lc",
            command,
            cwd=self.root,
            env=self.env(),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,  # so the whole process group can be killed
        )
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            _kill(proc)
            out = await proc.stdout.read() if proc.stdout else b""
            return clip(out.decode(errors="replace")) + f"\n[Timed out after {timeout:.0f}s]", False
        except asyncio.CancelledError:
            _kill(proc)
            raise
        text = clip(out.decode(errors="replace"))
        return f"{text}\n[exit code {proc.returncode}]".lstrip(), proc.returncode == 0


def _kill(proc: asyncio.subprocess.Process) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(proc.pid, signal.SIGKILL)


def clip(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    """Keep the start and end of long output; the end usually has the error."""
    if len(text) <= limit:
        return text
    head, tail = text[: limit // 3], text[-(limit * 2 // 3) :]
    return f"{head}\n... ({len(text) - limit:,} characters omitted) ...\n{tail}"
