"""Ollama model management: list, pull (with progress), delete.

Pulls run as background tasks owned by the server, so they survive the browser
navigating away; the UI polls ``PullManager.snapshot()`` for progress.
"""

import asyncio
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field

import httpx

log = logging.getLogger(__name__)

# Ollama names: [host/][namespace/]model[:tag], e.g. "qwen3:8b" or "hf.co/org/repo:Q4_K_M".
MODEL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-/:]{0,199}$")


def valid_model_name(name: str) -> bool:
    return bool(MODEL_NAME.match(name)) and ".." not in name


@dataclass
class PullState:
    model: str
    status: str = "queued"
    completed: int = 0
    total: int = 0
    done: bool = False
    downloaded: bool = False
    error: str | None = None
    started_at: float = field(default_factory=time.time)
    # Per-layer progress; Ollama reports each blob separately.
    _layers: dict[str, tuple[int, int]] = field(default_factory=dict, repr=False)

    def public(self) -> dict:
        d = asdict(self)
        d.pop("_layers")
        return d


class OllamaAdmin:
    def __init__(self, server_url: str, client: httpx.AsyncClient):
        self.server_url = server_url.rstrip("/")
        self.client = client
        self.pulls: dict[str, PullState] = {}
        self._tasks: set[asyncio.Task] = set()

    async def installed(self) -> list[dict]:
        resp = await self.client.get(f"{self.server_url}/api/tags", timeout=10)
        resp.raise_for_status()
        return resp.json().get("models") or []

    async def delete(self, model: str) -> None:
        resp = await self.client.request(
            "DELETE", f"{self.server_url}/api/delete", json={"model": model, "name": model}
        )
        resp.raise_for_status()

    def start_pull(self, model: str) -> PullState:
        existing = self.pulls.get(model)
        if existing and not existing.done:
            return existing
        state = PullState(model=model)
        self.pulls[model] = state
        task = asyncio.create_task(self._pull(state))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return state

    def snapshot(self) -> list[dict]:
        # Forget finished pulls after 10 minutes.
        cutoff = time.time() - 600
        self.pulls = {k: s for k, s in self.pulls.items() if not s.done or s.started_at > cutoff}
        return [s.public() for s in self.pulls.values()]

    async def _pull(self, state: PullState) -> None:
        try:
            async with self.client.stream(
                "POST",
                f"{self.server_url}/api/pull",
                json={"model": state.model, "name": state.model, "stream": True},
                timeout=httpx.Timeout(30, read=None),
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line.strip():
                        apply_progress(state, json.loads(line))
                    if state.done or state.downloaded:
                        break
            if state.downloaded:
                await self._warm_up(state)
            elif not state.done:
                state.error = state.error or "Pull ended unexpectedly"
            state.done = True
        except (httpx.HTTPError, ValueError) as exc:
            log.warning("Pull of %s failed: %s", state.model, exc)
            state.done, state.error = True, str(exc) or type(exc).__name__

    async def _warm_up(self, state: PullState) -> None:
        """Load the model and generate one token.

        The first run of a model on a GPU can trigger a one-time CUDA kernel compile that
        takes minutes on some architectures. Doing it here means it happens during install,
        not on the user's first message. Failures are ignored (e.g. embedding-only models).
        """
        state.status = "Optimizing for your GPU (first run only)"
        try:
            resp = await self.client.post(
                f"{self.server_url}/api/generate",
                json={"model": state.model, "prompt": "Hi", "options": {"num_predict": 1}},
                timeout=httpx.Timeout(30, read=900),
            )
            if resp.is_error:
                log.info("Warm-up of %s skipped: %s", state.model, resp.text[:200])
        except httpx.HTTPError as exc:
            log.info("Warm-up of %s failed: %s", state.model, exc)
        state.status = "success"


def apply_progress(state: PullState, event: dict) -> None:
    if "error" in event:
        state.done, state.error, state.status = True, event["error"], "error"
        return
    state.status = event.get("status", state.status)
    digest = event.get("digest")
    if digest and event.get("total"):
        state._layers[digest] = (event.get("completed", 0), event["total"])
        state.completed = sum(c for c, _ in state._layers.values())
        state.total = sum(t for _, t in state._layers.values())
    if state.status == "success":
        state.downloaded = True
        state.completed = state.total
