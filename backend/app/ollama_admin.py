"""Ollama model management: list, pull (with progress and GPU warm-up), delete, unload."""

import asyncio
import json
import logging
import re

import httpx

from app.jobs import JobRegistry, PullState

log = logging.getLogger(__name__)

# Ollama names: [host/][namespace/]model[:tag], e.g. "qwen3:8b" or "hf.co/org/repo:Q4_K_M".
MODEL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-/:]{0,199}$")


def valid_model_name(name: str) -> bool:
    return bool(MODEL_NAME.match(name)) and ".." not in name


class OllamaAdmin:
    def __init__(self, server_url: str, client: httpx.AsyncClient, jobs: JobRegistry):
        self.server_url = server_url.rstrip("/")
        self.client = client
        self.jobs = jobs

    async def installed(self) -> list[dict]:
        resp = await self.client.get(f"{self.server_url}/api/tags", timeout=10)
        resp.raise_for_status()
        return resp.json().get("models") or []

    async def delete(self, model: str) -> None:
        resp = await self.client.request(
            "DELETE", f"{self.server_url}/api/delete", json={"model": model, "name": model}
        )
        resp.raise_for_status()

    async def unload_all(self) -> None:
        """Free GPU memory held by loaded models (e.g. before image generation)."""
        try:
            for m in await self._loaded():
                await self.client.post(
                    f"{self.server_url}/api/generate",
                    json={"model": m["name"], "keep_alive": 0},
                    timeout=30,
                )
            # Unloading is asynchronous; wait (briefly) until the memory is actually free.
            for _ in range(20):
                if not await self._loaded():
                    return
                await asyncio.sleep(0.5)
        except (httpx.HTTPError, ValueError) as exc:
            log.info("Could not unload Ollama models: %s", exc)

    async def _loaded(self) -> list[dict]:
        resp = await self.client.get(f"{self.server_url}/api/ps", timeout=5)
        return resp.json().get("models") or []

    def start_pull(self, model: str) -> PullState:
        return self.jobs.start(model, self._pull)

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
                    if state.error or state.downloaded:
                        break
        except (httpx.HTTPError, ValueError) as exc:
            log.warning("Pull of %s failed: %s", state.model, exc)
            state.error = str(exc) or type(exc).__name__
            return
        if state.downloaded:
            await self._warm_up(state)
        elif not state.error:
            state.error = "Pull ended unexpectedly"

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
        state.error, state.status = event["error"], "error"
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
