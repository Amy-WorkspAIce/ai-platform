"""Ollama model management: list, pull (with progress and GPU warm-up), delete, unload."""

import asyncio
import hashlib
import json
import logging
import re
from pathlib import Path

import httpx

from app.jobs import JobRegistry, PullState

log = logging.getLogger(__name__)

HF = "https://huggingface.co"
GGUF_QUANT = re.compile(r"[-._]((?:I?Q\d[\w]*?)|F16|BF16|F32)(-\d+-of-\d+)?\.gguf$", re.I)

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

    # ---- gated GGUF repos: Amy downloads with the user's token, then hands Ollama the file ----

    def start_import(self, model: str, token: str, cache_dir: Path) -> PullState:
        """Install ``hf.co/<repo>:<quant>`` from a gated repo. Ollama can't send a Hugging Face
        token when it pulls, so Amy downloads the GGUF itself and creates the model from it."""

        async def run(state: PullState) -> None:
            await self._import_gguf(state, token, cache_dir)

        return self.jobs.start(model, run)

    async def _import_gguf(self, state: PullState, token: str, cache_dir: Path) -> None:
        repo, _, quant = state.model.removeprefix("hf.co/").partition(":")
        headers = {"Authorization": f"Bearer {token}"}
        state.status = "Checking the model on Hugging Face"
        info = await self.client.get(
            f"{HF}/api/models/{repo}", params={"blobs": "true"}, headers=headers, timeout=30
        )
        if info.status_code in (401, 403):
            raise RuntimeError(
                f"Hugging Face denied access to {repo}. Open {HF}/{repo}, accept its license "
                "while logged in, and check your token in Settings."
            )
        info.raise_for_status()
        files = [
            s
            for s in info.json().get("siblings", [])
            if (m := GGUF_QUANT.search(s["rfilename"]))
            and m.group(1).upper() == quant.upper()
            and "mmproj" not in s["rfilename"].lower()
        ]
        if not files:
            raise RuntimeError(f"No {quant} GGUF file in {repo}")
        if len(files) > 1:
            raise RuntimeError(
                f"{quant} is split into {len(files)} files, which Ollama can't import. "
                "Pick a smaller quantization."
            )
        name, size = files[0]["rfilename"], files[0].get("size") or 0
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / Path(name).name
        try:
            digest = await self._download(
                state, f"{HF}/{repo}/resolve/main/{name}", headers, path, size
            )
            state.status = "Handing the model to Ollama"
            await self._push_blob(path, digest)
            resp = await self.client.post(
                f"{self.server_url}/api/create",
                json={"model": state.model, "files": {path.name: digest}, "stream": False},
                timeout=httpx.Timeout(30, read=1800),
            )
            if resp.is_error:
                raise RuntimeError(f"Ollama couldn't create the model: {resp.text[:300]}")
        finally:
            path.unlink(missing_ok=True)
        state.downloaded = True
        await self._warm_up(state)

    async def _download(
        self, state: PullState, url: str, headers: dict, path: Path, size: int
    ) -> str:
        state.status = "Downloading model"
        state.total, state.completed = size, 0
        sha = hashlib.sha256()
        async with self.client.stream(
            "GET", url, headers=headers, follow_redirects=True, timeout=httpx.Timeout(30, read=300)
        ) as resp:
            if resp.status_code in (401, 403):
                raise RuntimeError(
                    "Hugging Face denied the download; accept the model's license first."
                )
            resp.raise_for_status()
            state.total = int(resp.headers.get("content-length") or size)
            with path.open("wb") as f:
                async for chunk in resp.aiter_bytes(1 << 20):
                    f.write(chunk)
                    sha.update(chunk)
                    state.completed += len(chunk)
        return f"sha256:{sha.hexdigest()}"

    async def _push_blob(self, path: Path, digest: str) -> None:
        exists = await self.client.head(f"{self.server_url}/api/blobs/{digest}", timeout=30)
        if exists.status_code == 200:
            return

        async def chunks():
            with path.open("rb") as f:
                while chunk := f.read(1 << 20):
                    yield chunk

        resp = await self.client.post(
            f"{self.server_url}/api/blobs/{digest}",
            content=chunks(),
            headers={"Content-Length": str(path.stat().st_size)},
            timeout=httpx.Timeout(30, write=1800, read=600),
        )
        if resp.is_error:
            raise RuntimeError(f"Ollama rejected the file: {resp.text[:300]}")

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
