"""Gated GGUF repos: Amy downloads with the user's token and creates the model in Ollama."""

import asyncio
import hashlib
import json

import httpx

from app.jobs import JobRegistry
from app.ollama_admin import OllamaAdmin

GGUF = b"GGUF" + b"\0" * 2000
DIGEST = "sha256:" + hashlib.sha256(GGUF).hexdigest()


def fake(calls: list, allowed: bool = True):
    def handle(req: httpx.Request) -> httpx.Response:
        calls.append((req.method, req.url.path))
        auth = req.headers.get("Authorization") == "Bearer hf_tok"
        if req.url.host == "huggingface.co":
            if not (auth and allowed):
                return httpx.Response(403)
            if req.url.path == "/api/models/org/Model-GGUF":
                return httpx.Response(
                    200,
                    json={
                        "gated": "auto",
                        "siblings": [
                            {"rfilename": "Model-Q4_K_M.gguf", "size": len(GGUF)},
                            {"rfilename": "Model-Q8_0-00001-of-00002.gguf", "size": 9},
                            {"rfilename": "Model-Q8_0-00002-of-00002.gguf", "size": 9},
                            {"rfilename": "mmproj-Model-Q4_K_M.gguf", "size": 9},
                        ],
                    },
                )
            if req.url.path == "/org/Model-GGUF/resolve/main/Model-Q4_K_M.gguf":
                return httpx.Response(200, content=GGUF)
        if req.url.path == f"/api/blobs/{DIGEST}":
            if req.method == "HEAD":
                return httpx.Response(404)
            assert req.read() == GGUF
            return httpx.Response(201)
        if req.url.path == "/api/create":
            body = json.loads(req.content)
            assert body["model"] == "hf.co/org/Model-GGUF:Q4_K_M"
            assert body["files"] == {"Model-Q4_K_M.gguf": DIGEST}
            return httpx.Response(200, json={"status": "success"})
        if req.url.path == "/api/generate":
            return httpx.Response(200, json={"response": "hi"})
        return httpx.Response(404)

    return handle


async def run(model: str, tmp_path, allowed: bool = True):
    calls: list = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(fake(calls, allowed))) as client:
        admin = OllamaAdmin("http://ollama:11434", client, JobRegistry())
        state = admin.start_import(model, "hf_tok", tmp_path / "cache")
        while not state.done:
            await asyncio.sleep(0.01)
    return state, calls


def test_gated_gguf_is_downloaded_and_created(tmp_path):
    state, calls = asyncio.run(run("hf.co/org/Model-GGUF:Q4_K_M", tmp_path))
    assert state.error is None and state.status == "success"
    assert state.completed == len(GGUF)
    assert ("POST", "/api/create") in calls and ("POST", "/api/generate") in calls
    assert list((tmp_path / "cache").iterdir()) == []  # the download is cleaned up


def test_split_files_and_denied_access_are_explained(tmp_path):
    state, _ = asyncio.run(run("hf.co/org/Model-GGUF:Q8_0", tmp_path))
    assert "split into 2 files" in state.error
    state, _ = asyncio.run(run("hf.co/org/Model-GGUF:Q4_K_M", tmp_path, allowed=False))
    assert "accept its license" in state.error
