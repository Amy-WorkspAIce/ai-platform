# Architecture

## Role in the suite

Amy WorkspAIce is a set of independent apps, one repo and one container each.
The AI Platform is the **AI service** of the suite:

```
 other apps / SDKs ──(OpenAI API)──▶  AI Platform  ──▶ model engines
                                           │           (Ollama, llama.cpp, vLLM, hosted APIs)
                                           │
                                           └──(MCP tools)──▶ other apps
                                                (docs, sheets, tasks, ...)
```

- **App → AI:** apps call `/v1/chat/completions` for AI features. They can point
  at a self-hosted platform or the hosted one; the contract is the same.
- **AI → App:** apps expose MCP servers; the platform registers them as
  *connectors* and uses their tools in chats/agents (planned, see
  [connectors.md](connectors.md)).

The platform **does not implement inference**. It brokers requests to engines.

## Inside the container

```
┌──────────────────────── container (:8484) ─────────────────────────┐
│ entrypoint.sh  (PUID/PGID → drop root, supervise processes)        │
│                                                                    │
│  uvicorn ── FastAPI app ─┬─ /health          (public)              │
│                          ├─ /api/*           (management, key)     │
│                          ├─ /v1/*            (OpenAI-compatible)   │
│                          └─ /*               (React SPA, static)   │
│                                │                                   │
│                                ▼ httpx                             │
│  ollama serve (127.0.0.1:11434, only when AMY_BUNDLED_OLLAMA=true) │
│                                                                    │
│ /data  ── secrets/, amy.db, ollama/models, images/, cache/cuda,    │
│           engines/<engine>/venv, models/<engine>/<model>           │
└────────────────────────────────────────────────────────────────────┘
```

## Code layout

```
backend/            Python 3.11+ / FastAPI
  app/main.py       app factory, router wiring, SPA serving
  app/config.py     settings (AMY_* env vars)
  app/security.py   API key generation + bearer auth
  app/providers/    model providers (OpenAI-compatible, Ollama) + registry
  app/hardware.py   GPU/RAM/disk detection (nvidia-smi, /proc/meminfo)
  app/catalog/      curated model catalog (catalog.json), fit + recommendations
  app/ollama_admin.py  install (background pulls + GPU warm-up), list, delete
  app/jobs.py       background install jobs with pollable progress
  app/engines/      on-demand engines for images/audio: installer, worker supervisor,
                    and the worker scripts that run inside each engine's own venv
  app/db.py         SQLite: conversations, messages (incl. tool calls), image gallery
  app/agent.py      tool-calling loop, approvals, streaming events
  app/tools/        web (search providers, page fetch with LAN blocking), workspace
                    (confined file tools, shell commands with scrubbed env)
  app/api/          routers: system, models, conversations, agent, media, openai (/v1)
  tests/
frontend/           TypeScript / React / Vite (built into backend's static dir)
  src/views/        ChatView, ImagesView, AudioView, ModelsView
scripts/            maintenance (refresh_catalog_sizes.py)
docker/             entrypoint
deploy/unraid/      Community Applications template
deploy/docker/      compose files (CPU, GPU override, external Ollama)
```

## Engines (images, audio)

Ollama only runs language models. For other modalities the platform installs an
**engine** on demand: the first time a user installs, say, an image model, the
platform creates `/data/engines/image/venv`, pip-installs PyTorch + diffusers into it, then
downloads the model to `/data/models/image/<model>`. Engines run as **worker
subprocesses** (stdlib HTTP on 127.0.0.1) that start on the first request and stop
after `AMY_ENGINE_IDLE_MINUTES` idle, freeing GPU memory. Before generating an image,
the platform asks Ollama to unload its models so the image pipeline gets the VRAM.

| Engine | Packages | Models |
|---|---|---|
| `image` | torch, diffusers, transformers, accelerate | FLUX.2 klein, SDXL, Z-Image Turbo, SD 1.5 |
| `audio` | faster-whisper (+ cuBLAS/cuDNN on GPU), kokoro-onnx | Whisper, Kokoro |

Pipelines are placed on the GPU in the cheapest way that fits: fully on GPU, per-component
CPU offload, or sequential offload for small cards.

## Model IDs

Models are exposed as `<provider>/<upstream-name>`, for example `ollama/llama3.2` or
`vllm/meta-llama/Llama-3.1-8B-Instruct`. Only the first `/` separates the two parts.
Engine models use the engine as the prefix: `image/flux2-klein-4b`, `audio/kokoro-82m`.

## Design rules

- Single port, single container, state only in `/data`.
- Zero required config; secure defaults (auth on).
- Anything another app might need goes through a documented API. Nothing is coupled through internals.
