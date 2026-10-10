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
│ /data  ── secrets/api_key, ollama/models, (future) app.db          │
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
  app/api/          routers: system (/health, /api), models (/api), openai (/v1)
  tests/
frontend/           TypeScript / React / Vite (built into backend's static dir)
  src/views/        ChatView (chat), ModelsView (discovery + install)
scripts/            maintenance (refresh_catalog_sizes.py)
docker/             entrypoint
deploy/unraid/      Community Applications template
deploy/docker/      compose files (CPU, GPU override, external Ollama)
```

## Model IDs

Models are exposed as `<provider>/<upstream-name>`, for example `ollama/llama3.2` or
`vllm/meta-llama/Llama-3.1-8B-Instruct`. Only the first `/` separates the two parts.

## Design rules

- Single port, single container, state only in `/data`.
- Zero required config; secure defaults (auth on).
- Anything another app might need goes through a documented API. Nothing is coupled through internals.
