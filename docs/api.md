# API

Interactive docs (OpenAPI) are served at `/docs` on a running instance.

All endpoints except `/health` require `Authorization: Bearer <api key>`.

## Public

| Method | Path | Description |
|---|---|---|
| GET | `/health` | `{"status":"ok","version":"..."}`. For container health checks. |

## OpenAI-compatible (`/v1`)

| Method | Path | Description |
|---|---|---|
| GET | `/v1/models` | Models from every reachable provider, as `provider/model` IDs. |
| POST | `/v1/chat/completions` | Standard OpenAI chat request. Supports `stream: true` (SSE). The request is forwarded to the provider unchanged, except that `model` is rewritten to the upstream name. |

Planned: `/v1/embeddings`, `/v1/completions`, tool calling with connector tools.

## Management (`/api`)

| Method | Path | Description |
|---|---|---|
| GET | `/api/system` | Version and provider status. |
| GET | `/api/hardware` | Detected GPUs (VRAM), RAM, CPU count, free disk in `/data`. |
| GET | `/api/catalog` | Curated model catalog annotated with `fit`, `installed`, `needed_gb` per variant, plus `recommendations` per use case. |
| GET | `/api/models/installed` | Models installed in Ollama. |
| POST | `/api/models/pull` | `{"model": "qwen3:8b"}`, starts a background download (202). Accepts any Ollama tag or `hf.co/user/repo:quant`. |
| GET | `/api/models/pulls` | Progress of active and recent downloads (`status`, `completed`, `total`, `done`, `error`). |
| DELETE | `/api/models/{model}` | Deletes an installed model. |

After a download finishes, the platform loads the model once and generates a single
token. On some GPUs, the first run compiles CUDA kernels, which can take minutes. Doing
it during install means users don't wait on their first message. The status shows
"Optimizing for your GPU (first run only)" while this happens.

Planned: provider CRUD, API key management, connectors, conversations.
