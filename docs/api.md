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

Planned: provider CRUD, model pull/delete, API key management, connectors, conversations.
