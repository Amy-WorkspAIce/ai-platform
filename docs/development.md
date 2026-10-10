# Development

## Prerequisites

- Python 3.11+
- Node 22+
- An Ollama server for real chats (optional; tests mock providers)

## Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

export AMY_DATA_DIR=./.data AMY_API_KEY=dev   # Windows: $env:AMY_DATA_DIR="./.data"; $env:AMY_API_KEY="dev"
uvicorn app.main:create_app --factory --reload --port 8484
```

Checks (CI runs the same):

```bash
ruff check . && ruff format --check . && pytest
```

## Frontend

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173, proxies /v1 /api /health to :8484
npm run build      # outputs dist/, which the Docker image copies to /app/static
```

## Container

```bash
docker build -t amy-ai-platform .
docker run --rm -p 8484:8484 -v ./data:/data --gpus all amy-ai-platform
```

## Conventions

- Settings are always `AMY_*` env vars, defined in `backend/app/config.py`, and
  documented in `docs/configuration.md`.
- New public endpoints go under `/api` (management) or `/v1` (OpenAI-compatible),
  and are documented in `docs/api.md`.
- Everything except `/health` and static files requires auth.
