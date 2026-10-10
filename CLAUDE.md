# Notes for AI coding assistants

This is the **AI Platform** of Amy WorkspAIce: a self-hosted AI service (OpenAI-compatible API
plus web UI) packaged as a single container, Unraid-first but portable.

- Read `docs/architecture.md` first.
- Backend: Python/FastAPI in `backend/app`. Frontend: React/Vite/TS in `frontend/src`.
- The platform **brokers** to model engines (Ollama, OpenAI-compatible); it never runs inference itself.
- Other suite apps are separate repos and integrate only via `/v1` (App → AI) and MCP connectors
  (AI → App, see `docs/connectors.md`). Don't embed docs/tasks/etc. features here.
- Every new setting: `AMY_*` in `app/config.py` + `docs/configuration.md` + the Unraid template if
  user-facing.
- Checks: `ruff check . && ruff format --check . && pytest` in `backend/`; `npm run build` in `frontend/`.
- Model catalog: `backend/app/catalog/catalog.json`. After adding models, run
  `python scripts/refresh_catalog_sizes.py --write` to verify tags and pull exact sizes from the
  Ollama registry. Image/audio families use engines marked `available: false` until those
  engines are implemented.
- Image/audio engines live in `backend/app/engines/`. Worker scripts (`*_worker.py`,
  `_worker_common.py`) run inside the engine's own venv: **stdlib only at import time**, never
  import from `app.*` there. Bump `EngineSpec.version` when changing an engine's packages.
- License is AGPL-3.0; only add dependencies with compatible licenses.
