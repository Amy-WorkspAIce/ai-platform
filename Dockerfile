# syntax=docker/dockerfile:1
#
# Two image variants:
#   full (default)  - bundles Ollama + CUDA runtime. Plug-and-play on Unraid / any NVIDIA host.
#   slim            - platform only; point AMY_OLLAMA_URL / AMY_PROVIDERS at external engines.
#
#   docker build -t amy-ai-platform .               # full
#   docker build -t amy-ai-platform:slim --target slim .

ARG OLLAMA_VERSION=latest

# ---- frontend build ----
FROM node:22-alpine AS frontend
WORKDIR /src
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-fund --no-audit
COPY frontend/ ./
RUN npm run build

# ---- slim: platform only ----
FROM python:3.12-slim AS slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    AMY_DATA_DIR=/data AMY_STATIC_DIR=/app/static AMY_BUNDLED_OLLAMA=false
WORKDIR /app
COPY backend/ /app/backend/
RUN pip install --no-cache-dir /app/backend
COPY --from=frontend /src/dist /app/static
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
EXPOSE 8484
VOLUME /data
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8484/health')"
ENTRYPOINT ["/entrypoint.sh"]

# ---- full: platform + bundled Ollama (CUDA) ----
FROM ollama/ollama:${OLLAMA_VERSION} AS full
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    AMY_DATA_DIR=/data AMY_STATIC_DIR=/app/static AMY_BUNDLED_OLLAMA=true \
    OLLAMA_HOST=127.0.0.1:11434 OLLAMA_MODELS=/data/ollama/models \
    NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    CUDA_CACHE_PATH=/data/cache/cuda CUDA_CACHE_MAXSIZE=4294967296 \
    PATH=/opt/venv/bin:$PATH
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-venv \
 && rm -rf /var/lib/apt/lists/* \
 && python3 -m venv /opt/venv
WORKDIR /app
COPY backend/ /app/backend/
RUN pip install --no-cache-dir /app/backend
COPY --from=frontend /src/dist /app/static
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
EXPOSE 8484
VOLUME /data
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8484/health')"
ENTRYPOINT ["/entrypoint.sh"]
