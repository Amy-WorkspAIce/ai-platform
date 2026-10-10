# syntax=docker/dockerfile:1
#
# Two image variants:
#   full (default)  - bundles Ollama + CUDA runtime. Plug-and-play on Unraid / any NVIDIA host.
#   slim            - platform only; point AMY_OLLAMA_URL / AMY_PROVIDERS at external engines.
#
#   docker build -t amy-ai-platform .               # full
#   docker build -t amy-ai-platform:slim --target slim .
#
# Both include a small developer toolchain (git, ripgrep, curl, Node.js) for agent mode.
# The platform itself lives in /opt/venv, which agent commands don't see on their PATH,
# so `pip install` in a workspace can't break it.

ARG OLLAMA_VERSION=latest

# ---- frontend build ----
FROM node:22-alpine AS frontend
WORKDIR /src
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-fund --no-audit
COPY frontend/ ./
RUN npm run build

# ---- Node.js runtime for agent workspaces (glibc build, copied into the final images) ----
FROM node:22-bookworm-slim AS node

# ---- slim: platform only ----
FROM python:3.12-slim AS slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    AMY_DATA_DIR=/data AMY_STATIC_DIR=/app/static AMY_BUNDLED_OLLAMA=false \
    PATH=/opt/venv/bin:$PATH
RUN apt-get update \
 && apt-get install -y --no-install-recommends git ripgrep curl ca-certificates procps \
 && rm -rf /var/lib/apt/lists/* \
 && python -m venv /opt/venv
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s ../lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
 && ln -s ../lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx
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
# Ollama: 16k context for agent work; flash attention + q8_0 KV cache halve its memory cost.
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    AMY_DATA_DIR=/data AMY_STATIC_DIR=/app/static AMY_BUNDLED_OLLAMA=true \
    OLLAMA_HOST=127.0.0.1:11434 OLLAMA_MODELS=/data/ollama/models \
    OLLAMA_CONTEXT_LENGTH=16384 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 \
    NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    CUDA_CACHE_PATH=/data/cache/cuda CUDA_CACHE_MAXSIZE=4294967296 \
    PATH=/opt/venv/bin:$PATH
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-venv git ripgrep curl ca-certificates procps \
 && rm -rf /var/lib/apt/lists/* \
 && python3 -m venv /opt/venv
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s ../lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
 && ln -s ../lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx
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
