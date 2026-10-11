#!/bin/bash
# Container entrypoint.
#  - Creates a runtime user from PUID/PGID (Unraid defaults: 99/100) and drops root.
#  - Optionally starts the bundled Ollama server and SearXNG (web search).
#  - Runs the platform. If either process exits, the container exits so Docker restarts it.
set -euo pipefail

PUID="${PUID:-1000}"
PGID="${PGID:-1000}"
DATA="${AMY_DATA_DIR:-/data}"
PORT="${AMY_PORT:-8484}"

mkdir -p "$DATA"

run_as() { "$@"; }
if [ "$(id -u)" = "0" ]; then
  getent group "$PGID" >/dev/null || groupadd -o -g "$PGID" amy
  getent passwd "$PUID" >/dev/null || useradd -o -u "$PUID" -g "$PGID" -d "$DATA" -s /usr/sbin/nologin amy
  chown "$PUID:$PGID" "$DATA"
  [ -d "$DATA/secrets" ] && chown -R "$PUID:$PGID" "$DATA/secrets"
  [ -d "$DATA/ollama" ] && chown "$PUID:$PGID" "$DATA/ollama"
  # Compiled GPU kernels; persisting them avoids a slow first run after every restart.
  if [ -n "${CUDA_CACHE_PATH:-}" ]; then
    mkdir -p "$CUDA_CACHE_PATH" && chown -R "$PUID:$PGID" "$CUDA_CACHE_PATH"
  fi
  export HOME="$DATA"
  run_as() { setpriv --reuid="$PUID" --regid="$PGID" --init-groups "$@"; }
fi

pids=()
if [ "${AMY_BUNDLED_OLLAMA:-false}" = "true" ] && command -v ollama >/dev/null; then
  mkdir -p "${OLLAMA_MODELS:-$DATA/ollama/models}"
  [ "$(id -u)" = "0" ] && chown -R "$PUID:$PGID" "$DATA/ollama"
  echo "[amy] starting bundled Ollama (models in ${OLLAMA_MODELS:-$DATA/ollama/models})"
  run_as ollama serve &
  pids+=($!)
fi

searxng_loop=""
if [ "${AMY_BUNDLED_SEARXNG:-false}" = "true" ] && [ -x /opt/searxng/bin/granian ]; then
  # Per-install secret (SearXNG refuses to start with its default key).
  secret_file="$DATA/secrets/searxng_secret"
  mkdir -p "$DATA/secrets"
  [ -s "$secret_file" ] || head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n' > "$secret_file"
  chmod 600 "$secret_file"
  [ "$(id -u)" = "0" ] && chown -R "$PUID:$PGID" "$DATA/secrets"
  export SEARXNG_SECRET; SEARXNG_SECRET="$(cat "$secret_file")"
  echo "[amy] starting bundled SearXNG (web search) on 127.0.0.1:8888"
  # Restart loop: web search being down must not take the whole container down.
  (
    while true; do
      run_as /opt/searxng/bin/granian --interface wsgi --host 127.0.0.1 --port 8888 \
        --workers 1 --log-level warning searx.webapp:app || true
      echo "[amy] SearXNG exited; restarting in 5s"
      sleep 5
    done
  ) &
  searxng_loop=$!
fi

echo "[amy] starting AI Platform on :$PORT"
run_as python -m uvicorn app.main:create_app --factory --host 0.0.0.0 --port "$PORT" \
  --proxy-headers --forwarded-allow-ips="*" &
pids+=($!)

stop_all() {
  kill "${pids[@]}" 2>/dev/null || true
  if [ -n "$searxng_loop" ]; then
    kill "$searxng_loop" 2>/dev/null || true
    pkill -f "searx.webapp:app" 2>/dev/null || true
  fi
}
trap stop_all TERM INT
wait -n "${pids[@]}"
status=$?
stop_all
exit "$status"
