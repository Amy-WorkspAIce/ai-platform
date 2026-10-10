#!/bin/bash
# Container entrypoint.
#  - Creates a runtime user from PUID/PGID (Unraid defaults: 99/100) and drops root.
#  - Optionally starts the bundled Ollama server.
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

echo "[amy] starting AI Platform on :$PORT"
run_as python -m uvicorn app.main:create_app --factory --host 0.0.0.0 --port "$PORT" \
  --proxy-headers --forwarded-allow-ips="*" &
pids+=($!)

trap 'kill "${pids[@]}" 2>/dev/null' TERM INT
wait -n "${pids[@]}"
status=$?
kill "${pids[@]}" 2>/dev/null || true
exit "$status"
