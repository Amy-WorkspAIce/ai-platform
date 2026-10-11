"""Shared helpers for engine workers.

Workers run inside an engine's own virtualenv (``/data/engines/<engine>/venv``), so this
module must only use the standard library. Engine libraries are imported lazily by the
worker scripts.
"""

import argparse
import json
import logging
import os
import shutil
import sys
import traceback
import urllib.request
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# (status, content_type, body)
Response = tuple[int, str, bytes]
Route = Callable[[dict], Response]

log = logging.getLogger("amy.worker")


def json_response(data: dict, status: int = 200) -> Response:
    return status, "application/json", json.dumps(data).encode()


def serve(port: int, routes: dict[str, Route]) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib API
            route = routes.get(f"GET {self.path}")
            if self.path == "/health":
                self._send(*json_response({"status": "ok"}))
            elif route is not None:
                self._send(*route({}))
            else:
                self._send(*json_response({"error": "not found"}, 404))

        def do_POST(self) -> None:  # noqa: N802 - stdlib API
            route = routes.get(self.path)
            if route is None:
                self._send(*json_response({"error": "not found"}, 404))
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                self._send(*route(body))
            except Exception as exc:  # report any engine failure to the platform
                traceback.print_exc()
                self._send(*json_response({"error": f"{type(exc).__name__}: {exc}"}, 500))

        def _send(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"[amy-worker] listening on 127.0.0.1:{port}", flush=True)
    server.serve_forever()


def download(spec: dict, dest: Path) -> None:
    """Download model files described by a catalog ``source`` spec into ``dest``."""
    dest.mkdir(parents=True, exist_ok=True)
    if spec["type"] == "hf":
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=spec["repo"],
            revision=spec.get("revision"),
            allow_patterns=spec.get("allow_patterns"),
            local_dir=dest,
        )
    elif spec["type"] == "urls":
        for name, url in spec["files"].items():
            target = dest / name
            if target.exists():
                continue
            partial = target.with_suffix(target.suffix + ".partial")
            with urllib.request.urlopen(url, timeout=60) as resp, partial.open("wb") as f:  # noqa: S310 - catalog URLs
                shutil.copyfileobj(resp, f, length=1024 * 1024)
            partial.rename(target)
    else:
        raise ValueError(f"Unknown source type {spec['type']!r}")


def main(
    routes: dict[str, Route], after_download: Callable[[Path, dict], None] | None = None
) -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, required=True)
    d = sub.add_parser("download")
    d.add_argument("--spec", required=True, help="JSON source spec")
    d.add_argument("--dest", required=True)
    d.add_argument("--config", default="{}", help="JSON engine config, for post-download steps")
    args = parser.parse_args()
    if args.cmd == "serve":
        serve(args.port, routes)
    else:
        download(json.loads(args.spec), Path(args.dest))
        if after_download:
            after_download(Path(args.dest), json.loads(args.config))
        sys.exit(0)
