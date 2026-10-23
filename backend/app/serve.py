"""Run the platform: plain HTTP, plus HTTPS on a second port when a certificate is present.

Browsers only allow the microphone (dictation, recording a voice) on HTTPS pages, so the
container also serves HTTPS, with a self-signed certificate made on first start (see
``docker/entrypoint.sh``) or one you put in its place. Both ports serve the same app in one
process, so jobs and engine workers are shared.

    python -m app.serve        # AMY_PORT (8484), and AMY_HTTPS_PORT (8443) if AMY_TLS_DIR has
                               # cert.pem and key.pem
"""

import asyncio
import os
from pathlib import Path

import uvicorn

from app.main import create_app


def configs() -> list[uvicorn.Config]:
    app = create_app()
    common = {"host": "0.0.0.0", "proxy_headers": True, "forwarded_allow_ips": "*"}  # noqa: S104
    out = [uvicorn.Config(app, port=int(os.environ.get("AMY_PORT", "8484")), **common)]
    port = int(os.environ.get("AMY_HTTPS_PORT", "8443") or 0)
    tls = Path(os.environ.get("AMY_TLS_DIR", "/data/secrets/tls"))
    if port and (tls / "cert.pem").is_file() and (tls / "key.pem").is_file():
        out.append(
            uvicorn.Config(
                app,
                port=port,
                ssl_certfile=str(tls / "cert.pem"),
                ssl_keyfile=str(tls / "key.pem"),
                lifespan="off",  # the HTTP server runs the app's startup and shutdown
                **common,
            )
        )
    return out


async def main() -> None:
    servers = [uvicorn.Server(c) for c in configs()]
    if len(servers) > 1:
        print(f"[amy] HTTPS on :{servers[1].config.port}", flush=True)
    tasks = [asyncio.create_task(s.serve()) for s in servers]
    # When one stops (a signal, or a port already in use), stop the other too.
    _, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for s in servers:
        s.should_exit = True
    await asyncio.gather(*pending)


if __name__ == "__main__":
    asyncio.run(main())
