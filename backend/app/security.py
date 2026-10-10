"""API key authentication.

On first start, if no ``AMY_API_KEY`` is configured, a key is generated, stored in
``<data_dir>/secrets/api_key`` and printed once to the logs. Every request to ``/api``
or ``/v1`` must send it as ``Authorization: Bearer <key>``.
"""

import hmac
import logging
import os
import secrets
from pathlib import Path

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import Settings

log = logging.getLogger(__name__)

_bearer = HTTPBearer(auto_error=False)


def load_or_create_api_key(settings: Settings) -> str:
    if settings.api_key:
        return settings.api_key

    key_file = settings.data_dir / "secrets" / "api_key"
    if key_file.exists():
        return key_file.read_text(encoding="utf-8").strip()

    key = "amy-" + secrets.token_urlsafe(32)
    _write_private(key_file, key)
    log.warning(
        "\n\n  No AMY_API_KEY set. Generated one and saved it to %s\n"
        "  Your API key (shown once): %s\n",
        key_file,
        key,
    )
    return key


def _write_private(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(content)


def require_api_key(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> None:
    expected: str = request.app.state.api_key
    if credentials is None or not hmac.compare_digest(
        credentials.credentials.encode(), expected.encode()
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
