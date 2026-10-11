"""Settings people change from the UI (the Settings page), as opposed to deployment config.

Secrets live in ``<data>/secrets`` with owner-only permissions; environment variables still
win (``AMY_HF_TOKEN``), so a deployment can pin a value and the UI shows it as read-only.
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import httpx

log = logging.getLogger(__name__)

HF_WHOAMI = "https://huggingface.co/api/whoami-v2"


@dataclass
class HuggingFaceAuth:
    token: str | None
    username: str | None
    source: str | None  # "env" | "ui" | None


class UserSettings:
    def __init__(self, data_dir: Path, env_hf_token: str | None = None):
        self._hf_file = data_dir / "secrets" / "huggingface.json"
        self._env_hf_token = env_hf_token

    # ---- Hugging Face ----

    def huggingface(self) -> HuggingFaceAuth:
        if self._env_hf_token:
            return HuggingFaceAuth(self._env_hf_token, None, "env")
        try:
            data = json.loads(self._hf_file.read_text(encoding="utf-8"))
            return HuggingFaceAuth(data.get("token"), data.get("username"), "ui")
        except (OSError, ValueError):
            return HuggingFaceAuth(None, None, None)

    @property
    def hf_token(self) -> str | None:
        return self.huggingface().token

    def save_hf_token(self, token: str, username: str | None) -> None:
        self._hf_file.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self._hf_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"token": token, "username": username}, f)

    def clear_hf_token(self) -> None:
        self._hf_file.unlink(missing_ok=True)


async def verify_hf_token(token: str, client: httpx.AsyncClient) -> str:
    """Return the Hugging Face username for a token, or raise ValueError."""
    try:
        resp = await client.get(HF_WHOAMI, headers={"Authorization": f"Bearer {token}"}, timeout=15)
    except httpx.HTTPError as exc:
        raise ValueError(f"Couldn't reach Hugging Face: {exc}") from exc
    if resp.status_code == 401:
        raise ValueError("Hugging Face rejected this token. Check that it's copied correctly.")
    if resp.is_error:
        raise ValueError(f"Hugging Face returned HTTP {resp.status_code}")
    return resp.json().get("name") or "unknown"
