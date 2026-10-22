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
        self._prefs_file = data_dir / "settings.json"
        self._env_hf_token = env_hf_token

    # ---- preferences (non-secret) ----

    def _prefs(self) -> dict:
        try:
            return json.loads(self._prefs_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _save_prefs(self, prefs: dict) -> None:
        self._prefs_file.parent.mkdir(parents=True, exist_ok=True)
        self._prefs_file.write_text(json.dumps(prefs, indent=2), encoding="utf-8")

    def model_defaults(self) -> dict[str, str]:
        """Pinned model per capability; missing means "best installed"."""
        return dict(self._prefs().get("models") or {})

    def set_model_default(self, capability: str, model: str | None) -> None:
        prefs = self._prefs()
        models = dict(prefs.get("models") or {})
        if model:
            models[capability] = model
        else:
            models.pop(capability, None)
        prefs["models"] = models
        self._save_prefs(prefs)

    @property
    def memory_enabled(self) -> bool:
        """Use long-term memory in chats (recall)."""
        return bool(self._prefs().get("memory_enabled", True))

    @property
    def memory_auto(self) -> bool:
        """Learn new facts automatically after each reply."""
        return bool(self._prefs().get("memory_auto", True))

    def set_memory(self, enabled: bool | None = None, auto: bool | None = None) -> None:
        prefs = self._prefs()
        if enabled is not None:
            prefs["memory_enabled"] = enabled
        if auto is not None:
            prefs["memory_auto"] = auto
        self._save_prefs(prefs)

    @property
    def voice(self) -> str | None:
        """The voice read-aloud and the speak tool use when none is asked for."""
        return self._prefs().get("voice") or None

    def set_voice(self, voice: str | None) -> None:
        prefs = self._prefs()
        prefs["voice"] = voice or None
        self._save_prefs(prefs)

    @property
    def web_access(self) -> bool:
        return bool(self._prefs().get("web_access", True))

    def set_web_access(self, enabled: bool) -> None:
        prefs = self._prefs()
        prefs["web_access"] = enabled
        self._save_prefs(prefs)

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
