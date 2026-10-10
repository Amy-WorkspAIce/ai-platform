"""Runtime configuration, read from environment variables prefixed with ``AMY_``."""

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProviderConfig(BaseModel):
    """An OpenAI-compatible model provider (llama.cpp server, vLLM, LM Studio, hosted APIs...)."""

    name: str
    base_url: str
    api_key: str | None = None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AMY_", env_file=".env", extra="ignore")

    # Where all persistent state lives. Map this to a volume.
    data_dir: Path = Path("/data")

    # API key required for /api and /v1. Auto-generated into data_dir if unset.
    api_key: str | None = None

    # Ollama: bundled inside the container, or any reachable Ollama server.
    # Set to an empty string to disable the Ollama provider entirely.
    ollama_url: str = "http://127.0.0.1:11434"

    # Extra OpenAI-compatible providers, as JSON:
    #   AMY_PROVIDERS='[{"name":"openai","base_url":"https://api.openai.com/v1","api_key":"sk-..."}]'
    providers: list[ProviderConfig] = []

    # Directory holding the built frontend. Served at / when present.
    static_dir: Path = Path("/app/static")

    # Comma-separated origins allowed to call the API from a browser (for other apps' frontends).
    cors_origins: str = ""

    # Timeout for upstream model requests, in seconds. Generation can be slow on big models.
    upstream_timeout: float = 600.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
