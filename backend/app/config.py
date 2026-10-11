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

    # Override GPU memory detection (GB), e.g. when Ollama runs on a different machine
    # or the GPU isn't NVIDIA. Used for model recommendations only.
    gpu_vram_gb: float | None = None

    # PyTorch wheel index for the image engine on NVIDIA GPUs (cu128 supports Turing
    # through Blackwell and needs driver 570+). CPU-only machines use the CPU index.
    torch_index_url: str = "https://download.pytorch.org/whl/cu128"

    # Stop idle image/audio engine workers after this many minutes, freeing their memory.
    engine_idle_minutes: float = 10

    # Web access for tool-using chats. search_provider: auto | searxng | brave | duckduckgo.
    search_provider: str = "auto"
    searxng_url: str | None = None
    # Set in the full image, which runs SearXNG inside the container. A blank searxng_url
    # (e.g. an empty Unraid template field) then means "use the bundled one".
    bundled_searxng: bool = False
    brave_api_key: str | None = None
    # Allow fetch_url to reach private/LAN addresses (off: protects your network from
    # pages that try to steer the model into probing it).
    web_allow_private: bool = False

    # Hugging Face token for gated models. Normally set on the Settings page; this
    # environment variable overrides it (and makes it read-only in the UI).
    hf_token: str | None = None

    # Agent mode.
    agent_max_steps: int = 40
    agent_command_timeout: float = 600
    # GitHub token made available to agent git commands (clone/push private repos).
    github_token: str | None = None

    # Timeout for upstream model requests, in seconds. Generation can be slow on big models.
    upstream_timeout: float = 600.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
