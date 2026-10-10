"""Model providers.

The platform does not run models itself. It brokers requests to engines that speak
the OpenAI API (Ollama, llama.cpp server, vLLM, LM Studio, hosted APIs...).
Model IDs exposed by this platform are namespaced: ``<provider>/<upstream model>``.
"""

import logging
from dataclasses import dataclass

import httpx

log = logging.getLogger(__name__)


@dataclass
class Provider:
    name: str
    base_url: str  # root of the OpenAI-compatible API, e.g. http://host:11434/v1
    api_key: str | None = None
    kind: str = "openai"

    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def url(self, path: str) -> str:
        return self.base_url.rstrip("/") + "/" + path.lstrip("/")

    async def list_models(self, client: httpx.AsyncClient) -> list[str]:
        """Return upstream model names. An unreachable provider yields no models."""
        try:
            resp = await client.get(self.url("models"), headers=self.headers(), timeout=10)
            resp.raise_for_status()
            return [m["id"] for m in resp.json().get("data", [])]
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            log.info("Provider %s unavailable: %s", self.name, exc)
            return []


class OllamaProvider(Provider):
    """Ollama, which serves an OpenAI-compatible API under /v1.

    Extra Ollama-native management (pull/delete models, VRAM info) goes here later.
    """

    def __init__(self, server_url: str, name: str = "ollama"):
        super().__init__(name=name, base_url=server_url.rstrip("/") + "/v1", kind="ollama")
