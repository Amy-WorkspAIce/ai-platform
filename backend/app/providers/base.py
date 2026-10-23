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

    async def quick_reply(self, client: httpx.AsyncClient, payload: dict, timeout=300) -> str:
        """A non-streaming completion for a background job (memory, summaries). Reasoning
        models would spend minutes, or the whole token budget, thinking about it, so ask them
        not to; a server that rejects the option gets the plain request."""
        url = self.url("chat/completions")
        body = {**payload, "stream": False, "reasoning_effort": "none"}
        resp = await client.post(url, json=body, headers=self.headers(), timeout=timeout)
        if resp.status_code in (400, 422):
            body.pop("reasoning_effort")
            resp = await client.post(url, json=body, headers=self.headers(), timeout=timeout)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"].get("content") or ""

    async def list_models(self, client: httpx.AsyncClient) -> list[str]:
        """Return upstream model names. An unreachable provider yields no models."""
        try:
            resp = await client.get(self.url("models"), headers=self.headers(), timeout=10)
            resp.raise_for_status()
            # Ollama returns "data": null when no models are pulled.
            return [m["id"] for m in resp.json().get("data") or []]
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            log.info("Provider %s unavailable: %s", self.name, exc)
            return []

    async def list_details(self, client: httpx.AsyncClient) -> list[dict]:
        """Chat models with what's known about them: [{name, size_gb?, tools?}]."""
        return [{"name": m} for m in await self.list_models(client)]


class OllamaProvider(Provider):
    """Ollama, which serves an OpenAI-compatible API under /v1.

    Extra Ollama-native management (pull/delete models, VRAM info) goes here later.
    """

    def __init__(self, server_url: str, name: str = "ollama"):
        super().__init__(name=name, base_url=server_url.rstrip("/") + "/v1", kind="ollama")
        self.server_url = server_url.rstrip("/")

    async def list_details(self, client: httpx.AsyncClient) -> list[dict]:
        try:
            resp = await client.get(f"{self.server_url}/api/tags", timeout=10)
            resp.raise_for_status()
            models = resp.json().get("models") or []
        except (httpx.HTTPError, ValueError) as exc:
            log.info("Provider %s unavailable: %s", self.name, exc)
            return []
        out = []
        for m in models:
            caps = m.get("capabilities")  # newer Ollama only
            if caps is not None and "completion" not in caps:
                continue  # embedding models can't chat
            out.append(
                {
                    "name": m["name"],
                    "size_gb": round(m["size"] / 1e9, 2) if m.get("size") else None,
                    "tools": ("tools" in caps) if caps is not None else None,
                }
            )
        return out
