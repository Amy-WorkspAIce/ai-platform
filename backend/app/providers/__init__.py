from app.config import Settings
from app.providers.base import OllamaProvider, Provider


class ProviderRegistry:
    def __init__(self, providers: list[Provider]):
        self._providers = {p.name: p for p in providers}

    @classmethod
    def from_settings(cls, settings: Settings) -> "ProviderRegistry":
        providers: list[Provider] = []
        if settings.ollama_url:
            providers.append(OllamaProvider(settings.ollama_url))
        providers += [
            Provider(name=p.name, base_url=p.base_url, api_key=p.api_key)
            for p in settings.providers
        ]
        return cls(providers)

    def all(self) -> list[Provider]:
        return list(self._providers.values())

    def resolve(self, model_id: str) -> tuple[Provider, str] | None:
        """Split ``provider/model`` into the provider and its upstream model name.

        Only the first ``/`` is a separator: upstream names may contain slashes
        (e.g. ``ollama/hf.co/org/model:Q4_K_M``).
        """
        name, sep, upstream = model_id.partition("/")
        provider = self._providers.get(name)
        if not sep or provider is None or not upstream:
            return None
        return provider, upstream


__all__ = ["OllamaProvider", "Provider", "ProviderRegistry"]
