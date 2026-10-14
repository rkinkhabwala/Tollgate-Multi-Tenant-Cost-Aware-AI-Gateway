"""Provider registry: catalog `provider` name → adapter instance.

Only providers with configuration are registered. In Phase 1 that's Ollama; OpenAI
and Anthropic are registered in Phase 3 when their API keys are set, so the
gateway runs with no hosted keys at all.
"""

from __future__ import annotations

from gateway.providers.base import ProviderAdapter


class ProviderRegistry:
    def __init__(self, adapters: dict[str, ProviderAdapter] | None = None) -> None:
        self._adapters: dict[str, ProviderAdapter] = dict(adapters or {})

    def register(self, provider: str, adapter: ProviderAdapter) -> None:
        self._adapters[provider] = adapter

    def get(self, provider: str) -> ProviderAdapter | None:
        return self._adapters.get(provider)

    def __contains__(self, provider: str) -> bool:
        return provider in self._adapters

    async def aclose(self) -> None:
        for adapter in self._adapters.values():
            close = getattr(adapter, "aclose", None)
            if close is not None:
                await close()
