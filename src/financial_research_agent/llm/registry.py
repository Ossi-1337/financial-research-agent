from __future__ import annotations

from typing import Self

from financial_research_agent.llm.contracts import (
    ChatProvider,
    EmbeddingProvider,
    ProviderError,
    ProviderErrorCode,
)
from financial_research_agent.llm.offline import OfflineTestProvider


class ProviderRegistry:
    def __init__(self) -> None:
        self._chat_providers: dict[str, ChatProvider] = {}
        self._embedding_providers: dict[str, EmbeddingProvider] = {}

    def register_chat_provider(self, name: str, provider: ChatProvider) -> Self:
        self._chat_providers[_normalize_name(name)] = provider
        return self

    def register_embedding_provider(self, name: str, provider: EmbeddingProvider) -> Self:
        self._embedding_providers[_normalize_name(name)] = provider
        return self

    def chat_provider(self, name: str) -> ChatProvider:
        provider_name = _normalize_name(name)
        try:
            return self._chat_providers[provider_name]
        except KeyError as exc:
            raise ProviderError(
                code=ProviderErrorCode.PROVIDER_UNAVAILABLE,
                message=f"Chat provider is not registered: {provider_name}",
                provider=provider_name,
            ) from exc

    def embedding_provider(self, name: str) -> EmbeddingProvider:
        provider_name = _normalize_name(name)
        try:
            return self._embedding_providers[provider_name]
        except KeyError as exc:
            raise ProviderError(
                code=ProviderErrorCode.PROVIDER_UNAVAILABLE,
                message=f"Embedding provider is not registered: {provider_name}",
                provider=provider_name,
            ) from exc

    def has_chat_provider(self, name: str) -> bool:
        return _normalize_name(name) in self._chat_providers

    def has_embedding_provider(self, name: str) -> bool:
        return _normalize_name(name) in self._embedding_providers


def create_offline_provider_registry() -> ProviderRegistry:
    provider = OfflineTestProvider()
    return (
        ProviderRegistry()
        .register_chat_provider(provider.provider, provider)
        .register_embedding_provider(provider.provider, provider)
    )


def _normalize_name(name: str) -> str:
    provider_name = name.strip()
    if provider_name == "":
        raise ValueError("provider name is required")
    return provider_name
