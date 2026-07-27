from __future__ import annotations

from dataclasses import dataclass, field
from typing import Self

from financial_research_agent.llm.openai import OpenAIProvider
from financial_research_agent.settings import DEFAULT_LITELLM_BASE_URL, ProviderSettings


def _litellm_limitations() -> tuple[str, ...]:
    return (
        "Capabilities depend on the model and upstream provider configured in LiteLLM.",
        "Gateway routing, retries, fallback, budgets, and telemetry are not managed by this app.",
        "A remote gateway may incur provider cost and apply separate retention policies.",
    )


@dataclass(frozen=True, slots=True)
class LiteLLMGatewayProvider(OpenAIProvider):
    base_url: str = DEFAULT_LITELLM_BASE_URL
    provider: str = "litellm"
    service_label: str = "LiteLLM gateway"
    user_agent: str = "financial-research-agent/litellm"
    require_api_key: bool = False
    limitations: tuple[str, ...] = field(default_factory=_litellm_limitations)

    @classmethod
    def from_settings(cls, settings: ProviderSettings) -> Self:
        return cls(
            model=settings.llm_model,
            api_key=settings.litellm_api_key,
            base_url=settings.litellm_base_url,
            embedding_model=settings.embedding_model or settings.llm_model,
            timeout_seconds=settings.llm_timeout_seconds,
        )
