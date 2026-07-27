from __future__ import annotations

import asyncio
import os

import pytest

from financial_research_agent.llm import (
    AnthropicProvider,
    ChatMessage,
    ChatRequest,
    GeminiProvider,
    LiteLLMGatewayProvider,
)
from financial_research_agent.settings import Settings


@pytest.mark.parametrize(
    ("provider_name", "flag", "provider_type"),
    (
        ("anthropic", "FRA_ANTHROPIC_SMOKE_TEST", AnthropicProvider),
        ("gemini", "FRA_GEMINI_SMOKE_TEST", GeminiProvider),
        ("litellm", "FRA_LITELLM_SMOKE_TEST", LiteLLMGatewayProvider),
    ),
)
def test_hosted_provider_live_smoke_is_explicitly_gated(
    provider_name: str,
    flag: str,
    provider_type,
) -> None:
    if os.environ.get(flag) != "1":
        pytest.skip(f"Set {flag}=1 to run the {provider_name} smoke test.")
    settings = Settings.from_env()
    if settings.provider.llm_provider != provider_name:
        pytest.skip(f"Set FRA_LLM_PROVIDER={provider_name} to run this smoke test.")

    async def scenario() -> str:
        provider = provider_type.from_settings(settings.provider)
        health = await provider.check_health()
        assert health.reachable is True, health.error
        assert health.authenticated is True, health.error
        response = await provider.chat(
            ChatRequest(
                messages=(ChatMessage(role="user", content="Reply with exactly: ok"),),
                max_output_tokens=16,
            )
        )
        return response.message.content

    assert asyncio.run(scenario()).strip()
