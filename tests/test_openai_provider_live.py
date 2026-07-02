from __future__ import annotations

import asyncio
import os

import pytest

from financial_research_agent.llm import ChatMessage, ChatRequest, OpenAIProvider
from financial_research_agent.settings import Settings


def test_openai_live_smoke_is_explicitly_gated() -> None:
    if os.environ.get("FRA_OPENAI_SMOKE_TEST") != "1":
        pytest.skip("Set FRA_OPENAI_SMOKE_TEST=1 to run the hosted OpenAI smoke test.")

    settings = Settings.from_env()
    if settings.provider.openai_api_key is None:
        pytest.skip("Set FRA_OPENAI_API_KEY or OPENAI_API_KEY to run the hosted smoke test.")
    if settings.provider.llm_provider != "openai":
        pytest.skip("Set FRA_LLM_PROVIDER=openai to run the hosted smoke test.")

    async def scenario() -> str:
        provider = OpenAIProvider.from_settings(settings.provider)
        health = await provider.check_health()
        assert health.reachable is True
        assert health.authenticated is True
        response = await provider.chat(
            ChatRequest(messages=[ChatMessage(role="user", content="Reply with: ok")])
        )
        return response.message.content

    assert asyncio.run(scenario()).strip()
