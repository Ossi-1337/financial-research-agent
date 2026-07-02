from __future__ import annotations

import asyncio
import os

from financial_research_agent.llm import ChatMessage, ChatRequest, MessageRole, OfflineTestProvider
from financial_research_agent.llm.local_openai import OpenAICompatibleLocalProvider
from financial_research_agent.settings import Settings


def test_local_endpoint_smoke_or_offline_fallback() -> None:
    if os.environ.get("FRA_LOCAL_SMOKE_TEST") == "1":
        settings = Settings.from_env()
        provider = OpenAICompatibleLocalProvider.from_settings(settings.provider)
        health = asyncio.run(provider.check_health())
        assert health.reachable, health.error
        return

    provider = OfflineTestProvider()
    response = asyncio.run(
        provider.chat(
            ChatRequest(
                messages=[ChatMessage(role=MessageRole.USER, content="Local smoke fallback.")]
            )
        )
    )

    assert response.provider == "offline-test"
