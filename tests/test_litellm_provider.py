from __future__ import annotations

import asyncio

import httpx

from financial_research_agent.llm import (
    ChatMessage,
    ChatRequest,
    LiteLLMGatewayProvider,
    ProviderCapability,
    create_default_provider_registry,
)
from financial_research_agent.settings import ProviderSettings


def test_litellm_reuses_openai_wire_without_requiring_local_gateway_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        assert "authorization" not in request.headers
        return httpx.Response(
            200,
            json={
                "model": "research-model",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "Ready"},
                    }
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1},
            },
        )

    async def scenario():
        client = httpx.AsyncClient(
            base_url="http://litellm.test/v1/",
            transport=httpx.MockTransport(handler),
        )
        try:
            provider = LiteLLMGatewayProvider(
                model="research-model",
                base_url="http://litellm.test/v1",
                client=client,
            )
            return await provider.chat(
                ChatRequest(messages=(ChatMessage(role="user", content="Hello"),))
            )
        finally:
            await client.aclose()

    response = asyncio.run(scenario())
    assert response.provider == "litellm"
    assert response.message.content == "Ready"


def test_litellm_health_uses_models_and_reports_gateway_limitations() -> None:
    async def scenario():
        client = httpx.AsyncClient(
            base_url="http://litellm.test/v1/",
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(
                    200,
                    json={"data": [{"id": "research-model"}]},
                )
            ),
        )
        try:
            provider = LiteLLMGatewayProvider(
                model="research-model",
                base_url="http://litellm.test/v1",
                client=client,
            )
            return await provider.check_health()
        finally:
            await client.aclose()

    health = asyncio.run(scenario())
    assert health.status == "ok"
    assert health.authenticated is True
    assert any("upstream provider" in item for item in health.limitations)


def test_default_registry_exposes_new_provider_capabilities() -> None:
    registry = create_default_provider_registry(ProviderSettings(llm_model="configured-model"))
    assert registry.has_chat_provider("anthropic")
    assert not registry.has_embedding_provider("anthropic")
    assert registry.has_chat_provider("gemini")
    assert registry.has_embedding_provider("gemini")
    assert registry.has_chat_provider("litellm")
    assert registry.has_embedding_provider("litellm")
    assert (
        ProviderCapability.EMBEDDINGS in registry.embedding_provider("gemini").metadata.capabilities
    )
