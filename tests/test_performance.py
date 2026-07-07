from __future__ import annotations

import asyncio
from decimal import Decimal
from pathlib import Path

from financial_research_agent.llm import (
    ChatMessage,
    ChatRequest,
    EmbeddingRequest,
    EmbeddingResponse,
    MessageRole,
    ModelMetadata,
    ProviderCapability,
    TokenUsage,
)
from financial_research_agent.performance import (
    CachingEmbeddingProvider,
    LocalEmbeddingCache,
    ProviderCallKind,
    ProviderRate,
    check_chat_request_budget,
    default_local_model_profiles,
    estimate_chat_request_tokens,
    estimate_provider_cost_usd,
    measured_chat,
    prompt_budgets_for_limits,
)
from financial_research_agent.performance.budgeting import PromptBudget
from financial_research_agent.settings import Settings


def test_prompt_budget_estimates_chat_request_and_reports_over_budget() -> None:
    request = ChatRequest(
        messages=(ChatMessage(role=MessageRole.USER, content="x" * 41),),
        model="offline-test",
    )
    budget = PromptBudget(name="test", max_input_tokens=5, max_output_tokens=20)

    check = check_chat_request_budget(request, budget)

    assert estimate_chat_request_tokens(request) == 11
    assert check.over_budget is True
    assert check.recommended_max_output_tokens == 20
    assert "exceed budget" in check.warnings[0]


def test_provider_cost_estimation_uses_local_rate_card_and_reports_unknown_hosted_cost() -> None:
    local_cost, local_source, local_warnings = estimate_provider_cost_usd(
        provider="local-openai",
        model="local-model",
        input_tokens=1000,
        output_tokens=500,
    )
    hosted_cost, hosted_source, hosted_warnings = estimate_provider_cost_usd(
        provider="openai",
        model="future-model",
        input_tokens=1000,
        output_tokens=500,
    )
    custom_cost, custom_source, _ = estimate_provider_cost_usd(
        provider="custom",
        model="custom-model",
        input_tokens=1_000_000,
        output_tokens=500_000,
        rates=(
            ProviderRate(
                provider="custom",
                model_pattern="custom-*",
                input_cost_per_million_tokens_usd=Decimal("1.25"),
                output_cost_per_million_tokens_usd=Decimal("2.50"),
                source="test_rate_card",
            ),
        ),
    )

    assert local_cost == "0.000000"
    assert local_source == "local_runtime_excludes_electricity_and_hardware_cost"
    assert local_warnings == ()
    assert hosted_cost is None
    assert hosted_source == "not_estimated"
    assert "No local rate card" in hosted_warnings[0]
    assert custom_cost == "2.500000"
    assert custom_source == "test_rate_card"


def test_prompt_budget_limits_can_be_configured_consistently() -> None:
    budgets = prompt_budgets_for_limits(max_input_tokens=500, max_output_tokens=100)

    assert budgets["chat"].max_input_tokens == 500
    assert budgets["cited_answer"].max_input_tokens == 500
    assert budgets["cited_answer"].max_output_tokens == 100
    assert budgets["cited_answer"].recommended_max_output_tokens == 1


def test_measured_chat_returns_latency_usage_and_cost_metrics() -> None:
    provider = _PerfChatProvider()
    request = ChatRequest(messages=(ChatMessage(role="user", content="Hello"),))

    measured = asyncio.run(measured_chat(provider, request))

    assert measured.value.message.content == "ok"
    assert measured.metrics.call_kind == ProviderCallKind.CHAT
    assert measured.metrics.provider == "offline-test"
    assert measured.metrics.total_tokens == 7
    assert measured.metrics.latency_ms >= 0
    assert measured.metrics.estimated_cost_usd == "0.000000"


def test_local_model_profiles_cover_three_resource_levels() -> None:
    profiles = default_local_model_profiles()

    assert [profile.id for profile in profiles] == ["small", "medium", "strong"]
    assert profiles[0].suggested_context_tokens < profiles[-1].suggested_context_tokens
    assert all(profile.model_guidance for profile in profiles)


def test_embedding_cache_persists_hash_only_and_avoids_repeated_provider_calls(
    tmp_path: Path,
) -> None:
    settings = Settings.from_env({"FRA_HOME": str(tmp_path)})
    provider = _CountingEmbeddingProvider()
    cache = LocalEmbeddingCache.from_settings(settings)
    cached_provider = CachingEmbeddingProvider(provider, cache)
    request = EmbeddingRequest(input_texts=("same text", "same text"), model="fixture-model")

    first = asyncio.run(cached_provider.embed(request))
    second = asyncio.run(cached_provider.embed(request))
    payload = (tmp_path / "cache" / "embedding_cache.json").read_text(encoding="utf-8")

    assert first.metadata["cache_hits"] == "0"
    assert first.metadata["cache_misses"] == "1"
    assert first.metadata["request_duplicates"] == "1"
    assert second.metadata["cache_hits"] == "2"
    assert second.metadata["cache_misses"] == "0"
    assert len(provider.calls) == 1
    assert provider.calls[0].input_texts == ("same text",)
    assert "same text" not in payload
    assert cache.count() == 1


class _PerfChatProvider:
    @property
    def metadata(self) -> ModelMetadata:
        return ModelMetadata(provider="offline-test", model="offline-test")

    async def chat(self, _request: ChatRequest):
        from financial_research_agent.llm import ChatResponse

        return ChatResponse(
            message=ChatMessage(role="assistant", content="ok"),
            provider="offline-test",
            model="offline-test",
            usage=TokenUsage(input_tokens=5, output_tokens=2),
        )


class _CountingEmbeddingProvider:
    def __init__(self) -> None:
        self.calls: list[EmbeddingRequest] = []

    @property
    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            provider="fixture",
            model="fixture-model",
            capabilities=(ProviderCapability.EMBEDDINGS,),
        )

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        self.calls.append(request)
        return EmbeddingResponse(
            embeddings=tuple((float(len(text)), 1.0) for text in request.input_texts),
            provider="fixture",
            model=request.model or "fixture-model",
            usage=TokenUsage(input_tokens=sum(len(text.split()) for text in request.input_texts)),
        )
