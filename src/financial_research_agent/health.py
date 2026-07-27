from __future__ import annotations

import asyncio
import platform
from typing import Any

from financial_research_agent import __version__
from financial_research_agent.llm.anthropic import AnthropicProvider
from financial_research_agent.llm.gemini import GeminiProvider
from financial_research_agent.llm.litellm import LiteLLMGatewayProvider
from financial_research_agent.llm.local_openai import OpenAICompatibleLocalProvider
from financial_research_agent.llm.openai import OpenAIProvider
from financial_research_agent.llm.registry import create_default_provider_registry
from financial_research_agent.performance import (
    default_local_model_profiles,
    prompt_budgets_for_limits,
)
from financial_research_agent.retrieval import LocalVectorIndex
from financial_research_agent.settings import ProviderTask, Settings
from financial_research_agent.storage import LocalStorageManager


def build_health_report(settings: Settings) -> dict[str, Any]:
    storage = LocalStorageManager.from_settings(settings).inspect()
    provider_registry = create_default_provider_registry(settings.provider)
    embedding_selection = settings.provider.selection_for_task(ProviderTask.EMBEDDINGS)
    retrieval_index = LocalVectorIndex.from_settings(settings)
    report: dict[str, Any] = {
        "app": "financial-research-agent",
        "version": __version__,
        "status": "ok",
        "runtime": {
            "implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
        },
        "provider": settings.provider.to_dict(),
        "data_sources": settings.data_sources.to_dict(),
        "storage": {
            "provider": storage.provider,
            "app_home": str(storage.app_home),
            "dataset_count": len(storage.datasets),
            "existing_dataset_count": sum(1 for entry in storage.datasets if entry.exists),
            "migration_count": len(storage.migrations),
            "warnings": list(storage.warnings),
        },
        "retrieval": {
            "provider": settings.retrieval.provider,
            "top_k": settings.retrieval.top_k,
            "min_score": settings.retrieval.min_score,
            "index": retrieval_index.metadata().to_dict(),
            "embedding_provider": embedding_selection.provider,
            "embedding_model": embedding_selection.model,
            "embedding_provider_registered": provider_registry.has_embedding_provider(
                embedding_selection.provider
            ),
        },
        "interoperability": settings.interoperability.to_dict(),
        "a2a": settings.a2a.to_dict(),
        "background": settings.background.to_dict(),
        "performance": {
            **settings.performance.to_dict(),
            "prompt_budgets": {
                name: budget.to_dict()
                for name, budget in prompt_budgets_for_limits(
                    max_input_tokens=settings.performance.prompt_budget_input_tokens,
                    max_output_tokens=settings.performance.prompt_budget_output_tokens,
                ).items()
            },
            "local_model_profiles": [
                profile.to_dict() for profile in default_local_model_profiles()
            ],
        },
        "security": settings.security.to_dict(),
        "paths": settings.local_paths.to_dict(),
        "environment": settings.environment,
        "notes": [
            "Foundation health check only.",
            (
                "LLM calls use offline-test by default; network providers only run when "
                "explicitly configured."
            ),
            (
                "SEC company ticker lookup and Alpha Vantage daily market data ingestion "
                "are available; market data requires an explicit API key."
            ),
            (
                "SEC companyfacts financial statement ingestion is available for SEC filers "
                "through official EDGAR XBRL JSON APIs."
            ),
            (
                "SEC EDGAR filing ingestion is available for HTML/TXT primary documents "
                "with local raw and extracted-text storage."
            ),
            (
                "Local vector retrieval can index stored filing chunks when an embedding "
                "provider is explicitly configured."
            ),
        ],
    }
    if settings.provider.llm_provider == "local-openai":
        local_endpoint = asyncio.run(
            OpenAICompatibleLocalProvider.from_settings(settings.provider).check_health()
        )
        report["local_endpoint"] = local_endpoint.to_dict()
        if not local_endpoint.reachable:
            report["status"] = "degraded"
    elif settings.provider.llm_provider == "openai":
        online_provider = asyncio.run(
            OpenAIProvider.from_settings(settings.provider).check_health()
        )
        report["online_provider"] = online_provider.to_dict()
        if not online_provider.reachable or not online_provider.authenticated:
            report["status"] = "degraded"
    elif settings.provider.llm_provider in {"anthropic", "gemini", "litellm"}:
        provider = {
            "anthropic": AnthropicProvider.from_settings,
            "gemini": GeminiProvider.from_settings,
            "litellm": LiteLLMGatewayProvider.from_settings,
        }[settings.provider.llm_provider](settings.provider)
        provider_health = asyncio.run(provider.check_health())
        report["online_provider"] = provider_health.to_dict()
        if not provider_health.reachable or not provider_health.authenticated:
            report["status"] = "degraded"
    return report
