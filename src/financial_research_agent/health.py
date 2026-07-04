from __future__ import annotations

import asyncio
import platform
from typing import Any

from financial_research_agent import __version__
from financial_research_agent.llm.local_openai import OpenAICompatibleLocalProvider
from financial_research_agent.llm.openai import OpenAIProvider
from financial_research_agent.settings import Settings


def build_health_report(settings: Settings) -> dict[str, Any]:
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
        "paths": settings.local_paths.to_dict(),
        "environment": settings.environment,
        "notes": [
            "Foundation health check only.",
            (
                "LLM calls use offline-test by default; local-openai or openai only run "
                "when configured."
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
    return report
