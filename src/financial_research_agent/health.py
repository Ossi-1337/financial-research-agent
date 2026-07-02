from __future__ import annotations

import asyncio
import platform
from typing import Any

from financial_research_agent import __version__
from financial_research_agent.llm.local_openai import OpenAICompatibleLocalProvider
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
        "paths": settings.local_paths.to_dict(),
        "environment": settings.environment,
        "notes": [
            "Foundation health check only.",
            "LLM calls use offline-test by default and local-openai only when configured.",
            "No financial data ingestion, database, or agents are configured yet.",
        ],
    }
    if settings.provider.llm_provider == "local-openai":
        local_endpoint = asyncio.run(
            OpenAICompatibleLocalProvider.from_settings(settings.provider).check_health()
        )
        report["local_endpoint"] = local_endpoint.to_dict()
        if not local_endpoint.reachable:
            report["status"] = "degraded"
    return report
