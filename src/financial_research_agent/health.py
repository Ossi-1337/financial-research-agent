from __future__ import annotations

import platform
from typing import Any

from financial_research_agent import __version__
from financial_research_agent.settings import Settings


def build_health_report(settings: Settings) -> dict[str, Any]:
    return {
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
            "Milestone 02 foundation only.",
            "No real LLM calls, financial data ingestion, database, or agents are configured yet.",
        ],
    }
