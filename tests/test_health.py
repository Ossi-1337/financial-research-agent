from __future__ import annotations

from financial_research_agent.health import build_health_report
from financial_research_agent.settings import Settings


def test_health_report_has_no_external_requirements() -> None:
    settings = Settings.from_env({})

    report = build_health_report(settings)

    assert report["status"] == "ok"
    assert report["provider"]["llm_provider"] == "offline-test"
    assert report["notes"][0] == "Foundation health check only."
    assert "No real LLM calls" in report["notes"][1]
