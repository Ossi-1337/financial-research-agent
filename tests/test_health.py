from __future__ import annotations

from financial_research_agent.health import build_health_report
from financial_research_agent.llm.local_openai import LocalEndpointHealth, LocalRuntime
from financial_research_agent.llm.openai import OnlineProviderHealth
from financial_research_agent.settings import Settings


def test_health_report_has_no_external_requirements() -> None:
    settings = Settings.from_env({})

    report = build_health_report(settings)

    assert report["status"] == "ok"
    assert report["provider"]["llm_provider"] == "offline-test"
    assert report["data_sources"]["market_data_provider"] == "alpha-vantage"
    assert report["data_sources"]["alpha_vantage_api_key_configured"] is False
    assert report["data_sources"]["financial_statement_provider"] == "sec-companyfacts"
    assert report["data_sources"]["filing_provider"] == "sec-edgar"
    assert report["storage"]["provider"] == "local-json"
    assert report["storage"]["dataset_count"] >= 1
    assert report["retrieval"]["provider"] == "local-vector"
    assert report["retrieval"]["embedding_provider"] == "disabled"
    assert report["notes"][0] == "Foundation health check only."
    assert "offline-test by default" in report["notes"][1]
    assert "companyfacts financial statement ingestion" in report["notes"][3]
    assert "SEC EDGAR filing ingestion" in report["notes"][4]


def test_health_report_includes_local_endpoint_when_configured(monkeypatch) -> None:
    class FakeProvider:
        @classmethod
        def from_settings(cls, _settings):
            return cls()

        async def check_health(self) -> LocalEndpointHealth:
            return LocalEndpointHealth(
                provider="local-openai",
                runtime=LocalRuntime.LLAMA_CPP,
                base_url="http://127.0.0.1:8080/v1/",
                model="local-model",
                reachable=True,
                status="ok",
                available_models=("local-model",),
            )

    monkeypatch.setattr(
        "financial_research_agent.health.OpenAICompatibleLocalProvider",
        FakeProvider,
    )
    settings = Settings.from_env(
        {
            "FRA_LLM_PROVIDER": "local-openai",
            "FRA_LLM_MODEL": "local-model",
        }
    )

    report = build_health_report(settings)

    assert report["status"] == "ok"
    assert report["local_endpoint"]["reachable"] is True
    assert report["local_endpoint"]["available_models"] == ["local-model"]


def test_health_report_includes_openai_provider_when_configured(monkeypatch) -> None:
    class FakeProvider:
        @classmethod
        def from_settings(cls, _settings):
            return cls()

        async def check_health(self) -> OnlineProviderHealth:
            return OnlineProviderHealth(
                provider="openai",
                base_url="https://api.openai.test/v1/",
                model="gpt-5.5",
                reachable=True,
                authenticated=True,
                status="ok",
                available_models=("gpt-5.5",),
            )

    monkeypatch.setattr("financial_research_agent.health.OpenAIProvider", FakeProvider)
    settings = Settings.from_env(
        {
            "FRA_LLM_PROVIDER": "openai",
            "FRA_LLM_MODEL": "gpt-5.5",
            "FRA_OPENAI_API_KEY": "test-key",
        }
    )

    report = build_health_report(settings)

    assert report["status"] == "ok"
    assert report["online_provider"]["authenticated"] is True
    assert report["online_provider"]["available_models"] == ["gpt-5.5"]
