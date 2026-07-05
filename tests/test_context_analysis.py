from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from financial_research_agent.context_analysis import (
    ConfidenceLabel,
    ContextAnalysisResult,
    ContextAnalysisStatus,
    ContextFinding,
    ContextRecency,
    ContextScope,
    ContextSourceItem,
    ContextSourceType,
    NewsMacroSectorAgent,
    SourceReliability,
    create_default_context_source_strategy,
)
from financial_research_agent.llm.registry import create_offline_provider_registry
from financial_research_agent.settings import Settings
from financial_research_agent.web import ChatSessionStore, create_app

NOW = datetime(2026, 7, 5, 12, tzinfo=UTC)


def test_context_source_item_is_immutable_and_requires_source_metadata() -> None:
    item = _source_item(id="company-1", source_url="https://example.test/company")

    with pytest.raises(FrozenInstanceError):
        item.title = "changed"
    with pytest.raises(TypeError):
        item.metadata["key"] = "changed"
    with pytest.raises(ValueError, match="source_url must be an http"):
        _source_item(id="invalid-url", source_url="file:///tmp/source")
    with pytest.raises(ValueError, match="published_at must be timezone-aware"):
        _source_item(id="naive-date", published_at=datetime(2026, 7, 5))


def test_finding_requires_evidence_or_limitation() -> None:
    with pytest.raises(ValueError, match="source_item_ids or limitations"):
        ContextFinding(
            id="finding:empty",
            scope=ContextScope.COMPANY,
            title="Empty Finding",
            summary="No support.",
            confidence=ConfidenceLabel.UNKNOWN,
        )


def test_analysis_result_validates_source_references() -> None:
    source = _source_item(id="known-source")
    finding = ContextFinding(
        id="finding:company",
        scope=ContextScope.COMPANY,
        title="Company Finding",
        summary="Supported by source.",
        confidence=ConfidenceLabel.MEDIUM,
        source_item_ids=("known-source",),
    )

    result = ContextAnalysisResult(
        id="context-result",
        query="Context query",
        status=ContextAnalysisStatus.COMPLETE,
        created_at=NOW,
        source_items=(source,),
        findings=(finding,),
        source_strategy=create_default_context_source_strategy(),
    )

    assert result.findings[0].source_item_ids == ("known-source",)
    with pytest.raises(ValueError, match="unique ids"):
        ContextAnalysisResult(
            id="duplicate-result",
            query="Context query",
            status=ContextAnalysisStatus.COMPLETE,
            created_at=NOW,
            source_items=(source, source),
            findings=(finding,),
            source_strategy=create_default_context_source_strategy(),
        )
    with pytest.raises(ValueError, match="unknown source item"):
        ContextAnalysisResult(
            id="unknown-reference-result",
            query="Context query",
            status=ContextAnalysisStatus.COMPLETE,
            created_at=NOW,
            source_items=(source,),
            findings=(
                ContextFinding(
                    id="finding:unknown",
                    scope=ContextScope.COMPANY,
                    title="Unknown Finding",
                    summary="Unsupported reference.",
                    confidence=ConfidenceLabel.LOW,
                    source_item_ids=("missing-source",),
                ),
            ),
            source_strategy=create_default_context_source_strategy(),
        )


def test_source_strategy_covers_milestone_context_categories() -> None:
    categories = {item.category for item in create_default_context_source_strategy()}

    assert {
        ContextSourceType.COMPANY_NEWS,
        ContextSourceType.MACRO_INDICATOR,
        ContextSourceType.RATES,
        ContextSourceType.CURRENCY,
        ContextSourceType.COMMODITY,
        ContextSourceType.SECTOR_CONTEXT,
    }.issubset(categories)


def test_agent_separates_company_macro_and_sector_context() -> None:
    agent = NewsMacroSectorAgent(now=lambda: NOW)
    result = agent.analyze(
        query="Explain current Novo context.",
        company_symbols=("nvo",),
        sector="healthcare",
        region="eu",
        source_items=(
            _source_item(
                id="company-event",
                title="Company source update",
                source_name="Novo Investor Relations",
                source_type=ContextSourceType.COMPANY_EVENT,
                reliability=SourceReliability.COMPANY_SOURCE,
                scope=ContextScope.COMPANY,
                company_symbols=("NVO",),
                published_at=datetime(2026, 7, 4, tzinfo=UTC),
            ),
            _source_item(
                id="macro-rate",
                title="ECB policy rate release",
                source_name="ECB",
                source_type=ContextSourceType.RATES,
                reliability=SourceReliability.OFFICIAL,
                scope=ContextScope.MACRO,
                region="EU",
                published_at=datetime(2026, 7, 4, tzinfo=UTC),
            ),
            _source_item(
                id="sector-context",
                title="Healthcare sector policy update",
                source_name="Sector Data API",
                source_type=ContextSourceType.SECTOR_CONTEXT,
                reliability=SourceReliability.DOCUMENTED_API,
                scope=ContextScope.SECTOR,
                sector="Healthcare",
                published_at=datetime(2026, 7, 3, tzinfo=UTC),
            ),
        ),
    )

    findings_by_scope = {finding.scope: finding for finding in result.findings}

    assert result.status == ContextAnalysisStatus.COMPLETE
    assert findings_by_scope[ContextScope.COMPANY].source_item_ids == ("company-event",)
    assert findings_by_scope[ContextScope.MACRO].source_item_ids == ("macro-rate",)
    assert findings_by_scope[ContextScope.SECTOR].source_item_ids == ("sector-context",)
    assert "2026-07-04" in findings_by_scope[ContextScope.COMPANY].summary
    assert "Novo Investor Relations" in findings_by_scope[ContextScope.COMPANY].summary
    assert (
        "News items are contextual sources" not in findings_by_scope[ContextScope.COMPANY].warnings
    )


def test_agent_deduplicates_repeated_or_syndicated_sources() -> None:
    agent = NewsMacroSectorAgent(now=lambda: NOW)
    result = agent.analyze(
        query="Check repeated source handling.",
        source_items=(
            _source_item(
                id="low-reliability-copy",
                source_url="https://example.test/repeated",
                reliability=SourceReliability.SECONDARY,
            ),
            _source_item(
                id="official-copy",
                source_url="https://example.test/repeated",
                reliability=SourceReliability.OFFICIAL,
            ),
        ),
    )

    assert [item.id for item in result.source_items] == ["official-copy"]
    assert result.warnings == ("Deduplicated 1 repeated or syndicated context source item(s).",)


def test_agent_reports_no_reliable_recent_source_when_only_unknown_sources_exist() -> None:
    agent = NewsMacroSectorAgent(now=lambda: NOW)
    result = agent.analyze(
        query="Need reliable context.",
        source_items=(
            _source_item(
                id="unknown-blog",
                reliability=SourceReliability.UNKNOWN,
                published_at=datetime(2026, 7, 4, tzinfo=UTC),
            ),
        ),
    )

    assert result.status == ContextAnalysisStatus.NO_RELIABLE_SOURCES
    assert all(not finding.source_item_ids for finding in result.findings)
    assert all("No reliable recent" in finding.summary for finding in result.findings)


def test_agent_marks_stale_reliable_context_as_partial() -> None:
    agent = NewsMacroSectorAgent(now=lambda: NOW)
    result = agent.analyze(
        query="Need fresh context.",
        source_items=(
            _source_item(
                id="old-company-source",
                reliability=SourceReliability.COMPANY_SOURCE,
                published_at=datetime(2026, 5, 1, tzinfo=UTC),
            ),
        ),
    )
    company_finding = next(
        finding for finding in result.findings if finding.scope == ContextScope.COMPANY
    )

    assert result.status == ContextAnalysisStatus.PARTIAL
    assert company_finding.recency == ContextRecency.STALE
    assert company_finding.source_item_ids == ("old-company-source",)
    assert company_finding.limitations == (
        "Company-Specific Events uses non-recent source items; refresh before relying on it.",
    )


def test_context_analysis_api_returns_source_linked_output(tmp_path) -> None:
    client = TestClient(
        create_app(
            settings=Settings.from_env({"FRA_HOME": str(tmp_path)}),
            registry=create_offline_provider_registry(),
            session_store=ChatSessionStore(),
        )
    )

    response = client.post(
        "/api/context-analysis",
        json={
            "query": "What is the current macro context?",
            "region": "US",
            "source_items": [
                {
                    "id": "fed-rates",
                    "title": "Federal Reserve rate release",
                    "summary": "TEST TOOL OUTPUT official macro context.",
                    "source_url": "https://example.test/fed-rates",
                    "source_name": "Federal Reserve",
                    "source_type": "rates",
                    "reliability": "official",
                    "scope": "macro",
                    "retrieved_at": NOW.isoformat(),
                    "published_at": datetime(2026, 7, 4, tzinfo=UTC).isoformat(),
                    "region": "US",
                }
            ],
        },
    )
    analysis = response.json()["analysis"]

    assert response.status_code == 200
    assert analysis["status"] == "partial"
    assert analysis["source_items"][0]["source_url"] == "https://example.test/fed-rates"
    assert analysis["findings"][1]["source_item_ids"] == ["fed-rates"]
    assert analysis["source_strategy"][0]["category"] == "company_news"


def test_context_analysis_api_rejects_invalid_source_metadata(tmp_path) -> None:
    client = TestClient(
        create_app(
            settings=Settings.from_env({"FRA_HOME": str(tmp_path)}),
            registry=create_offline_provider_registry(),
            session_store=ChatSessionStore(),
        )
    )

    response = client.post(
        "/api/context-analysis",
        json={
            "query": "Bad source.",
            "source_items": [
                {
                    "id": "bad",
                    "title": "Bad source",
                    "summary": "TEST TOOL OUTPUT invalid URL.",
                    "source_url": "file:///tmp/source",
                    "source_name": "Bad Source",
                    "source_type": "company_news",
                    "reliability": "official",
                    "scope": "company",
                    "retrieved_at": NOW.isoformat(),
                }
            ],
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"]["error"] == "invalid_context_source"


def _source_item(
    *,
    id: str,
    title: str = "Context source title",
    summary: str = "TEST TOOL OUTPUT context source summary.",
    source_url: str | None = None,
    source_name: str = "Test Source",
    source_type: ContextSourceType = ContextSourceType.COMPANY_NEWS,
    reliability: SourceReliability = SourceReliability.REPUTABLE_NEWS,
    scope: ContextScope = ContextScope.COMPANY,
    retrieved_at: datetime = NOW,
    published_at: datetime | None = NOW,
    company_symbols: tuple[str, ...] = (),
    sector: str | None = None,
    region: str | None = None,
) -> ContextSourceItem:
    return ContextSourceItem(
        id=id,
        title=title,
        summary=summary,
        source_url=source_url or f"https://example.test/{id}",
        source_name=source_name,
        source_type=source_type,
        reliability=reliability,
        scope=scope,
        retrieved_at=retrieved_at,
        published_at=published_at,
        company_symbols=company_symbols,
        sector=sector,
        region=region,
        metadata={"fixture": "true"},
    )
