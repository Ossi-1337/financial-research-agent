from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from financial_research_agent.entities import (
    CompanySearchCandidate,
    CompanySearchResult,
    CompanySearchStatus,
    EntityIdentifier,
    EntityIdentifierType,
    ResolvedCompany,
    ResolvedSecurity,
    SourceMetadata,
)
from financial_research_agent.llm import ToolCall
from financial_research_agent.tools import (
    ToolContext,
    ToolErrorCode,
    ToolPermission,
    ToolResultStatus,
    create_default_tool_registry,
)


def test_current_utc_datetime_tool_returns_iso_utc_timestamp() -> None:
    registry = create_default_tool_registry()
    result = asyncio.run(
        registry.execute(
            ToolCall(id="call_time", name="current_utc_datetime"),
            _tool_context("current_utc_datetime"),
        )
    )

    timestamp = result.data["utc_datetime"]

    assert result.status == ToolResultStatus.SUCCEEDED
    assert result.tool_call_id == "call_time"
    assert isinstance(timestamp, str)
    assert timestamp.endswith("Z")
    datetime.fromisoformat(timestamp.replace("Z", "+00:00"))


def test_calculate_ratio_tool_succeeds_and_handles_division_by_zero() -> None:
    registry = create_default_tool_registry()

    success = asyncio.run(
        registry.execute(
            ToolCall(
                id="call_ratio",
                name="calculate_ratio",
                arguments={"numerator": 5, "denominator": 2, "precision": 2},
            ),
            _tool_context("calculate_ratio"),
        )
    )
    failure = asyncio.run(
        registry.execute(
            ToolCall(
                id="call_zero",
                name="calculate_ratio",
                arguments={"numerator": 5, "denominator": 0},
            ),
            _tool_context("calculate_ratio"),
        )
    )

    assert success.status == ToolResultStatus.SUCCEEDED
    assert success.data["ratio"] == "2.50"
    assert failure.status == ToolResultStatus.FAILED
    assert failure.error_code == ToolErrorCode.DIVISION_BY_ZERO


def test_resolve_company_stub_returns_no_fake_company_data() -> None:
    registry = create_default_tool_registry()

    result = asyncio.run(
        registry.execute(
            ToolCall(
                id="call_company",
                name="resolve_company_stub",
                arguments={"query": "Novo Nordisk"},
            ),
            _tool_context("resolve_company_stub"),
        )
    )

    assert result.status == ToolResultStatus.SUCCEEDED
    assert result.data["query"] == "Novo Nordisk"
    assert result.data["matches"] == ()
    assert result.data["real_entity_resolution_available"] is False
    assert result.warnings == (
        "Real company entity resolution requires an injected company search provider.",
    )


def test_resolve_company_tool_returns_reviewable_candidates() -> None:
    registry = create_default_tool_registry(company_search_provider=FakeCompanySearchProvider())

    result = asyncio.run(
        registry.execute(
            ToolCall(
                id="call_company",
                name="resolve_company",
                arguments={"query": "Novo Nordisk", "limit": 3},
            ),
            _tool_context("resolve_company"),
        )
    )

    assert result.status == ToolResultStatus.SUCCEEDED
    assert result.source == "fake-company-search"
    assert result.data["status"] == "review_required"
    assert result.data["candidates"][0]["securities"][0]["ticker"] == "NVO"
    assert result.warnings == ("limit=3",)


def test_read_local_evidence_tool_reads_injected_mapping_and_reports_missing_items() -> None:
    registry = create_default_tool_registry()
    context = ToolContext(
        allowed_permissions=(ToolPermission.LOCAL_READ,),
        allowed_tools=("read_local_evidence",),
        local_evidence={
            "ev_1": {
                "title": "Source title",
                "excerpt": "Evidence excerpt.",
            }
        },
    )

    found = asyncio.run(
        registry.execute(
            ToolCall(
                id="call_found",
                name="read_local_evidence",
                arguments={"evidence_id": "ev_1"},
            ),
            context,
        )
    )
    missing = asyncio.run(
        registry.execute(
            ToolCall(
                id="call_missing",
                name="read_local_evidence",
                arguments={"evidence_id": "ev_missing"},
            ),
            context,
        )
    )

    assert found.status == ToolResultStatus.SUCCEEDED
    assert found.data["evidence"]["title"] == "Source title"
    assert missing.status == ToolResultStatus.FAILED
    assert missing.error_code == ToolErrorCode.NOT_FOUND


class FakeCompanySearchProvider:
    async def search(self, query: str, *, limit: int = 10) -> CompanySearchResult:
        source = SourceMetadata(
            provider="fake-company-search",
            provider_status="test fixture",
            source_url="https://example.invalid/company-search-fixture",
            retrieved_at=datetime(2026, 7, 4, tzinfo=UTC),
            attribution="test fixture",
        )
        company = ResolvedCompany(
            id="fixture:company:novo",
            legal_name="TEST TOOL OUTPUT NOVO NORDISK",
            identifiers=(EntityIdentifier(EntityIdentifierType.TICKER, "NVO", source="fixture"),),
        )
        security = ResolvedSecurity(
            id="fixture:security:nvo",
            company_id=company.id,
            ticker="NVO",
            name=company.legal_name,
        )
        return CompanySearchResult(
            query=query,
            status=CompanySearchStatus.REVIEW_REQUIRED,
            candidates=(
                CompanySearchCandidate(
                    company=company,
                    securities=(security,),
                    score=90,
                    match_reason="test_fixture",
                    source=source,
                ),
            ),
            source=source,
            warnings=(f"limit={limit}",),
        )


def _tool_context(*tool_names: str) -> ToolContext:
    return ToolContext(
        allowed_permissions=tuple(ToolPermission),
        allowed_tools=tool_names,
    )
