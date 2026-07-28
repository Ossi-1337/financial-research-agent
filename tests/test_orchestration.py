from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime

import pytest

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
from financial_research_agent.orchestration import (
    AgentHandoff,
    DelegationRequest,
    DelegationResult,
    HandoffConfidence,
    OrchestratorExecutionPolicy,
    OrchestratorHandoffStatus,
    OrchestratorResearchInput,
    OrchestratorRunStatus,
    OrchestratorStepKind,
    ResearchOrchestrator,
    default_orchestrator_plan,
)
from financial_research_agent.orchestration.store import orchestrated_research_run_from_dict

NOW = datetime(2026, 7, 27, 12, tzinfo=UTC)


def test_orchestration_contracts_are_immutable_and_a2a_only() -> None:
    request = OrchestratorResearchInput(query="Research TEST TOOL OUTPUT company")

    assert OrchestratorExecutionPolicy.DISTRIBUTED_A2A.value == "distributed_a2a"
    assert OrchestratorExecutionPolicy.SEQUENTIAL_LOCAL_SAFE.value == "sequential_local_safe"
    assert [step.id for step in default_orchestrator_plan()] == [
        "resolve_company",
        "refresh_market_data",
        "refresh_financial_statements",
        "refresh_filings",
        "financial_report_analysis",
        "stock_price_analysis",
        "context_analysis",
        "synthesis",
    ]
    with pytest.raises(FrozenInstanceError):
        request.query = "changed"  # type: ignore[misc]


def test_orchestrator_requires_dispatcher() -> None:
    with pytest.raises(ValueError, match="step_dispatcher"):
        ResearchOrchestrator(
            company_search_provider=FixtureCompanySearchProvider(),
            step_dispatcher=None,  # type: ignore[arg-type]
        )


def test_research_input_requires_supported_roles_and_synthesis() -> None:
    with pytest.raises(ValueError, match="must include synthesis"):
        OrchestratorResearchInput(
            query="Research TEST TOOL OUTPUT company",
            specialist_roles=("stock",),
        )
    with pytest.raises(ValueError, match="unsupported specialist role"):
        OrchestratorResearchInput(
            query="Research TEST TOOL OUTPUT company",
            specialist_roles=("stock", "unknown", "synthesis"),
        )


@pytest.mark.parametrize(
    ("roles", "expected_steps"),
    [
        (
            ("stock", "synthesis"),
            ["refresh_market_data", "stock_price_analysis", "synthesis"],
        ),
        (
            ("financial-report", "synthesis"),
            [
                "refresh_financial_statements",
                "refresh_filings",
                "financial_report_analysis",
                "synthesis",
            ],
        ),
        (
            ("context", "synthesis"),
            ["context_analysis", "synthesis"],
        ),
    ],
)
def test_orchestrator_dispatches_only_selected_specialists_and_prerequisites(
    roles: tuple[str, ...],
    expected_steps: list[str],
) -> None:
    dispatcher = RecordingDispatcher()
    orchestrator = ResearchOrchestrator(
        company_search_provider=FixtureCompanySearchProvider(),
        step_dispatcher=dispatcher,
        now=lambda: NOW,
    )

    run = asyncio.run(
        orchestrator.run(
            OrchestratorResearchInput(
                query="Research TEST TOOL OUTPUT company",
                specialist_roles=roles,
            )
        )
    )

    assert run.status == OrchestratorRunStatus.COMPLETE
    assert run.specialist_roles == roles
    assert [request.step_id for request in dispatcher.requests] == expected_steps
    assert [handoff.step_id for handoff in run.handoffs[1:]] == expected_steps
    assert not any(handoff.status == OrchestratorHandoffStatus.SKIPPED for handoff in run.handoffs)


def test_selected_roles_and_agent_runtime_round_trip_with_legacy_defaults() -> None:
    dispatcher = RecordingDispatcher()
    orchestrator = ResearchOrchestrator(
        company_search_provider=FixtureCompanySearchProvider(),
        step_dispatcher=dispatcher,
        now=lambda: NOW,
    )
    run = asyncio.run(
        orchestrator.run(
            OrchestratorResearchInput(
                query="Research TEST TOOL OUTPUT company",
                refresh=False,
                specialist_roles=("stock", "synthesis"),
                agent_provider="scripted",
                agent_model="scripted-model",
            )
        )
    )

    restored = orchestrated_research_run_from_dict(run.to_dict())
    legacy_payload = run.to_dict()
    legacy_payload.pop("specialist_roles")
    legacy_payload.pop("agent_provider")
    legacy_payload.pop("agent_model")
    legacy = orchestrated_research_run_from_dict(legacy_payload)

    assert restored.specialist_roles == ("stock", "synthesis")
    assert restored.agent_provider == "scripted"
    assert restored.agent_model == "scripted-model"
    assert legacy.specialist_roles == ("financial-report", "stock", "context", "synthesis")
    assert legacy.agent_provider is None
    assert legacy.agent_model is None


def test_orchestrator_uses_only_dispatcher_and_stable_specialist_order() -> None:
    dispatcher = RecordingDispatcher()
    orchestrator = ResearchOrchestrator(
        company_search_provider=FixtureCompanySearchProvider(),
        step_dispatcher=dispatcher,
        now=lambda: NOW,
    )

    run = asyncio.run(
        orchestrator.run(
            OrchestratorResearchInput(
                query="Research TEST TOOL OUTPUT company",
                refresh=False,
            )
        )
    )

    assert run.status == OrchestratorRunStatus.COMPLETE
    assert run.execution_policy == OrchestratorExecutionPolicy.DISTRIBUTED_A2A
    assert [request.step_id for request in dispatcher.requests] == [
        "financial_report_analysis",
        "stock_price_analysis",
        "context_analysis",
        "synthesis",
    ]
    assert [handoff.kind for handoff in run.handoffs[-4:]] == [
        OrchestratorStepKind.FINANCIAL_REPORT_ANALYSIS,
        OrchestratorStepKind.STOCK_PRICE_ANALYSIS,
        OrchestratorStepKind.CONTEXT_ANALYSIS,
        OrchestratorStepKind.SYNTHESIS,
    ]


def test_specialist_failure_produces_partial_report_without_local_fallback() -> None:
    dispatcher = RecordingDispatcher(failed_step="stock_price_analysis")
    orchestrator = ResearchOrchestrator(
        company_search_provider=FixtureCompanySearchProvider(),
        step_dispatcher=dispatcher,
        now=lambda: NOW,
    )

    run = asyncio.run(
        orchestrator.run(OrchestratorResearchInput(query="Research company", refresh=False))
    )

    assert run.status == OrchestratorRunStatus.PARTIAL
    stock = next(
        handoff
        for handoff in run.handoffs
        if handoff.kind == OrchestratorStepKind.STOCK_PRICE_ANALYSIS
    )
    assert stock.error_code == "a2a_agent_unavailable"
    assert not any(
        handoff.execution and handoff.execution.mode.value == "local" for handoff in run.handoffs
    )


def test_missing_valid_synthesis_fails_run() -> None:
    dispatcher = RecordingDispatcher(failed_step="synthesis")
    orchestrator = ResearchOrchestrator(
        company_search_provider=FixtureCompanySearchProvider(),
        step_dispatcher=dispatcher,
        now=lambda: NOW,
    )

    run = asyncio.run(
        orchestrator.run(OrchestratorResearchInput(query="Research company", refresh=False))
    )

    assert run.status == OrchestratorRunStatus.FAILED
    assert run.synthesis_summary is None


class FixtureCompanySearchProvider:
    async def search(self, query: str, *, limit: int = 10) -> CompanySearchResult:
        del limit
        company = ResolvedCompany(
            id="fixture:company:test",
            legal_name="TEST TOOL OUTPUT COMPANY",
            identifiers=(
                EntityIdentifier(EntityIdentifierType.CIK, "0000000001", source="fixture"),
            ),
        )
        source = SourceMetadata(
            provider="fixture",
            provider_status="test fixture",
            source_url="https://example.invalid/company",
            retrieved_at=NOW,
            attribution="test fixture",
        )
        return CompanySearchResult(
            query=query,
            status=CompanySearchStatus.REVIEW_REQUIRED,
            candidates=(
                CompanySearchCandidate(
                    company=company,
                    securities=(
                        ResolvedSecurity(
                            id="fixture:security:test",
                            company_id=company.id,
                            ticker="TEST",
                            name=company.legal_name,
                        ),
                    ),
                    score=100,
                    match_reason="fixture exact match",
                    source=source,
                ),
            ),
            source=source,
        )


class RecordingDispatcher:
    def __init__(self, failed_step: str | None = None) -> None:
        self.failed_step = failed_step
        self.requests: list[DelegationRequest] = []

    async def dispatch(self, request: DelegationRequest, *, run=None) -> DelegationResult:
        del run
        self.requests.append(request)
        failed = request.step_id == self.failed_step
        output = (
            {}
            if failed
            else {
                "summary": "Source-backed TEST TOOL OUTPUT synthesis.",
                "analysis": {"fixture": "TEST TOOL OUTPUT"},
            }
        )
        return DelegationResult(
            handoff=AgentHandoff(
                id=f"handoff:{request.step_id}",
                step_id=request.step_id,
                kind=request.expected_kind,
                status=(
                    OrchestratorHandoffStatus.FAILED
                    if failed
                    else OrchestratorHandoffStatus.SUCCEEDED
                ),
                started_at=NOW,
                completed_at=NOW,
                output=output,
                evidence_ids=("evidence:test:1",) if not failed else (),
                limitations=("Specialist unavailable.",) if failed else (),
                confidence=HandoffConfidence.HIGH if not failed else HandoffConfidence.UNKNOWN,
                error_code="a2a_agent_unavailable" if failed else None,
                error_message="Specialist unavailable." if failed else None,
            )
        )
