from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import UTC, datetime
from uuid import uuid4

from financial_research_agent.entities import (
    CompanySearchCandidate,
    CompanySearchError,
    CompanySearchProvider,
    CompanySearchResult,
    CompanySearchStatus,
    EntityIdentifierType,
    ResolvedSecurity,
)
from financial_research_agent.orchestration.contracts import (
    AgentHandoff,
    HandoffConfidence,
    OrchestratedResearchRun,
    OrchestratorExecutionPolicy,
    OrchestratorHandoffStatus,
    OrchestratorResearchInput,
    OrchestratorRunStatus,
    OrchestratorStepKind,
    default_orchestrator_plan,
)
from financial_research_agent.orchestration.dispatch import (
    AgentRole,
    DelegationRequest,
    ResearchStepDispatcher,
)
from financial_research_agent.orchestration.store import OrchestratorRunStore


class ResearchOrchestrator:
    """Owns research planning, A2A delegation, validation, and run persistence."""

    def __init__(
        self,
        *,
        company_search_provider: CompanySearchProvider,
        step_dispatcher: ResearchStepDispatcher,
        run_store: OrchestratorRunStore | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        if step_dispatcher is None:
            raise ValueError("step_dispatcher is required")
        self._company_search_provider = company_search_provider
        self._step_dispatcher = step_dispatcher
        self._run_store = run_store
        self._now = now or (lambda: datetime.now(UTC))

    async def run(
        self,
        request: OrchestratorResearchInput,
        *,
        progress_observer: Callable[[OrchestratedResearchRun], None] | None = None,
    ) -> OrchestratedResearchRun:
        created_at = _aware_now(self._now())
        run = OrchestratedResearchRun(
            id=request.run_id or f"orchestrator_run_{uuid4().hex}",
            query=request.query,
            status=OrchestratorRunStatus.RUNNING,
            created_at=created_at,
            updated_at=created_at,
            execution_policy=OrchestratorExecutionPolicy.DISTRIBUTED_A2A,
            plan=default_orchestrator_plan(),
            warnings=("Research uses the canonical A2A specialist topology.",),
            scenario_id=request.scenario_id,
            scenario_version=request.scenario_version,
        )
        run = self._save(run, progress_observer)

        resolution, _result, candidate = await self._resolve_company(request)
        run = self._append_handoff(run, resolution, progress_observer)
        if candidate is None:
            return self._save(
                replace(
                    run,
                    status=OrchestratorRunStatus.FAILED,
                    limitations=(
                        *run.limitations,
                        "Research stopped because no reviewable company was resolved.",
                    ),
                    updated_at=_aware_now(self._now()),
                ),
                progress_observer,
            )

        security = candidate.securities[0]
        cik = _candidate_identifier(candidate, EntityIdentifierType.CIK)
        run = self._save(
            replace(
                run,
                selected_company=candidate.company.to_dict(),
                selected_security=security.to_dict(),
                updated_at=_aware_now(self._now()),
            ),
            progress_observer,
        )

        for handoff in await self._refresh_data(request, run, candidate, security, cik):
            run = self._append_handoff(run, handoff, progress_observer)
        for handoff in await self._dispatch_specialists(
            request,
            run,
            candidate,
            security,
            cik,
        ):
            run = self._append_handoff(run, handoff, progress_observer)

        synthesis = await self._dispatch_synthesis(run)
        run = self._append_handoff(run, synthesis, progress_observer)
        final_status = _final_status(run.handoffs)
        return self._save(
            replace(
                run,
                status=final_status,
                synthesis_summary=_synthesis_summary(synthesis),
                limitations=tuple(
                    dict.fromkeys(
                        limitation for handoff in run.handoffs for limitation in handoff.limitations
                    )
                ),
                updated_at=_aware_now(self._now()),
            ),
            progress_observer,
        )

    async def _resolve_company(
        self,
        request: OrchestratorResearchInput,
    ) -> tuple[AgentHandoff, CompanySearchResult | None, CompanySearchCandidate | None]:
        started_at = _aware_now(self._now())
        company_query = request.company_query or request.query
        try:
            result = await self._company_search_provider.search(
                company_query,
                limit=request.company_search_limit,
            )
        except CompanySearchError as exc:
            return (
                _handoff(
                    kind=OrchestratorStepKind.COMPANY_RESOLUTION,
                    step_id="resolve_company",
                    status=OrchestratorHandoffStatus.FAILED,
                    started_at=started_at,
                    completed_at=_aware_now(self._now()),
                    input_summary={"query": company_query},
                    limitations=(exc.message,),
                    error_code=exc.code.value,
                    error_message=exc.message,
                ),
                None,
                None,
            )
        if result.status == CompanySearchStatus.NO_MATCHES or not result.candidates:
            limitation = "Company resolution returned no reviewable candidates."
            return (
                _handoff(
                    kind=OrchestratorStepKind.COMPANY_RESOLUTION,
                    step_id="resolve_company",
                    status=OrchestratorHandoffStatus.FAILED,
                    started_at=started_at,
                    completed_at=_aware_now(self._now()),
                    input_summary={"query": company_query},
                    output=result.to_dict(),
                    limitations=(limitation,),
                    error_code="no_company_match",
                    error_message=limitation,
                ),
                result,
                None,
            )
        candidate = result.candidates[0]
        return (
            _handoff(
                kind=OrchestratorStepKind.COMPANY_RESOLUTION,
                step_id="resolve_company",
                status=OrchestratorHandoffStatus.SUCCEEDED,
                started_at=started_at,
                completed_at=_aware_now(self._now()),
                input_summary={"query": company_query},
                output={"result": result.to_dict(), "selected_candidate": candidate.to_dict()},
                warnings=tuple(
                    dict.fromkeys(
                        (
                            *result.warnings,
                            *candidate.warnings,
                            "Top reviewable company candidate selected for bounded research.",
                        )
                    )
                ),
                confidence=HandoffConfidence.MEDIUM,
            ),
            result,
            candidate,
        )

    async def _refresh_data(
        self,
        request: OrchestratorResearchInput,
        run: OrchestratedResearchRun,
        candidate: CompanySearchCandidate,
        security: ResolvedSecurity,
        cik: str | None,
    ) -> tuple[AgentHandoff, ...]:
        if not request.refresh:
            now = _aware_now(self._now())
            return tuple(
                _skipped_handoff(kind, step_id, now, "Refresh disabled by request.")
                for kind, step_id in (
                    (OrchestratorStepKind.MARKET_DATA_REFRESH, "refresh_market_data"),
                    (
                        OrchestratorStepKind.FINANCIAL_STATEMENT_REFRESH,
                        "refresh_financial_statements",
                    ),
                    (OrchestratorStepKind.FILING_REFRESH, "refresh_filings"),
                )
            )
        delegations = (
            DelegationRequest(
                role=AgentRole.STOCK,
                run_id=run.id,
                step_id="refresh_market_data",
                correlation_id=run.id,
                expected_kind=OrchestratorStepKind.MARKET_DATA_REFRESH,
                payload={
                    "security_id": security.id,
                    "ticker": security.ticker,
                    "exchange_mic": security.exchange_mic,
                    "exchange_name": security.exchange_name,
                    "currency": security.currency,
                    "outputsize": request.market_outputsize,
                    "benchmark_symbol": request.benchmark_symbol,
                },
            ),
            DelegationRequest(
                role=AgentRole.FINANCIAL_REPORT,
                run_id=run.id,
                step_id="refresh_financial_statements",
                correlation_id=run.id,
                expected_kind=OrchestratorStepKind.FINANCIAL_STATEMENT_REFRESH,
                payload={
                    "company_id": candidate.company.id,
                    "legal_name": candidate.company.legal_name,
                    "cik": cik or "",
                    "fiscal_years": request.fiscal_years,
                },
            ),
            DelegationRequest(
                role=AgentRole.FINANCIAL_REPORT,
                run_id=run.id,
                step_id="refresh_filings",
                correlation_id=run.id,
                expected_kind=OrchestratorStepKind.FILING_REFRESH,
                payload={
                    "company_id": candidate.company.id,
                    "legal_name": candidate.company.legal_name,
                    "cik": cik or "",
                    "forms": list(request.filing_forms),
                    "limit": request.filing_limit,
                    "form_limits": dict(request.filing_form_limits),
                },
            ),
        )
        results = await asyncio.gather(
            *(self._step_dispatcher.dispatch(item, run=run) for item in delegations)
        )
        return tuple(result.handoff for result in results)

    async def _dispatch_specialists(
        self,
        request: OrchestratorResearchInput,
        run: OrchestratedResearchRun,
        candidate: CompanySearchCandidate,
        security: ResolvedSecurity,
        cik: str | None,
    ) -> tuple[AgentHandoff, ...]:
        delegations = (
            DelegationRequest(
                role=AgentRole.FINANCIAL_REPORT,
                run_id=run.id,
                step_id="financial_report_analysis",
                correlation_id=run.id,
                expected_kind=OrchestratorStepKind.FINANCIAL_REPORT_ANALYSIS,
                payload={
                    "company_id": candidate.company.id,
                    "legal_name": candidate.company.legal_name,
                    "cik": cik or "",
                },
            ),
            DelegationRequest(
                role=AgentRole.STOCK,
                run_id=run.id,
                step_id="stock_price_analysis",
                correlation_id=run.id,
                expected_kind=OrchestratorStepKind.STOCK_PRICE_ANALYSIS,
                payload={
                    "security_id": security.id,
                    "ticker": security.ticker,
                    "exchange_mic": security.exchange_mic,
                    "exchange_name": security.exchange_name,
                    "currency": security.currency,
                    "benchmark_symbol": request.benchmark_symbol,
                },
            ),
            DelegationRequest(
                role=AgentRole.CONTEXT,
                run_id=run.id,
                step_id="context_analysis",
                correlation_id=run.id,
                expected_kind=OrchestratorStepKind.CONTEXT_ANALYSIS,
                payload={
                    "query": request.query,
                    "company_symbols": [security.ticker],
                    "source_items": [item.to_dict() for item in request.context_source_items],
                },
            ),
        )
        results = await asyncio.gather(
            *(self._step_dispatcher.dispatch(item, run=run) for item in delegations)
        )
        return tuple(result.handoff for result in results)

    async def _dispatch_synthesis(self, run: OrchestratedResearchRun) -> AgentHandoff:
        result = await self._step_dispatcher.dispatch(
            DelegationRequest(
                role=AgentRole.SYNTHESIS,
                run_id=run.id,
                step_id="synthesis",
                correlation_id=run.id,
                expected_kind=OrchestratorStepKind.SYNTHESIS,
                payload={"handoff_ids": [handoff.id for handoff in run.handoffs]},
            ),
            run=run,
        )
        return result.handoff

    def _append_handoff(
        self,
        run: OrchestratedResearchRun,
        handoff: AgentHandoff,
        observer: Callable[[OrchestratedResearchRun], None] | None,
    ) -> OrchestratedResearchRun:
        return self._save(
            replace(
                run,
                handoffs=(*run.handoffs, handoff),
                updated_at=_aware_now(self._now()),
            ),
            observer,
        )

    def _save(
        self,
        run: OrchestratedResearchRun,
        observer: Callable[[OrchestratedResearchRun], None] | None,
    ) -> OrchestratedResearchRun:
        stored = self._run_store.save(run) if self._run_store is not None else run
        if observer is not None:
            observer(stored)
        return stored


def _final_status(handoffs: tuple[AgentHandoff, ...]) -> OrchestratorRunStatus:
    synthesis = next(
        (
            handoff
            for handoff in reversed(handoffs)
            if handoff.kind == OrchestratorStepKind.SYNTHESIS
        ),
        None,
    )
    if synthesis is None or synthesis.status == OrchestratorHandoffStatus.FAILED:
        return OrchestratorRunStatus.FAILED
    required = tuple(
        handoff
        for handoff in handoffs
        if handoff.kind
        in {
            OrchestratorStepKind.FINANCIAL_REPORT_ANALYSIS,
            OrchestratorStepKind.STOCK_PRICE_ANALYSIS,
            OrchestratorStepKind.CONTEXT_ANALYSIS,
        }
    )
    if synthesis.status != OrchestratorHandoffStatus.SUCCEEDED or any(
        handoff.status != OrchestratorHandoffStatus.SUCCEEDED for handoff in required
    ):
        return OrchestratorRunStatus.PARTIAL
    return OrchestratorRunStatus.COMPLETE


def _synthesis_summary(handoff: AgentHandoff) -> str | None:
    value = handoff.output.get("summary")
    return str(value).strip() if value is not None and str(value).strip() else None


def _candidate_identifier(
    candidate: CompanySearchCandidate,
    identifier_type: EntityIdentifierType,
) -> str | None:
    return next(
        (
            identifier.value
            for identifier in candidate.company.identifiers
            if identifier.identifier_type == identifier_type
        ),
        None,
    )


def _handoff(
    *,
    kind: OrchestratorStepKind,
    step_id: str,
    status: OrchestratorHandoffStatus,
    started_at: datetime,
    completed_at: datetime,
    input_summary: Mapping[str, str] | None = None,
    output: Mapping[str, object] | None = None,
    warnings: tuple[str, ...] = (),
    limitations: tuple[str, ...] = (),
    confidence: HandoffConfidence = HandoffConfidence.UNKNOWN,
    error_code: str | None = None,
    error_message: str | None = None,
) -> AgentHandoff:
    return AgentHandoff(
        id=f"handoff_{kind.value}_{uuid4().hex}",
        step_id=step_id,
        kind=kind,
        status=status,
        started_at=started_at,
        completed_at=completed_at,
        input_summary=dict(input_summary or {}),
        output=dict(output or {}),
        warnings=warnings,
        limitations=limitations,
        confidence=confidence,
        error_code=error_code,
        error_message=error_message,
    )


def _skipped_handoff(
    kind: OrchestratorStepKind,
    step_id: str,
    occurred_at: datetime,
    reason: str,
) -> AgentHandoff:
    return _handoff(
        kind=kind,
        step_id=step_id,
        status=OrchestratorHandoffStatus.SKIPPED,
        started_at=occurred_at,
        completed_at=occurred_at,
        limitations=(reason,),
    )


def _aware_now(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
