from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping

from financial_research_agent.llm import (
    ChatMessage,
    ChatProvider,
    ChatRequest,
    MessageRole,
    ProviderError,
)
from financial_research_agent.orchestration import (
    OrchestratedResearchRun,
    OrchestratorHandoffStatus,
    OrchestratorResearchInput,
    OrchestratorRunStatus,
    OrchestratorStepKind,
    ResearchOrchestrator,
)
from financial_research_agent.report_exports import (
    ReportEvidenceIndex,
    ReportExportError,
    ReportExportService,
    build_report_evidence_index,
)
from financial_research_agent.settings import DEFAULT_SEC_USER_AGENT, Settings

from .context import load_context_snapshot
from .contracts import (
    ScenarioCatalog,
    ScenarioCheck,
    ScenarioCheckStatus,
    ScenarioDefinition,
    ScenarioError,
    ScenarioErrorCode,
    ScenarioExecutionResult,
    ScenarioExecutionStatus,
    ScenarioLocalQA,
)

_SOURCE_MARKER = re.compile(r"\[S[1-9][0-9]*\]")


class ScenarioRunner:
    def __init__(
        self,
        *,
        settings: Settings,
        catalog: ScenarioCatalog,
        orchestrator: ResearchOrchestrator,
        export_service: ReportExportService,
        chat_provider: ChatProvider | None = None,
        chat_model: str | None = None,
    ) -> None:
        self.settings = settings
        self.catalog = catalog
        self.orchestrator = orchestrator
        self.export_service = export_service
        self.chat_provider = chat_provider
        self.chat_model = chat_model

    def prepare(self, scenario_id: str, *, refresh: bool = True) -> OrchestratorResearchInput:
        scenario = self.catalog.get(scenario_id)
        self._preflight(refresh=refresh)
        context = load_context_snapshot(
            scenario.context_resource,
            scenario_id=scenario.id,
        )
        return OrchestratorResearchInput(
            query=scenario.query,
            refresh=refresh,
            fiscal_years=scenario.fiscal_years,
            filing_forms=tuple(scenario.filing_form_limits),
            filing_form_limits=scenario.filing_form_limits,
            market_outputsize=scenario.market_outputsize,
            benchmark_symbol=scenario.benchmark_symbol,
            context_source_items=context,
            scenario_id=scenario.id,
            scenario_version=scenario.version,
        )

    async def run(
        self,
        scenario_id: str,
        *,
        refresh: bool = True,
        with_local_qa: bool = False,
    ) -> ScenarioExecutionResult:
        scenario = self.catalog.get(scenario_id)
        run = await self.orchestrator.run(self.prepare(scenario_id, refresh=refresh))
        return await self.finalize(run, scenario=scenario, with_local_qa=with_local_qa)

    async def finalize(
        self,
        run: OrchestratedResearchRun,
        *,
        scenario: ScenarioDefinition | None = None,
        with_local_qa: bool = False,
    ) -> ScenarioExecutionResult:
        definition = scenario or self.catalog.get(run.scenario_id or "")
        evidence = build_report_evidence_index(
            run,
            redaction_policy=self.export_service.redaction_policy,
        )
        checks = list(_scenario_checks(definition, run, evidence))
        snapshot = None
        try:
            snapshot = self.export_service.export(run)
            checks.append(
                ScenarioCheck(
                    id="report_export",
                    status=ScenarioCheckStatus.PASSED,
                    message="Markdown, HTML, and PDF snapshots were created.",
                    details={"export_id": snapshot.export_id},
                )
            )
        except ReportExportError as exc:
            checks.append(
                ScenarioCheck(
                    id="report_export",
                    status=ScenarioCheckStatus.FAILED,
                    message="Scenario report export failed.",
                    details={"error": exc.code.value},
                )
            )

        local_qa = None
        if with_local_qa:
            try:
                local_qa = await self._local_qa(run, evidence)
                checks.append(
                    ScenarioCheck(
                        id="local_qa",
                        status=local_qa.status,
                        message="Optional source-bounded local Q&A completed.",
                        details={"provider": local_qa.provider, "model": local_qa.model},
                    )
                )
            except (ProviderError, ScenarioError) as exc:
                checks.append(
                    ScenarioCheck(
                        id="local_qa",
                        status=ScenarioCheckStatus.WARNING,
                        message="Optional source-bounded local Q&A was unavailable.",
                        details={"error": getattr(exc, "code", "local_qa_unavailable")},
                    )
                )
        return ScenarioExecutionResult(
            scenario=definition,
            status=_execution_status(checks),
            run=run,
            checks=tuple(checks),
            export=snapshot,
            local_qa=local_qa,
        )

    def _preflight(self, *, refresh: bool) -> None:
        if not refresh:
            return
        data_sources = self.settings.data_sources
        if (
            data_sources.market_data_provider == "alpha-vantage"
            and data_sources.alpha_vantage_api_key is None
        ):
            raise ScenarioError(
                ScenarioErrorCode.MISSING_MARKET_DATA_CREDENTIALS,
                "FRA_ALPHA_VANTAGE_API_KEY is required for live scenario refresh.",
            )
        if data_sources.sec_user_agent == DEFAULT_SEC_USER_AGENT or ".local" in (
            data_sources.sec_user_agent.lower()
        ):
            raise ScenarioError(
                ScenarioErrorCode.INVALID_SEC_USER_AGENT,
                "Set FRA_SEC_USER_AGENT to an identifying application and real contact address.",
            )

    async def _local_qa(
        self,
        run: OrchestratedResearchRun,
        evidence: ReportEvidenceIndex,
    ) -> ScenarioLocalQA:
        if self.chat_provider is None:
            raise ScenarioError(
                ScenarioErrorCode.LOCAL_QA_UNAVAILABLE,
                "Optional local Q&A requires a configured chat provider.",
            )
        source_text = "\n".join(
            f"{source.marker} {source.source_name or 'Source'}: "
            f"{source.quote or 'No bounded excerpt available.'}"
            for source in evidence.sources
            if source.resolved
        )[:12_000]
        if source_text == "":
            raise ScenarioError(
                ScenarioErrorCode.LOCAL_QA_UNAVAILABLE,
                "Optional local Q&A requires resolved report evidence.",
            )
        response = await self.chat_provider.chat(
            ChatRequest(
                model=self.chat_model,
                temperature=0,
                max_output_tokens=400,
                messages=(
                    ChatMessage(
                        role=MessageRole.SYSTEM,
                        content=(
                            "Answer only from supplied source markers. Cite markers for factual "
                            "claims. State missing evidence. Do not give investment advice."
                        ),
                    ),
                    ChatMessage(
                        role=MessageRole.USER,
                        content=(
                            "Question: Summarize Novo Nordisk's current situation.\n"
                            f"Deterministic synthesis: {run.synthesis_summary or 'Unavailable'}\n"
                            f"Sources:\n{source_text}"
                        ),
                    ),
                ),
            )
        )
        available_markers = {source.marker for source in evidence.sources if source.resolved}
        claimed_markers = tuple(dict.fromkeys(_SOURCE_MARKER.findall(response.message.content)))
        markers = tuple(marker for marker in claimed_markers if marker in available_markers)
        return ScenarioLocalQA(
            status=(
                ScenarioCheckStatus.PASSED
                if markers and len(markers) == len(claimed_markers)
                else ScenarioCheckStatus.WARNING
            ),
            answer=response.message.content,
            provider=response.provider,
            model=response.model,
            source_markers=markers,
        )


def _scenario_checks(
    scenario: ScenarioDefinition,
    run: OrchestratedResearchRun,
    evidence: ReportEvidenceIndex,
) -> tuple[ScenarioCheck, ...]:
    company = _mapping(run.selected_company)
    security = _mapping(run.selected_security)
    identifiers = tuple(_mapping(item) for item in _sequence(company.get("identifiers")))
    cik_values = {str(item.get("value")) for item in identifiers if item.get("type") == "cik"}
    checks = [
        _equality_check("company_cik", scenario.expected_cik, cik_values, "resolved CIK"),
        _equality_check(
            "primary_security",
            scenario.preferred_ticker,
            {str(security.get("ticker"))},
            "primary ticker",
        ),
        _equality_check(
            "primary_exchange",
            scenario.preferred_exchange.casefold(),
            {str(security.get("exchange_name", "")).casefold()},
            "primary exchange",
        ),
    ]
    handoffs = {handoff.kind: handoff for handoff in run.handoffs}
    market = handoffs.get(OrchestratorStepKind.MARKET_DATA_REFRESH)
    market_output = _mapping(market.output) if market else {}
    primary_history = _mapping(market_output.get("history"))
    benchmark_history = _mapping(market_output.get("benchmark_history"))
    checks.append(
        _presence_check(
            "market_and_benchmark",
            bool(_sequence(primary_history.get("bars")))
            and bool(_sequence(benchmark_history.get("bars"))),
            "Primary and benchmark market histories are available.",
            details={
                "primary_bar_count": len(_sequence(primary_history.get("bars"))),
                "benchmark_bar_count": len(_sequence(benchmark_history.get("bars"))),
            },
        )
    )
    statement = handoffs.get(OrchestratorStepKind.FINANCIAL_STATEMENT_REFRESH)
    statement_payload = _mapping(_mapping(statement.output).get("statements")) if statement else {}
    statements = tuple(_mapping(item) for item in _sequence(statement_payload.get("statements")))
    annual_years = {
        _mapping(item.get("period")).get("fiscal_year")
        for item in statements
        if item.get("currency") == "DKK"
        and _mapping(item.get("period")).get("period_type") == "annual"
        and "ifrs-full" in _sequence(_mapping(item.get("source")).get("taxonomy_namespaces"))
    }
    checks.append(
        _presence_check(
            "ifrs_dkk_statements",
            len(annual_years) >= 2,
            "At least two annual DKK statement periods are available.",
            details={"period_count": len(annual_years)},
        )
    )
    filing = handoffs.get(OrchestratorStepKind.FILING_REFRESH)
    filing_payload = _mapping(_mapping(filing.output).get("filings")) if filing else {}
    filing_form_counts = Counter(
        str(item.get("form_type"))
        for item in _sequence(filing_payload.get("filings"))
        if isinstance(item, Mapping)
    )
    filing_limits_match = all(
        filing_form_counts[form] == expected_count
        for form, expected_count in scenario.filing_form_limits.items()
    )
    checks.append(
        _presence_check(
            "required_filings",
            filing_limits_match,
            "Required filing forms are stored at their configured limits.",
            details={"form_counts": dict(sorted(filing_form_counts.items()))},
        )
    )
    specialist_kinds = {
        OrchestratorStepKind.FINANCIAL_REPORT_ANALYSIS,
        OrchestratorStepKind.STOCK_PRICE_ANALYSIS,
        OrchestratorStepKind.CONTEXT_ANALYSIS,
        OrchestratorStepKind.SYNTHESIS,
    }
    specialist_statuses = {
        kind: handoffs[kind].status for kind in specialist_kinds if kind in handoffs
    }
    missing_or_failed = specialist_kinds.difference(specialist_statuses) or any(
        status in {OrchestratorHandoffStatus.FAILED, OrchestratorHandoffStatus.SKIPPED}
        for status in specialist_statuses.values()
    )
    partial = any(
        status == OrchestratorHandoffStatus.PARTIAL for status in specialist_statuses.values()
    )
    checks.append(
        ScenarioCheck(
            id="specialists_and_synthesis",
            status=(
                ScenarioCheckStatus.FAILED
                if missing_or_failed
                else ScenarioCheckStatus.WARNING
                if partial or run.status == OrchestratorRunStatus.PARTIAL
                else ScenarioCheckStatus.PASSED
            ),
            message=(
                "Specialist handoffs and deterministic synthesis are available."
                if not missing_or_failed
                else "One or more required specialist handoffs or synthesis outputs failed."
            ),
            details={
                "handoff_statuses": {
                    kind.value: status.value
                    for kind, status in sorted(
                        specialist_statuses.items(),
                        key=lambda item: item[0].value,
                    )
                },
                "run_status": run.status.value,
            },
        )
    )
    checks.append(
        _presence_check(
            "inspectable_evidence",
            any(source.resolved for source in evidence.sources),
            "Resolved evidence sources are inspectable.",
            details={
                "resolved": sum(source.resolved for source in evidence.sources),
                "unresolved": len(evidence.unresolved_evidence_ids),
            },
        )
    )
    return tuple(checks)


def _equality_check(check_id: str, expected: str, actual: set[str], label: str) -> ScenarioCheck:
    passed = expected in actual
    return ScenarioCheck(
        id=check_id,
        status=ScenarioCheckStatus.PASSED if passed else ScenarioCheckStatus.FAILED,
        message=f"Expected {label} {'was resolved' if passed else 'was not resolved'}.",
        details={"expected": expected, "actual": sorted(actual)},
    )


def _presence_check(
    check_id: str,
    present: bool,
    message: str,
    *,
    details: Mapping[str, object] | None = None,
) -> ScenarioCheck:
    return ScenarioCheck(
        id=check_id,
        status=ScenarioCheckStatus.PASSED if present else ScenarioCheckStatus.FAILED,
        message=message,
        details=details or {},
    )


def _execution_status(checks: list[ScenarioCheck]) -> ScenarioExecutionStatus:
    required_checks = tuple(check for check in checks if check.id != "local_qa")
    if any(check.status == ScenarioCheckStatus.FAILED for check in required_checks):
        return ScenarioExecutionStatus.FAILED
    if any(check.status == ScenarioCheckStatus.WARNING for check in required_checks):
        return ScenarioExecutionStatus.PARTIAL
    return ScenarioExecutionStatus.COMPLETE


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> tuple[object, ...]:
    return tuple(value) if isinstance(value, (list, tuple)) else ()
