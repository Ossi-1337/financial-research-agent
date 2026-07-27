from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from financial_research_agent.context_analysis import (
    ContextScope,
    ContextSourceItem,
    ContextSourceType,
    NewsMacroSectorAgent,
    SourceReliability,
)
from financial_research_agent.domain import FinancialStatementType
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
from financial_research_agent.filings import (
    FilingChunk,
    FilingCompany,
    FilingDocument,
    FilingDocumentFormat,
    FilingError,
    FilingErrorCode,
    FilingIngestionResult,
    FilingSource,
    FilingStore,
)
from financial_research_agent.market_data import (
    HistoricalPriceBar,
    HistoricalPriceResult,
    MarketDataError,
    MarketDataErrorCode,
    MarketDataSource,
    MarketDataStore,
    MarketSecurity,
    calculate_price_metrics,
)
from financial_research_agent.orchestration import (
    AgentHandoff,
    AgentRole,
    DelegationResult,
    HandoffConfidence,
    OrchestratedResearchRun,
    OrchestratorExecutionPolicy,
    OrchestratorHandoffStatus,
    OrchestratorResearchInput,
    OrchestratorRunStatus,
    OrchestratorRunStore,
    OrchestratorStepKind,
    ResearchOrchestrator,
    default_orchestrator_plan,
)
from financial_research_agent.report_analysis import FinancialReportAnalysisAgent
from financial_research_agent.settings import Settings
from financial_research_agent.statements import (
    FinancialStatementCompany,
    FinancialStatementPeriod,
    FinancialStatementPeriodType,
    FinancialStatementResult,
    FinancialStatementSource,
    FinancialStatementStore,
    NormalizedFinancialStatement,
)
from financial_research_agent.stock_analysis import StockPriceAnalysisAgent
from financial_research_agent.web import ChatSessionStore, create_app

NOW = datetime(2026, 7, 5, 12, tzinfo=UTC)


def test_orchestrator_contracts_are_immutable_and_plan_is_sequential() -> None:
    step = default_orchestrator_plan()[0]
    handoff = AgentHandoff(
        id="handoff:resolution",
        step_id=step.id,
        kind=step.kind,
        status=OrchestratorHandoffStatus.SUCCEEDED,
        started_at=NOW,
        completed_at=NOW,
        input_summary={"query": "Apple"},
        output={"selected": "AAPL"},
    )

    with pytest.raises(FrozenInstanceError):
        handoff.status = OrchestratorHandoffStatus.FAILED
    assert all(not item.can_run_parallel for item in default_orchestrator_plan())
    assert default_orchestrator_plan()[0].kind == OrchestratorStepKind.COMPANY_RESOLUTION
    with pytest.raises(ValueError, match="query is required"):
        OrchestratorResearchInput(query=" ")


def test_orchestrator_runs_workflow_and_persists_specialist_handoffs(tmp_path: Path) -> None:
    run_store = RecordingRunStore(storage_path=tmp_path / "orchestrator_runs.json")
    orchestrator = _orchestrator(run_store=run_store)

    run = asyncio.run(
        orchestrator.run(
            OrchestratorResearchInput(
                query="Apple financial situation",
                context_source_items=_context_sources(),
            )
        )
    )
    reloaded = OrchestratorRunStore(storage_path=tmp_path / "orchestrator_runs.json").get(run.id)

    assert run.status == OrchestratorRunStatus.PARTIAL
    assert run.selected_company["legal_name"] == "TEST TOOL OUTPUT APPLE INC."
    assert run.selected_security["ticker"] == "AAPL"
    assert run.synthesis_summary is not None
    assert reloaded is not None
    assert reloaded.to_dict() == run.to_dict()
    kinds = [handoff.kind for handoff in run.handoffs]
    synthesis = run.handoffs[-1]
    synthesis_report = synthesis.output["report"]
    assert OrchestratorStepKind.FINANCIAL_REPORT_ANALYSIS in kinds
    assert OrchestratorStepKind.STOCK_PRICE_ANALYSIS in kinds
    assert OrchestratorStepKind.CONTEXT_ANALYSIS in kinds
    assert kinds[-1] == OrchestratorStepKind.SYNTHESIS
    assert synthesis_report["sections"]["current_situation"]
    assert synthesis_report["sections"]["strengths"]
    assert synthesis_report["sections"]["risks"]
    assert synthesis_report["scenarios"]["upside"]["direction"] == "upside"
    assert synthesis_report["scenarios"]["downside"]["direction"] == "downside"
    assert any(
        evidence_id.startswith("statement:") for evidence_id in synthesis_report["evidence_ids"]
    )
    assert any(
        point["source_handoff_ids"]
        for section_points in synthesis_report["sections"].values()
        for point in section_points
    )
    assert "does not provide buy, sell, hold" in synthesis_report["no_recommendation_notice"]
    assert any(_has_specialists_before_synthesis(snapshot) for snapshot in run_store.snapshots)


def test_orchestrator_recovers_when_one_refresh_fails(tmp_path: Path) -> None:
    run_store = OrchestratorRunStore(storage_path=tmp_path / "orchestrator_runs.json")
    orchestrator = _orchestrator(
        market_data_provider=FailingMarketDataProvider(),
        run_store=run_store,
    )

    run = asyncio.run(
        orchestrator.run(
            OrchestratorResearchInput(
                query="Apple financial situation",
                context_source_items=_context_sources(),
            )
        )
    )

    assert run.status == OrchestratorRunStatus.PARTIAL
    assert run_store.get(run.id) is not None
    failed = next(
        handoff
        for handoff in run.handoffs
        if handoff.kind == OrchestratorStepKind.MARKET_DATA_REFRESH
    )
    report = next(
        handoff
        for handoff in run.handoffs
        if handoff.kind == OrchestratorStepKind.FINANCIAL_REPORT_ANALYSIS
    )
    stock = next(
        handoff
        for handoff in run.handoffs
        if handoff.kind == OrchestratorStepKind.STOCK_PRICE_ANALYSIS
    )

    assert failed.status == OrchestratorHandoffStatus.FAILED
    assert failed.error_code == "provider_unavailable"
    assert report.status in {
        OrchestratorHandoffStatus.SUCCEEDED,
        OrchestratorHandoffStatus.PARTIAL,
    }
    assert stock.status == OrchestratorHandoffStatus.PARTIAL
    assert "No stored market data" in " ".join(stock.limitations)


def test_distributed_dispatch_runs_specialists_concurrently_in_stable_order(
    tmp_path: Path,
) -> None:
    dispatcher = RecordingDistributedDispatcher()
    orchestrator = _orchestrator(
        run_store=OrchestratorRunStore(storage_path=tmp_path / "orchestrator_runs.json"),
        step_dispatcher=dispatcher,
    )

    run = asyncio.run(
        orchestrator.run(
            OrchestratorResearchInput(
                query="Apple distributed research",
                context_source_items=_context_sources(),
            )
        )
    )
    specialist_kinds = [
        handoff.kind
        for handoff in run.handoffs
        if handoff.kind
        in {
            OrchestratorStepKind.FINANCIAL_REPORT_ANALYSIS,
            OrchestratorStepKind.STOCK_PRICE_ANALYSIS,
            OrchestratorStepKind.CONTEXT_ANALYSIS,
        }
    ]

    assert run.execution_policy == OrchestratorExecutionPolicy.DISTRIBUTED_A2A
    assert dispatcher.max_active == 3
    assert specialist_kinds == [
        OrchestratorStepKind.FINANCIAL_REPORT_ANALYSIS,
        OrchestratorStepKind.STOCK_PRICE_ANALYSIS,
        OrchestratorStepKind.CONTEXT_ANALYSIS,
    ]
    assert run.handoffs[-1].kind == OrchestratorStepKind.SYNTHESIS
    assert run.synthesis_summary == "Distributed deterministic synthesis."


def test_distributed_specialist_outage_stays_partial_without_local_fallback(
    tmp_path: Path,
) -> None:
    dispatcher = RecordingDistributedDispatcher(failing_role=AgentRole.STOCK)
    orchestrator = _orchestrator(
        run_store=OrchestratorRunStore(storage_path=tmp_path / "orchestrator_runs.json"),
        step_dispatcher=dispatcher,
    )

    run = asyncio.run(
        orchestrator.run(
            OrchestratorResearchInput(
                query="Apple distributed outage",
                context_source_items=_context_sources(),
            )
        )
    )
    stock = next(
        handoff
        for handoff in run.handoffs
        if handoff.kind == OrchestratorStepKind.STOCK_PRICE_ANALYSIS
    )

    assert run.status == OrchestratorRunStatus.PARTIAL
    assert stock.status == OrchestratorHandoffStatus.FAILED
    assert stock.error_code == "a2a_agent_unavailable"
    assert dispatcher.calls.count(AgentRole.STOCK) == 1


def test_orchestrator_marks_refresh_disabled_steps_as_skipped(tmp_path: Path) -> None:
    run_store = OrchestratorRunStore(storage_path=tmp_path / "orchestrator_runs.json")
    orchestrator = _orchestrator(run_store=run_store)

    run = asyncio.run(
        orchestrator.run(
            OrchestratorResearchInput(
                query="Apple financial situation",
                refresh=False,
                context_source_items=_context_sources(),
            )
        )
    )
    skipped_handoffs = tuple(
        handoff
        for handoff in run.handoffs
        if handoff.kind
        in {
            OrchestratorStepKind.MARKET_DATA_REFRESH,
            OrchestratorStepKind.FINANCIAL_STATEMENT_REFRESH,
            OrchestratorStepKind.FILING_REFRESH,
        }
    )

    assert run.status == OrchestratorRunStatus.PARTIAL
    assert {handoff.status for handoff in skipped_handoffs} == {OrchestratorHandoffStatus.SKIPPED}
    assert all(handoff.started_at == NOW for handoff in skipped_handoffs)
    assert all(handoff.completed_at == NOW for handoff in skipped_handoffs)
    assert run_store.get(run.id) is not None


def test_orchestrator_refreshes_benchmark_and_uses_per_form_filing_limits(
    tmp_path: Path,
) -> None:
    market_provider = RecordingMarketDataProvider()
    filing_provider = RecordingFilingProvider()
    orchestrator = _orchestrator(
        market_data_provider=market_provider,
        filing_provider=filing_provider,
        run_store=OrchestratorRunStore(storage_path=tmp_path / "orchestrator_runs.json"),
    )

    run = asyncio.run(
        orchestrator.run(
            OrchestratorResearchInput(
                query="TEST FIXTURE benchmark and filings",
                benchmark_symbol="SPY",
                filing_forms=("20-F", "6-K"),
                filing_form_limits={"20-F": 1, "6-K": 1},
                scenario_id="novo-nordisk",
                scenario_version="1.0.0",
                context_source_items=_context_sources(),
            )
        )
    )
    market_handoff = next(
        item for item in run.handoffs if item.kind == OrchestratorStepKind.MARKET_DATA_REFRESH
    )
    filing_handoff = next(
        item for item in run.handoffs if item.kind == OrchestratorStepKind.FILING_REFRESH
    )

    assert market_provider.calls == [("AAPL", "compact"), ("SPY", "compact")]
    assert "benchmark_history" in market_handoff.output
    assert filing_provider.calls == [("20-F", 1), ("6-K", 1)]
    assert [item["form_type"] for item in filing_handoff.output["filings"]["filings"]] == [
        "20-F",
        "6-K",
    ]
    assert run.scenario_id == "novo-nordisk"
    assert run.scenario_version == "1.0.0"


def test_orchestrator_no_company_match_stores_inspectable_failed_run(tmp_path: Path) -> None:
    run_store = OrchestratorRunStore(storage_path=tmp_path / "orchestrator_runs.json")
    orchestrator = _orchestrator(
        company_search_provider=NoMatchCompanySearchProvider(),
        run_store=run_store,
    )

    run = asyncio.run(orchestrator.run(OrchestratorResearchInput(query="Unknown company")))
    stored = run_store.get(run.id)

    assert run.status == OrchestratorRunStatus.FAILED
    assert stored is not None
    assert stored.handoffs[0].kind == OrchestratorStepKind.COMPANY_RESOLUTION
    assert stored.handoffs[0].status == OrchestratorHandoffStatus.FAILED
    assert "no company candidate" in stored.synthesis_summary.lower()


def test_orchestrator_web_endpoint_runs_and_returns_stored_run(tmp_path: Path) -> None:
    settings = Settings.from_env({"FRA_HOME": str(tmp_path)})
    run_store = OrchestratorRunStore(storage_path=tmp_path / "orchestrator_runs.json")
    client = TestClient(
        create_app(
            settings=settings,
            session_store=ChatSessionStore(),
            company_search_provider=FakeCompanySearchProvider(),
            market_data_provider=FakeMarketDataProvider(),
            market_data_store=MarketDataStore(storage_path=tmp_path / "market_data.json"),
            financial_statement_provider=FakeFinancialStatementProvider(),
            financial_statement_store=FinancialStatementStore(
                storage_path=tmp_path / "statements.json"
            ),
            filing_provider=FakeFilingProvider(),
            filing_store=FilingStore(storage_path=tmp_path / "filings.json"),
            orchestrator_run_store=run_store,
        )
    )

    response = client.post(
        "/api/orchestrator/research",
        json={
            "query": "Apple financial situation",
            "context_source_items": [_context_sources()[0].to_dict()],
        },
    )
    run = response.json()["run"]
    listed = client.get("/api/orchestrator/runs").json()["runs"]
    stored_response = client.get(f"/api/orchestrator/runs/{run['id']}").json()
    stored = stored_response["run"]
    status = client.get("/api/status").json()

    assert response.status_code == 200
    assert run["id"].startswith("orchestrator_run_")
    assert listed[0]["id"] == run["id"]
    assert stored["id"] == run["id"]
    assert response.json()["synthesis_report"]["sections"]["current_situation"]
    assert stored_response["synthesis_report"]["scenarios"]["upside"]["direction"] == "upside"
    assert status["orchestration"]["stored_run_count"] == 1
    assert status["orchestration"]["recommendations"] == "disabled"


class RecordingRunStore(OrchestratorRunStore):
    def __init__(self, *, storage_path: Path) -> None:
        self.snapshots: list[OrchestratedResearchRun] = []
        super().__init__(storage_path=storage_path)

    def save(self, run: OrchestratedResearchRun) -> OrchestratedResearchRun:
        saved = super().save(run)
        self.snapshots.append(saved)
        return saved


class FakeCompanySearchProvider:
    async def search(self, query: str, *, limit: int = 10) -> CompanySearchResult:
        source = SourceMetadata(
            provider="fake-company-search",
            provider_status="test fixture",
            source_url="https://example.test/company-search",
            retrieved_at=NOW,
            attribution="test fixture",
        )
        company = ResolvedCompany(
            id="fixture:company:apple",
            legal_name="TEST TOOL OUTPUT APPLE INC.",
            identifiers=(
                EntityIdentifier(EntityIdentifierType.CIK, "320193", source="fixture"),
                EntityIdentifier(EntityIdentifierType.TICKER, "AAPL", source="fixture"),
            ),
        )
        security = ResolvedSecurity(
            id="fixture:security:aapl",
            company_id=company.id,
            ticker="AAPL",
            name=company.legal_name,
            exchange_name="Nasdaq",
            currency="USD",
            identifiers=(EntityIdentifier(EntityIdentifierType.CIK, "320193", source="fixture"),),
        )
        return CompanySearchResult(
            query=query,
            status=CompanySearchStatus.REVIEW_REQUIRED,
            candidates=(
                CompanySearchCandidate(
                    company=company,
                    securities=(security,),
                    score=95,
                    match_reason=f"fixture limit {limit}",
                    source=source,
                ),
            ),
            source=source,
        )


class NoMatchCompanySearchProvider:
    async def search(self, query: str, *, limit: int = 10) -> CompanySearchResult:
        return CompanySearchResult(
            query=query,
            status=CompanySearchStatus.NO_MATCHES,
            candidates=(),
            source=SourceMetadata(
                provider="fake-company-search",
                provider_status="test fixture",
                source_url="https://example.test/company-search",
                retrieved_at=NOW,
                attribution="test fixture",
            ),
        )


class FakeMarketDataProvider:
    async def fetch_daily_prices(
        self,
        security: MarketSecurity,
        *,
        outputsize: str = "compact",
    ) -> HistoricalPriceResult:
        bars = tuple(
            HistoricalPriceBar(
                security=security,
                priced_at=date(2026, 6, 1) + timedelta(days=index),
                open=Decimal(100 + index),
                high=Decimal(101 + index),
                low=Decimal(99 + index),
                close=Decimal(100 + index),
                volume=1_000 + index,
            )
            for index in range(25)
        )
        source = MarketDataSource(
            provider="alpha-vantage",
            provider_status="test fixture",
            source_url="https://example.test/market-data",
            retrieved_at=NOW,
            data_as_of=date(2026, 6, 25),
            attribution="test fixture",
        )
        return HistoricalPriceResult(
            security=security,
            bars=bars,
            source=source,
            metrics=calculate_price_metrics(bars),
            warnings=(f"outputsize={outputsize}",),
        )

    async def fetch_quote(self, security: MarketSecurity):
        raise NotImplementedError


class RecordingMarketDataProvider(FakeMarketDataProvider):
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def fetch_daily_prices(
        self,
        security: MarketSecurity,
        *,
        outputsize: str = "compact",
    ) -> HistoricalPriceResult:
        self.calls.append((security.symbol, outputsize))
        return await super().fetch_daily_prices(security, outputsize=outputsize)


class FailingMarketDataProvider:
    async def fetch_daily_prices(
        self,
        _security: MarketSecurity,
        *,
        outputsize: str = "compact",
    ) -> HistoricalPriceResult:
        raise MarketDataError(
            code=MarketDataErrorCode.PROVIDER_UNAVAILABLE,
            message=f"market data unavailable for outputsize={outputsize}",
            provider="alpha-vantage",
            retryable=True,
        )

    async def fetch_quote(self, security: MarketSecurity):
        raise NotImplementedError


class FakeFinancialStatementProvider:
    async def fetch_statements(
        self,
        company: FinancialStatementCompany,
        *,
        fiscal_years: int = 3,
    ) -> FinancialStatementResult:
        source = FinancialStatementSource(
            provider="sec-companyfacts",
            provider_status="test fixture",
            source_url="https://example.test/companyfacts",
            retrieved_at=NOW,
            data_as_of=date(2025, 12, 31),
            attribution="test fixture",
        )
        latest = FinancialStatementPeriod(
            fiscal_year=2025,
            fiscal_period="FY",
            period_type=FinancialStatementPeriodType.ANNUAL,
            period_start=date(2025, 1, 1),
            period_end=date(2025, 12, 31),
            form="10-K",
            accession_number="fixture-2025",
            filed_at=date(2026, 1, 31),
        )
        previous = FinancialStatementPeriod(
            fiscal_year=2024,
            fiscal_period="FY",
            period_type=FinancialStatementPeriodType.ANNUAL,
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
            form="10-K",
            accession_number="fixture-2024",
            filed_at=date(2025, 1, 31),
        )
        return FinancialStatementResult(
            company=company,
            statements=(
                _statement(
                    company,
                    source,
                    latest,
                    FinancialStatementType.INCOME_STATEMENT,
                    {"revenues": "1000"},
                ),
                _statement(
                    company,
                    source,
                    previous,
                    FinancialStatementType.INCOME_STATEMENT,
                    {"revenues": "900"},
                ),
                _statement(
                    company,
                    source,
                    latest,
                    FinancialStatementType.KEY_RATIOS,
                    {
                        "gross_margin": "0.55",
                        "operating_margin": "0.30",
                        "net_margin": "0.25",
                        "free_cash_flow_proxy": "200",
                        "current_ratio": "1.8",
                    },
                ),
                _statement(
                    company,
                    source,
                    latest,
                    FinancialStatementType.CASH_FLOW,
                    {"operating_cash_flow": "250"},
                ),
                _statement(
                    company,
                    source,
                    latest,
                    FinancialStatementType.BALANCE_SHEET,
                    {
                        "cash_and_cash_equivalents": "100",
                        "liabilities": "400",
                        "stockholders_equity": "600",
                    },
                ),
            ),
            source=source,
            warnings=(f"fiscal_years={fiscal_years}",),
        )


class FakeFilingProvider:
    async def ingest_latest(
        self,
        company: FilingCompany,
        *,
        forms: tuple[str, ...] = ("10-K", "10-Q"),
        limit: int = 1,
    ) -> FilingIngestionResult:
        source = FilingSource(
            provider="sec-edgar",
            provider_status="test fixture",
            source_url="https://example.test/submissions",
            retrieved_at=NOW,
            data_as_of=date(2026, 1, 31),
            attribution="test fixture",
        )
        filing = FilingDocument(
            id="fixture:filing:10-k",
            company=company,
            form_type=forms[0],
            accession_number="0000320193-26-000001",
            filing_date=date(2026, 1, 31),
            report_date=date(2025, 12, 31),
            publication_date=date(2026, 1, 31),
            document_url="https://example.test/aapl-10k.htm",
            source_url=source.source_url,
            document_format=FilingDocumentFormat.HTML,
            retrieved_at=NOW,
            local_raw_path="raw/aapl-10k.htm",
            local_text_path="text/aapl-10k.txt",
            source=source,
            chunk_ids=("fixture:chunk:1",),
        )
        chunk = FilingChunk(
            id="fixture:chunk:1",
            filing_id=filing.id,
            chunk_index=0,
            text=(
                "TEST TOOL OUTPUT guidance outlook risk uncertainty critical accounting "
                f"forms={','.join(forms)} limit={limit}"
            ),
            char_start=0,
            char_end=100,
            section_heading="Item 1A. Risk Factors",
            source_url=filing.document_url,
            accession_number=filing.accession_number,
            form_type=filing.form_type,
            metadata={"fixture": "true"},
        )
        return FilingIngestionResult(
            company=company,
            filings=(filing,),
            chunks=(chunk,),
            source=source,
            warnings=(),
        )


class FailingFilingProvider(FakeFilingProvider):
    async def ingest_latest(
        self,
        _company: FilingCompany,
        *,
        forms: tuple[str, ...] = ("10-K", "10-Q"),
        limit: int = 1,
    ) -> FilingIngestionResult:
        raise FilingError(
            code=FilingErrorCode.PROVIDER_UNAVAILABLE,
            message=f"filings unavailable for forms={','.join(forms)} limit={limit}",
            provider="sec-edgar",
            retryable=True,
        )


class RecordingFilingProvider(FakeFilingProvider):
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    async def ingest_latest(
        self,
        company: FilingCompany,
        *,
        forms: tuple[str, ...] = ("10-K", "10-Q"),
        limit: int = 1,
    ) -> FilingIngestionResult:
        self.calls.append((forms[0], limit))
        return await super().ingest_latest(company, forms=forms, limit=limit)


def _orchestrator(
    *,
    company_search_provider=None,
    market_data_provider=None,
    filing_provider=None,
    run_store: OrchestratorRunStore | None = None,
    step_dispatcher=None,
) -> ResearchOrchestrator:
    market_store = MarketDataStore()
    statement_store = FinancialStatementStore()
    filing_store = FilingStore()
    return ResearchOrchestrator(
        company_search_provider=company_search_provider or FakeCompanySearchProvider(),
        market_data_provider=market_data_provider or FakeMarketDataProvider(),
        market_data_store=market_store,
        financial_statement_provider=FakeFinancialStatementProvider(),
        financial_statement_store=statement_store,
        filing_provider=filing_provider or FakeFilingProvider(),
        filing_store=filing_store,
        financial_report_agent=FinancialReportAnalysisAgent(
            statement_store=statement_store,
            filing_store=filing_store,
            statement_provider="sec-companyfacts",
            filing_provider="sec-edgar",
        ),
        stock_price_agent=StockPriceAnalysisAgent(
            market_data_store=market_store,
            market_data_provider="alpha-vantage",
        ),
        context_agent=NewsMacroSectorAgent(now=lambda: NOW),
        run_store=run_store,
        step_dispatcher=step_dispatcher,
        now=lambda: NOW,
    )


class RecordingDistributedDispatcher:
    def __init__(self, *, failing_role: AgentRole | None = None) -> None:
        self.active = 0
        self.max_active = 0
        self.failing_role = failing_role
        self.calls: list[AgentRole] = []

    async def dispatch(self, request, *, run=None) -> DelegationResult:
        self.calls.append(request.role)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        if request.expected_kind != OrchestratorStepKind.SYNTHESIS:
            await asyncio.sleep(0.01)
        self.active -= 1
        output = (
            {"summary": "Distributed deterministic synthesis.", "report": {"status": "partial"}}
            if request.expected_kind == OrchestratorStepKind.SYNTHESIS
            else {"analysis": {"status": "test fixture"}}
        )
        status = (
            OrchestratorHandoffStatus.FAILED
            if request.role == self.failing_role
            else OrchestratorHandoffStatus.SUCCEEDED
        )
        return DelegationResult(
            handoff=AgentHandoff(
                id=f"handoff:{request.step_id}",
                step_id=request.step_id,
                kind=request.expected_kind,
                status=status,
                started_at=NOW,
                completed_at=NOW,
                output=output,
                confidence=HandoffConfidence.MEDIUM,
                error_code=(
                    "a2a_agent_unavailable" if status == OrchestratorHandoffStatus.FAILED else None
                ),
                error_message=(
                    "Specialist delegation failed safely."
                    if status == OrchestratorHandoffStatus.FAILED
                    else None
                ),
            )
        )


def _statement(
    company: FinancialStatementCompany,
    source: FinancialStatementSource,
    period: FinancialStatementPeriod,
    statement_type: FinancialStatementType,
    line_items: dict[str, str],
) -> NormalizedFinancialStatement:
    return NormalizedFinancialStatement(
        id=f"fixture:{statement_type.value}:{period.fiscal_year}:{len(line_items)}",
        company=company,
        statement_type=statement_type,
        period=period,
        currency="USD",
        line_items={key: Decimal(value) for key, value in line_items.items()},
        source=source,
    )


def _context_sources() -> tuple[ContextSourceItem, ...]:
    return (
        _context_source("company-context", ContextScope.COMPANY, ContextSourceType.COMPANY_NEWS),
        _context_source("macro-context", ContextScope.MACRO, ContextSourceType.RATES),
        _context_source("sector-context", ContextScope.SECTOR, ContextSourceType.SECTOR_CONTEXT),
    )


def _context_source(
    item_id: str,
    scope: ContextScope,
    source_type: ContextSourceType,
) -> ContextSourceItem:
    return ContextSourceItem(
        id=item_id,
        title=f"{scope.value} context",
        summary="TEST TOOL OUTPUT source-linked context.",
        source_url=f"https://example.test/{item_id}",
        source_name="Context Fixture",
        source_type=source_type,
        reliability=SourceReliability.OFFICIAL,
        scope=scope,
        retrieved_at=NOW,
        published_at=NOW,
        company_symbols=("AAPL",) if scope == ContextScope.COMPANY else (),
    )


def _has_specialists_before_synthesis(run: OrchestratedResearchRun) -> bool:
    kinds = [handoff.kind for handoff in run.handoffs]
    return (
        OrchestratorStepKind.FINANCIAL_REPORT_ANALYSIS in kinds
        and OrchestratorStepKind.STOCK_PRICE_ANALYSIS in kinds
        and OrchestratorStepKind.CONTEXT_ANALYSIS in kinds
        and OrchestratorStepKind.SYNTHESIS not in kinds
    )
