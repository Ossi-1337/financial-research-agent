from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from time import perf_counter_ns
from uuid import uuid4

from financial_research_agent.agents import (
    AgentRole as PromptAgentRole,
)
from financial_research_agent.agents import (
    AgentRuntimeError,
    AgentRuntimeResolver,
    PromptCatalog,
    PromptContract,
    StructuredAgentResult,
    StructuredAgentRunner,
    create_default_prompt_catalog,
)
from financial_research_agent.context_analysis import (
    ContextAnalysisStatus,
    ContextScope,
    ContextSourceItem,
    ContextSourceType,
    NewsMacroSectorAgent,
    SourceReliability,
)
from financial_research_agent.filings import (
    FilingCompany,
    FilingError,
    FilingIngestionResult,
    FilingProvider,
    FilingStore,
)
from financial_research_agent.llm import ProviderError
from financial_research_agent.market_data import (
    MarketDataError,
    MarketDataProvider,
    MarketDataStore,
    MarketSecurity,
)
from financial_research_agent.orchestration import (
    AgentExecutionMetadata,
    AgentExecutionMode,
    AgentHandoff,
    AgentRole,
    DelegationRequest,
    HandoffConfidence,
    OrchestratedResearchRun,
    OrchestratorHandoffStatus,
    OrchestratorStepKind,
)
from financial_research_agent.report_analysis import (
    FinancialReportAnalysisAgent,
    FinancialReportAnalysisCompany,
    FinancialReportAnalysisStatus,
)
from financial_research_agent.statements import (
    FinancialStatementCompany,
    FinancialStatementError,
    FinancialStatementProvider,
    FinancialStatementStore,
)
from financial_research_agent.stock_analysis import (
    StockPriceAnalysisAgent,
    StockPriceAnalysisSecurity,
    StockPriceAnalysisStatus,
)
from financial_research_agent.synthesis import SynthesisAgent, SynthesisReportStatus
from financial_research_agent.tools import (
    ToolContext,
    ToolPermission,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)


class SpecialistExecutionService:
    def __init__(
        self,
        *,
        financial_report_agent: FinancialReportAnalysisAgent,
        stock_price_agent: StockPriceAnalysisAgent,
        context_agent: NewsMacroSectorAgent,
        market_data_provider: MarketDataProvider | None = None,
        market_data_store: MarketDataStore | None = None,
        financial_statement_provider: FinancialStatementProvider | None = None,
        financial_statement_store: FinancialStatementStore | None = None,
        filing_provider: FilingProvider | None = None,
        filing_store: FilingStore | None = None,
        synthesis_agent: SynthesisAgent | None = None,
        run_store: object | None = None,
        agent_runtime: AgentRuntimeResolver,
        prompt_catalog: PromptCatalog | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.financial_report_agent = financial_report_agent
        self.stock_price_agent = stock_price_agent
        self.context_agent = context_agent
        self.market_data_provider = market_data_provider
        self.market_data_store = market_data_store
        self.financial_statement_provider = financial_statement_provider
        self.financial_statement_store = financial_statement_store
        self.filing_provider = filing_provider
        self.filing_store = filing_store
        self.synthesis_agent = synthesis_agent or SynthesisAgent()
        self.run_store = run_store
        self.agent_runtime = agent_runtime
        self.prompt_catalog = prompt_catalog or create_default_prompt_catalog()
        self._now = now or (lambda: datetime.now(UTC))

    async def execute(self, request: DelegationRequest) -> AgentHandoff:
        if request.expected_kind == OrchestratorStepKind.MARKET_DATA_REFRESH:
            return await self._refresh_market_data(request)
        if request.expected_kind == OrchestratorStepKind.FINANCIAL_STATEMENT_REFRESH:
            return await self._refresh_financial_statements(request)
        if request.expected_kind == OrchestratorStepKind.FILING_REFRESH:
            return await self._refresh_filings(request)
        if request.role == AgentRole.FINANCIAL_REPORT:
            return await self._financial_report(request)
        if request.role == AgentRole.STOCK:
            return await self._stock(request)
        if request.role == AgentRole.CONTEXT:
            return await self._context(request)
        if request.role == AgentRole.SYNTHESIS:
            return await self._synthesis(request)
        raise ValueError(f"unsupported specialist role: {request.role.value}")

    def _agent_runner(self, request: DelegationRequest) -> StructuredAgentRunner:
        run = (
            self.run_store.get(request.run_id)
            if self.run_store is not None and hasattr(self.run_store, "get")
            else None
        )
        if (
            isinstance(run, OrchestratedResearchRun)
            and run.agent_provider is not None
            and run.agent_model is not None
        ):
            runtime = self.agent_runtime.resolve_selection(
                provider_name=run.agent_provider,
                model=run.agent_model,
                require_research=True,
            )
        else:
            runtime = self.agent_runtime.resolve(require_research=True)
        return StructuredAgentRunner(
            runtime.provider,
            model=runtime.model,
            max_output_tokens=runtime.max_output_tokens,
        )

    async def _refresh_market_data(self, request: DelegationRequest) -> AgentHandoff:
        started_at = _aware_now(self._now())
        if request.role != AgentRole.STOCK:
            raise ValueError("market data refresh requires stock specialist")
        store = _required_dependency(self.market_data_store, "market_data_store")
        security = MarketSecurity(
            symbol=_text(request.payload, "ticker"),
            security_id=_optional_text(request.payload.get("security_id")),
            exchange_mic=_optional_text(request.payload.get("exchange_mic")),
            exchange_name=_optional_text(request.payload.get("exchange_name")),
            currency=_optional_text(request.payload.get("currency")),
        )
        outputsize = _text(request.payload, "outputsize")
        benchmark_symbol = _optional_text(request.payload.get("benchmark_symbol"))
        stored = store.get_history(symbol=security.symbol, now=started_at)
        primary_cache_reused = _is_fresh_cached(stored, store=store, now=started_at)
        try:
            if not primary_cache_reused:
                provider = _required_dependency(
                    self.market_data_provider,
                    "market_data_provider",
                )
                history = await provider.fetch_daily_prices(security, outputsize=outputsize)
                stored = store.save_history(history)
        except MarketDataError as exc:
            return _provider_failure(request, started_at, self._now(), exc.code.value, exc.message)
        if stored is None:
            raise RuntimeError("market data refresh produced no stored result")

        output: dict[str, object] = {"history": stored.to_dict()}
        warnings = list(stored.warnings)
        if primary_cache_reused:
            warnings.append("Fresh cached market data reused according to configured TTL.")
        limitations: list[str] = []
        status = OrchestratorHandoffStatus.SUCCEEDED
        error_code = None
        error_message = None
        if benchmark_symbol is not None and benchmark_symbol != security.symbol:
            try:
                stored_benchmark = store.get_history(symbol=benchmark_symbol, now=started_at)
                benchmark_cache_reused = _is_fresh_cached(
                    stored_benchmark,
                    store=store,
                    now=started_at,
                )
                if not benchmark_cache_reused:
                    provider = _required_dependency(
                        self.market_data_provider,
                        "market_data_provider",
                    )
                    benchmark = await provider.fetch_daily_prices(
                        MarketSecurity(
                            symbol=benchmark_symbol,
                            security_id=f"benchmark:{benchmark_symbol}",
                        ),
                        outputsize=outputsize,
                    )
                    stored_benchmark = store.save_history(benchmark)
                if stored_benchmark is None:
                    raise RuntimeError("benchmark refresh produced no stored result")
                output["benchmark_history"] = stored_benchmark.to_dict()
                warnings.extend(stored_benchmark.warnings)
                if benchmark_cache_reused:
                    warnings.append(
                        "Fresh cached benchmark data reused according to configured TTL."
                    )
            except MarketDataError as exc:
                status = OrchestratorHandoffStatus.PARTIAL
                error_code = exc.code.value
                error_message = exc.message
                limitations.append(
                    f"Benchmark {benchmark_symbol} could not be refreshed: {exc.message}"
                )
        return _handoff(
            request=request,
            status=status,
            started_at=started_at,
            completed_at=_aware_now(self._now()),
            input_summary={
                "symbol": security.symbol,
                "outputsize": outputsize,
                **({"benchmark_symbol": benchmark_symbol} if benchmark_symbol else {}),
            },
            output=output,
            warnings=tuple(dict.fromkeys(warnings)),
            limitations=tuple(limitations),
            confidence=HandoffConfidence.HIGH,
            error_code=error_code,
            error_message=error_message,
        )

    async def _refresh_financial_statements(
        self,
        request: DelegationRequest,
    ) -> AgentHandoff:
        started_at = _aware_now(self._now())
        if request.role != AgentRole.FINANCIAL_REPORT:
            raise ValueError("statement refresh requires financial-report specialist")
        cik = _optional_text(request.payload.get("cik"))
        if cik is None:
            return _skipped_handoff(
                request,
                started_at,
                "No CIK was available from company resolution.",
            )
        store = _required_dependency(
            self.financial_statement_store,
            "financial_statement_store",
        )
        fiscal_years = _positive_int(request.payload, "fiscal_years")
        company = FinancialStatementCompany(
            cik=cik,
            company_id=_optional_text(request.payload.get("company_id")),
            legal_name=_optional_text(request.payload.get("legal_name")),
        )
        stored = store.get_result(cik=cik, now=started_at)
        cache_reused = _is_fresh_cached(stored, store=store, now=started_at)
        try:
            if not cache_reused:
                provider = _required_dependency(
                    self.financial_statement_provider,
                    "financial_statement_provider",
                )
                result = await provider.fetch_statements(company, fiscal_years=fiscal_years)
                stored = store.save_result(result)
        except FinancialStatementError as exc:
            return _provider_failure(request, started_at, self._now(), exc.code.value, exc.message)
        if stored is None:
            raise RuntimeError("financial statement refresh produced no stored result")
        warnings = list(stored.warnings)
        if cache_reused:
            warnings.append("Fresh cached financial statements reused according to configured TTL.")
        return _handoff(
            request=request,
            status=OrchestratorHandoffStatus.SUCCEEDED,
            started_at=started_at,
            completed_at=_aware_now(self._now()),
            input_summary={"cik": cik, "fiscal_years": str(fiscal_years)},
            output={"statements": stored.to_dict()},
            warnings=tuple(dict.fromkeys(warnings)),
            confidence=HandoffConfidence.HIGH,
        )

    async def _refresh_filings(self, request: DelegationRequest) -> AgentHandoff:
        started_at = _aware_now(self._now())
        if request.role != AgentRole.FINANCIAL_REPORT:
            raise ValueError("filing refresh requires financial-report specialist")
        cik = _optional_text(request.payload.get("cik"))
        if cik is None:
            return _skipped_handoff(
                request,
                started_at,
                "No CIK was available from company resolution.",
            )
        store = _required_dependency(self.filing_store, "filing_store")
        forms = _text_tuple(request.payload, "forms")
        limit = _positive_int(request.payload, "limit")
        form_limits = _positive_int_mapping(request.payload.get("form_limits", {}))
        company = FilingCompany(
            cik=cik,
            company_id=_optional_text(request.payload.get("company_id")),
            legal_name=_optional_text(request.payload.get("legal_name")),
        )
        stored = store.get_result(cik=cik, now=started_at)
        cache_reused = _is_fresh_cached(stored, store=store, now=started_at)
        partial_errors: tuple[str, ...] = ()
        try:
            if not cache_reused:
                provider = _required_dependency(self.filing_provider, "filing_provider")
                result, partial_errors = await _ingest_filings(
                    provider,
                    company,
                    forms=forms,
                    limit=limit,
                    form_limits=form_limits,
                )
                stored = store.save_result(result)
        except FilingError as exc:
            return _provider_failure(request, started_at, self._now(), exc.code.value, exc.message)
        if stored is None:
            raise RuntimeError("filing refresh produced no stored result")
        warnings = list(stored.warnings)
        if cache_reused:
            warnings.append("Fresh cached filings reused according to configured TTL.")
        return _handoff(
            request=request,
            status=(
                OrchestratorHandoffStatus.PARTIAL
                if partial_errors
                else OrchestratorHandoffStatus.SUCCEEDED
            ),
            started_at=started_at,
            completed_at=_aware_now(self._now()),
            input_summary={
                "cik": cik,
                "forms": ",".join(forms),
                "limit": str(limit),
            },
            output={"filings": stored.to_dict()},
            warnings=tuple(dict.fromkeys(warnings)),
            limitations=partial_errors,
            confidence=HandoffConfidence.HIGH,
        )

    async def _financial_report(self, request: DelegationRequest) -> AgentHandoff:
        started_at = _aware_now(self._now())
        result_holder: dict[str, object] = {}

        async def load_evidence(
            _context: ToolContext,
            _arguments: Mapping[str, object],
        ) -> ToolResult:
            retrieval_started = perf_counter_ns()
            retrieval_query = _optional_text(request.payload.get("retrieval_query"))
            analysis_kwargs = (
                {"retrieval_query": retrieval_query} if retrieval_query is not None else {}
            )
            result = self.financial_report_agent.analyze(
                FinancialReportAnalysisCompany(
                    cik=_text(request.payload, "cik"),
                    company_id=_optional_text(request.payload.get("company_id")),
                    legal_name=_optional_text(request.payload.get("legal_name")),
                ),
                **analysis_kwargs,
            )
            result_holder["analysis"] = result
            result_holder["retrieval_duration_ms"] = _elapsed_ms(retrieval_started)
            bounded_analysis = _bounded_financial_analysis(result.to_dict())
            evidence_ids = tuple(
                str(item["id"]) for item in _mapping_list(bounded_analysis.get("evidence", ()))
            )
            result_holder["prompt_evidence_ids"] = evidence_ids
            retrieval_methods = _evidence_retrieval_methods(result.to_dict())
            result_holder["retrieval_methods"] = retrieval_methods
            return ToolResult.succeeded(
                tool_call_id="load_financial_report_evidence",
                tool_name="load_financial_report_evidence",
                data={
                    "analysis": bounded_analysis,
                    "evidence_ids": list(evidence_ids),
                    "retrieval_methods": list(retrieval_methods),
                },
                source="local_financial_evidence",
                warnings=result.warnings,
            )

        try:
            registry = ToolRegistry(
                (
                    _source_tool(
                        name="load_financial_report_evidence",
                        description="Load stored SEC statements and filing evidence.",
                        permission=ToolPermission.FINANCIAL_DATA,
                        handler=load_evidence,
                    ),
                )
            )
            agent = await self._agent_runner(request).run(
                contract=self.prompt_catalog.by_role(PromptAgentRole.FINANCIAL_REPORT_ANALYST),
                user_payload={
                    "task": "Analyze financial statements and filing evidence.",
                    "company": dict(request.payload),
                    "required_tool": "load_financial_report_evidence",
                },
                registry=registry,
                context=_tool_context(
                    "load_financial_report_evidence",
                    ToolPermission.FINANCIAL_DATA,
                ),
                known_evidence_ids=lambda: tuple(
                    str(item) for item in result_holder.get("prompt_evidence_ids", ())
                ),
                require_evidence=_payload_bool(request.payload, "evidence_required"),
            )
            result = result_holder.get("analysis")
            if result is None:
                raise AgentRuntimeError(
                    code="agent_tool_result_missing",
                    message="Financial evidence tool returned no analysis.",
                )
        except Exception as exc:
            return _failed_handoff(request, started_at, self._now(), exc)
        contract = self.prompt_catalog.by_role(PromptAgentRole.FINANCIAL_REPORT_ANALYST)
        return _handoff(
            request=request,
            status=_analysis_status(result.status),
            started_at=started_at,
            completed_at=_aware_now(self._now()),
            input_summary={"cik": _text(request.payload, "cik")},
            output={
                "analysis": result.to_dict(),
                "agent_output": dict(agent.output),
                "retrieval_methods": list(result_holder.get("retrieval_methods", ())),
                "retrieval_duration_ms": int(result_holder.get("retrieval_duration_ms", 0)),
            },
            evidence_ids=tuple(snippet.id for snippet in result.evidence),
            warnings=result.warnings,
            limitations=result.limitations,
            confidence=_confidence(result.status),
            execution=_agent_execution(request, contract, agent),
        )

    async def _stock(self, request: DelegationRequest) -> AgentHandoff:
        started_at = _aware_now(self._now())
        symbol = _text(request.payload, "ticker")
        result_holder: dict[str, object] = {}

        async def load_evidence(
            _context: ToolContext,
            _arguments: Mapping[str, object],
        ) -> ToolResult:
            retrieval_started = perf_counter_ns()
            result = self.stock_price_agent.analyze(
                StockPriceAnalysisSecurity(
                    symbol=symbol,
                    security_id=_optional_text(request.payload.get("security_id")),
                    exchange_mic=_optional_text(request.payload.get("exchange_mic")),
                    exchange_name=_optional_text(request.payload.get("exchange_name")),
                    currency=_optional_text(request.payload.get("currency")),
                ),
                benchmark_symbol=_optional_text(request.payload.get("benchmark_symbol")),
            )
            result_holder["analysis"] = result
            result_holder["retrieval_duration_ms"] = _elapsed_ms(retrieval_started)
            return ToolResult.succeeded(
                tool_call_id="load_stock_market_evidence",
                tool_name="load_stock_market_evidence",
                data={"analysis": result.to_dict()},
                source="local_market_evidence",
                warnings=result.warnings,
            )

        try:
            registry = ToolRegistry(
                (
                    _source_tool(
                        name="load_stock_market_evidence",
                        description="Load stored company and benchmark prices with metrics.",
                        permission=ToolPermission.MARKET_DATA,
                        handler=load_evidence,
                    ),
                )
            )
            agent = await self._agent_runner(request).run(
                contract=self.prompt_catalog.by_role(PromptAgentRole.STOCK_ANALYST),
                user_payload={
                    "task": "Analyze company and benchmark market evidence.",
                    "security": dict(request.payload),
                    "required_tool": "load_stock_market_evidence",
                },
                registry=registry,
                context=_tool_context("load_stock_market_evidence", ToolPermission.MARKET_DATA),
                known_evidence_ids=lambda: (
                    _stock_evidence_ids(result_holder["analysis"].to_dict())
                    if result_holder.get("analysis") is not None
                    else ()
                ),
                require_evidence=_payload_bool(request.payload, "evidence_required"),
            )
            result = result_holder.get("analysis")
            if result is None:
                raise AgentRuntimeError(
                    code="agent_tool_result_missing",
                    message="Market evidence tool returned no analysis.",
                )
        except Exception as exc:
            return _failed_handoff(request, started_at, self._now(), exc)
        known_evidence = _stock_evidence_ids(result.to_dict())
        contract = self.prompt_catalog.by_role(PromptAgentRole.STOCK_ANALYST)
        return _handoff(
            request=request,
            status=_analysis_status(result.status),
            started_at=started_at,
            completed_at=_aware_now(self._now()),
            input_summary={"symbol": symbol},
            output={
                "analysis": result.to_dict(),
                "agent_output": dict(agent.output),
                "retrieval_methods": ["market_data"],
                "retrieval_duration_ms": int(result_holder.get("retrieval_duration_ms", 0)),
            },
            evidence_ids=known_evidence,
            warnings=result.warnings,
            limitations=result.limitations,
            confidence=_confidence(result.status),
            execution=_agent_execution(request, contract, agent),
        )

    async def _context(self, request: DelegationRequest) -> AgentHandoff:
        started_at = _aware_now(self._now())
        source_items = tuple(
            _context_source_from_dict(item)
            for item in _mapping_list(request.payload.get("source_items", ()))
        )
        if not source_items:
            return _skipped_handoff(
                request,
                started_at,
                "No approved context sources were available.",
            )
        symbols = tuple(str(item) for item in _list(request.payload.get("company_symbols", ())))
        result_holder: dict[str, object] = {}

        async def load_evidence(
            _context: ToolContext,
            _arguments: Mapping[str, object],
        ) -> ToolResult:
            retrieval_started = perf_counter_ns()
            result = self.context_agent.analyze(
                query=_text(request.payload, "query"),
                source_items=source_items,
                company_symbols=symbols,
            )
            result_holder["analysis"] = result
            result_holder["retrieval_duration_ms"] = _elapsed_ms(retrieval_started)
            return ToolResult.succeeded(
                tool_call_id="load_context_evidence",
                tool_name="load_context_evidence",
                data={"analysis": result.to_dict()},
                source="bounded_context_sources",
                warnings=result.warnings,
            )

        try:
            known_evidence = tuple(item.id for item in source_items)
            registry = ToolRegistry(
                (
                    _source_tool(
                        name="load_context_evidence",
                        description="Load approved source-linked company and macro context.",
                        permission=ToolPermission.CONTEXT_DATA,
                        handler=load_evidence,
                    ),
                )
            )
            agent = await self._agent_runner(request).run(
                contract=self.prompt_catalog.by_role(PromptAgentRole.NEWS_MACRO_ANALYST),
                user_payload={
                    "task": "Analyze only approved source-linked context.",
                    "query": _text(request.payload, "query"),
                    "company_symbols": list(symbols),
                    "required_tool": "load_context_evidence",
                },
                registry=registry,
                context=_tool_context("load_context_evidence", ToolPermission.CONTEXT_DATA),
                known_evidence_ids=known_evidence,
                require_evidence=_payload_bool(request.payload, "evidence_required"),
            )
            result = result_holder.get("analysis")
            if result is None:
                raise AgentRuntimeError(
                    code="agent_tool_result_missing",
                    message="Context evidence tool returned no analysis.",
                )
        except Exception as exc:
            return _failed_handoff(request, started_at, self._now(), exc)
        evidence_ids = known_evidence
        contract = self.prompt_catalog.by_role(PromptAgentRole.NEWS_MACRO_ANALYST)
        return _handoff(
            request=request,
            status=_analysis_status(result.status),
            started_at=started_at,
            completed_at=_aware_now(self._now()),
            input_summary={"source_item_count": str(len(source_items))},
            output={
                "analysis": result.to_dict(),
                "agent_output": dict(agent.output),
                "retrieval_methods": ["approved_context"],
                "retrieval_duration_ms": int(result_holder.get("retrieval_duration_ms", 0)),
            },
            evidence_ids=tuple(dict.fromkeys(evidence_ids)),
            warnings=result.warnings,
            limitations=result.limitations,
            confidence=_confidence(result.status),
            execution=_agent_execution(request, contract, agent),
        )

    async def _synthesis(self, request: DelegationRequest) -> AgentHandoff:
        started_at = _aware_now(self._now())
        run = self._load_run(request.run_id)
        specialist_handoffs = tuple(
            handoff
            for handoff in run.handoffs
            if handoff.kind
            in {
                OrchestratorStepKind.FINANCIAL_REPORT_ANALYSIS,
                OrchestratorStepKind.STOCK_PRICE_ANALYSIS,
                OrchestratorStepKind.CONTEXT_ANALYSIS,
            }
        )
        result_holder: dict[str, object] = {}

        async def load_handoffs(
            _context: ToolContext,
            _arguments: Mapping[str, object],
        ) -> ToolResult:
            payload = [_synthesis_handoff_payload(handoff) for handoff in specialist_handoffs]
            result_holder["handoffs"] = payload
            return ToolResult.succeeded(
                tool_call_id="load_specialist_handoffs",
                tool_name="load_specialist_handoffs",
                data={"handoffs": payload},
                source="persisted_specialist_handoffs",
            )

        known_evidence = tuple(
            dict.fromkeys(
                evidence_id
                for handoff in specialist_handoffs
                for evidence_id in handoff.evidence_ids
            )
        )
        try:
            registry = ToolRegistry(
                (
                    _source_tool(
                        name="load_specialist_handoffs",
                        description="Load validated persisted specialist handoffs.",
                        permission=ToolPermission.HANDOFF_READ,
                        handler=load_handoffs,
                    ),
                )
            )
            agent = await self._agent_runner(request).run(
                contract=self.prompt_catalog.by_role(PromptAgentRole.SYNTHESIS_AGENT),
                user_payload={
                    "task": "Synthesize validated specialist handoffs.",
                    "query": run.query,
                    "handoff_ids": [handoff.id for handoff in specialist_handoffs],
                    "required_tool": "load_specialist_handoffs",
                },
                registry=registry,
                context=_tool_context("load_specialist_handoffs", ToolPermission.HANDOFF_READ),
                known_evidence_ids=known_evidence,
                require_evidence=run.evidence_required,
            )
            report = self.synthesis_agent.synthesize_agent_output(
                query=run.query,
                handoffs=specialist_handoffs,
                agent_output=agent.output,
                selected_company=run.selected_company,
                selected_security=run.selected_security,
                created_at=started_at,
            )
        except Exception as exc:
            return _failed_handoff(request, started_at, self._now(), exc)
        unknown_limitations = tuple(
            limitation for point in report.unknowns for limitation in point.limitations
        )
        contract = self.prompt_catalog.by_role(PromptAgentRole.SYNTHESIS_AGENT)
        return _handoff(
            request=request,
            status=(
                OrchestratorHandoffStatus.SUCCEEDED
                if report.status == SynthesisReportStatus.COMPLETE
                else OrchestratorHandoffStatus.PARTIAL
            ),
            started_at=started_at,
            completed_at=_aware_now(self._now()),
            input_summary={"specialist_handoff_count": str(len(specialist_handoffs))},
            output={
                "summary": report.summary,
                "report": report.to_dict(),
                "agent_output": dict(agent.output),
                "specialist_statuses": {
                    handoff.kind.value: handoff.status.value for handoff in specialist_handoffs
                },
            },
            evidence_ids=report.evidence_ids,
            warnings=tuple(
                dict.fromkeys(
                    (
                        *report.warnings,
                        *(
                            warning
                            for handoff in specialist_handoffs
                            for warning in handoff.warnings
                        ),
                    )
                )
            ),
            limitations=tuple(dict.fromkeys((*report.limitations, *unknown_limitations))),
            confidence=(
                HandoffConfidence.MEDIUM if specialist_handoffs else HandoffConfidence.UNKNOWN
            ),
            execution=_agent_execution(request, contract, agent),
        )

    def _load_run(self, run_id: str) -> OrchestratedResearchRun:
        getter = getattr(self.run_store, "get", None)
        run = getter(run_id) if callable(getter) else None
        if not isinstance(run, OrchestratedResearchRun):
            raise ValueError("orchestrator run is unavailable")
        return run


def _handoff(
    *,
    request: DelegationRequest,
    status: OrchestratorHandoffStatus,
    started_at: datetime,
    completed_at: datetime,
    input_summary: Mapping[str, str] | None = None,
    output: Mapping[str, object] | None = None,
    evidence_ids: tuple[str, ...] = (),
    warnings: tuple[str, ...] = (),
    limitations: tuple[str, ...] = (),
    confidence: HandoffConfidence = HandoffConfidence.UNKNOWN,
    error_code: str | None = None,
    error_message: str | None = None,
    execution: AgentExecutionMetadata | None = None,
) -> AgentHandoff:
    return AgentHandoff(
        id=f"handoff_{request.expected_kind.value}_{uuid4().hex}",
        step_id=request.step_id,
        kind=request.expected_kind,
        status=status,
        started_at=started_at,
        completed_at=completed_at,
        input_summary=dict(input_summary or {}),
        output=dict(output or {}),
        evidence_ids=evidence_ids,
        warnings=warnings,
        limitations=limitations,
        confidence=confidence,
        error_code=error_code,
        error_message=error_message,
        execution=execution,
    )


def _source_tool(
    *,
    name: str,
    description: str,
    permission: ToolPermission,
    handler,
) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=description,
        input_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        permissions=(permission,),
        timeout_seconds=30.0,
        handler=handler,
    )


def _tool_context(name: str, permission: ToolPermission) -> ToolContext:
    return ToolContext(
        allowed_tools=(name,),
        allowed_permissions=(permission,),
    )


def _agent_execution(
    request: DelegationRequest,
    contract: PromptContract,
    result: StructuredAgentResult,
) -> AgentExecutionMetadata:
    tool_status = ",".join(tool.status.value for tool in result.tool_results)
    return AgentExecutionMetadata(
        mode=AgentExecutionMode.A2A,
        agent_role=request.role.value,
        correlation_id=request.correlation_id,
        prompt_id=contract.id,
        prompt_version=contract.version.value,
        provider=result.provider,
        model=result.model,
        tool_status=tool_status,
        reasoning_summary=str(result.output.get("reasoning_summary", "")),
        skill_references=tuple(f"{skill.id}@{skill.version.value}" for skill in result.skills),
    )


def _synthesis_handoff_payload(handoff: AgentHandoff) -> dict[str, object]:
    agent_output = handoff.output.get("agent_output")
    return {
        "id": handoff.id,
        "kind": handoff.kind.value,
        "status": handoff.status.value,
        "agent_output": dict(agent_output) if isinstance(agent_output, Mapping) else {},
        "evidence_ids": list(handoff.evidence_ids),
        "warnings": list(handoff.warnings),
        "limitations": list(handoff.limitations),
        "confidence": handoff.confidence.value,
        "error_code": handoff.error_code,
    }


def _stock_evidence_ids(payload: Mapping[str, object]) -> tuple[str, ...]:
    security = _mapping(payload.get("security", {}))
    symbol = str(security.get("symbol") or "unknown").upper()
    evidence_ids: list[str] = []
    if payload.get("primary_source") is not None:
        evidence_ids.append(f"stock:{symbol}:primary")
    benchmark_value = payload.get("benchmark_security")
    benchmark = _mapping(benchmark_value) if isinstance(benchmark_value, Mapping) else {}
    benchmark_symbol = str(benchmark.get("symbol") or "").upper()
    if payload.get("benchmark_source") is not None and benchmark_symbol:
        evidence_ids.append(f"stock:{benchmark_symbol}:benchmark")
    return tuple(evidence_ids)


def _failed_handoff(
    request: DelegationRequest,
    started_at: datetime,
    now: datetime,
    error: Exception,
) -> AgentHandoff:
    if isinstance(error, AgentRuntimeError):
        error_code = error.code
        error_message = error.message
    elif isinstance(error, ProviderError):
        error_code = error.code.value
        error_message = "Configured agent provider failed."
    else:
        error_code = f"{request.role.value.replace('-', '_')}_failed"
        error_message = "Specialist analysis failed safely."
    return _handoff(
        request=request,
        status=OrchestratorHandoffStatus.FAILED,
        started_at=started_at,
        completed_at=_aware_now(now),
        error_code=error_code,
        error_message=error_message,
        limitations=(error_message,),
    )


def _provider_failure(
    request: DelegationRequest,
    started_at: datetime,
    now: datetime,
    error_code: str,
    error_message: str,
) -> AgentHandoff:
    return _handoff(
        request=request,
        status=OrchestratorHandoffStatus.FAILED,
        started_at=started_at,
        completed_at=_aware_now(now),
        error_code=error_code,
        error_message=error_message,
        limitations=(error_message,),
    )


def _skipped_handoff(
    request: DelegationRequest,
    started_at: datetime,
    reason: str,
) -> AgentHandoff:
    return _handoff(
        request=request,
        status=OrchestratorHandoffStatus.SKIPPED,
        started_at=started_at,
        completed_at=started_at,
        limitations=(reason,),
    )


async def _ingest_filings(
    provider: FilingProvider,
    company: FilingCompany,
    *,
    forms: tuple[str, ...],
    limit: int,
    form_limits: Mapping[str, int],
) -> tuple[FilingIngestionResult, tuple[str, ...]]:
    if not form_limits:
        return await provider.ingest_latest(company, forms=forms, limit=limit), ()
    results: list[FilingIngestionResult] = []
    errors: list[str] = []
    last_error: FilingError | None = None
    for form, form_limit in form_limits.items():
        try:
            results.append(await provider.ingest_latest(company, forms=(form,), limit=form_limit))
        except FilingError as exc:
            last_error = exc
            errors.append(f"{form} ingestion failed: {exc.message}")
    if not results:
        assert last_error is not None
        raise last_error
    return (
        FilingIngestionResult(
            company=company,
            filings=tuple(filing for result in results for filing in result.filings),
            chunks=tuple(chunk for result in results for chunk in result.chunks),
            source=results[0].source,
            warnings=tuple(
                dict.fromkeys(warning for result in results for warning in result.warnings)
            ),
        ),
        tuple(errors),
    )


def _analysis_status(value: object) -> OrchestratorHandoffStatus:
    if value in {
        FinancialReportAnalysisStatus.COMPLETE,
        StockPriceAnalysisStatus.COMPLETE,
        ContextAnalysisStatus.COMPLETE,
    }:
        return OrchestratorHandoffStatus.SUCCEEDED
    if value in {
        FinancialReportAnalysisStatus.PARTIAL,
        StockPriceAnalysisStatus.PARTIAL,
        ContextAnalysisStatus.PARTIAL,
    }:
        return OrchestratorHandoffStatus.PARTIAL
    return OrchestratorHandoffStatus.SKIPPED


def _confidence(value: object) -> HandoffConfidence:
    if value in {
        FinancialReportAnalysisStatus.COMPLETE,
        StockPriceAnalysisStatus.COMPLETE,
        ContextAnalysisStatus.COMPLETE,
    }:
        return HandoffConfidence.HIGH
    if value in {
        FinancialReportAnalysisStatus.PARTIAL,
        StockPriceAnalysisStatus.PARTIAL,
        ContextAnalysisStatus.PARTIAL,
    }:
        return HandoffConfidence.MEDIUM
    return HandoffConfidence.UNKNOWN


def _context_source_from_dict(payload: Mapping[str, object]) -> ContextSourceItem:
    return ContextSourceItem(
        id=_text(payload, "id"),
        title=_text(payload, "title"),
        summary=_text(payload, "summary"),
        source_url=_text(payload, "source_url"),
        source_name=_text(payload, "source_name"),
        source_type=ContextSourceType(_text(payload, "source_type")),
        reliability=SourceReliability(_text(payload, "reliability")),
        scope=ContextScope(_text(payload, "scope")),
        retrieved_at=_datetime(payload, "retrieved_at"),
        published_at=_optional_datetime(payload.get("published_at")),
        company_symbols=tuple(str(item) for item in _list(payload.get("company_symbols", ()))),
        sector=_optional_text(payload.get("sector")),
        region=_optional_text(payload.get("region")),
        topics=tuple(str(item) for item in _list(payload.get("topics", ()))),
        metadata={
            str(key): str(value) for key, value in _mapping(payload.get("metadata", {})).items()
        },
    )


def _text(payload: Mapping[str, object], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required")
    return value.strip()


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("optional text value must be a string")
    return value.strip() or None


def _payload_bool(payload: Mapping[str, object], name: str) -> bool:
    value = payload.get(name, False)
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _is_fresh_cached(
    value: object | None,
    *,
    store: object,
    now: datetime,
) -> bool:
    if value is None:
        return False
    source = getattr(value, "source", None)
    retrieved_at = getattr(source, "retrieved_at", None)
    stale_after = getattr(store, "stale_after", None)
    return (
        isinstance(retrieved_at, datetime)
        and stale_after is not None
        and retrieved_at + stale_after > now
    )


def _elapsed_ms(started_ns: int) -> int:
    return max(0, (perf_counter_ns() - started_ns) // 1_000_000)


def _evidence_retrieval_methods(analysis: Mapping[str, object]) -> tuple[str, ...]:
    methods: list[str] = []
    for item in _mapping_list(analysis.get("evidence", ())):
        metadata = item.get("metadata", {})
        if not isinstance(metadata, Mapping):
            continue
        value = metadata.get("retrieval_method")
        if isinstance(value, str) and value.strip():
            methods.append(value.strip())
    if not methods and analysis.get("evidence"):
        methods.append("statement_evidence")
    return tuple(dict.fromkeys(methods))


def _bounded_financial_analysis(analysis: Mapping[str, object]) -> dict[str, object]:
    payload = dict(analysis)
    evidence = _mapping_list(analysis.get("evidence", ()))
    findings = _mapping_list(analysis.get("findings", ()))
    ordered_ids = tuple(
        dict.fromkeys(
            (
                *(
                    str(evidence_id)
                    for finding in findings
                    for evidence_id in _list(finding.get("evidence_ids", ()))
                ),
                *(str(item.get("id")) for item in evidence if item.get("id") is not None),
            )
        )
    )
    evidence_by_id = {str(item.get("id")): item for item in evidence}
    selected: list[dict[str, object]] = []
    total_chars = 0
    for evidence_id in ordered_ids:
        item = evidence_by_id.get(evidence_id)
        if item is None or len(selected) >= 5 or total_chars >= 4_000:
            continue
        text = str(item.get("text", "")).strip()
        remaining = min(900, 4_000 - total_chars)
        if len(text) <= remaining:
            bounded = text
        elif remaining <= 3:
            bounded = text[:remaining]
        else:
            bounded = text[: remaining - 3].rstrip() + "..."
        selected.append({**dict(item), "text": bounded})
        total_chars += len(bounded)
    citation_ids = {
        str(item.get("citation_id")) for item in selected if item.get("citation_id") is not None
    }
    payload["evidence"] = selected
    payload["citations"] = [
        dict(item)
        for item in _mapping_list(analysis.get("citations", ()))
        if str(item.get("id")) in citation_ids
    ]
    return payload


def _list(value: object) -> list[object]:
    if not isinstance(value, list | tuple):
        raise ValueError("value must be a list")
    return list(value)


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("value must be an object")
    return value


def _mapping_list(value: object) -> tuple[Mapping[str, object], ...]:
    return tuple(_mapping(item) for item in _list(value))


def _positive_int(payload: Mapping[str, object], name: str) -> int:
    return _positive_int_value(payload.get(name), name)


def _positive_int_mapping(value: object) -> dict[str, int]:
    mapping = _mapping(value)
    result: dict[str, int] = {}
    for key, item in mapping.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("mapping keys must be non-empty strings")
        result[key.strip()] = _positive_int_value(item, "mapping value")
    return result


def _positive_int_value(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, float) and value.is_integer():
        parsed = int(value)
    else:
        raise ValueError(f"{name} must be a positive integer")
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def _text_tuple(payload: Mapping[str, object], name: str) -> tuple[str, ...]:
    values = tuple(str(item).strip() for item in _list(payload.get(name)))
    if not values or any(not item for item in values):
        raise ValueError(f"{name} must contain non-empty strings")
    return values


def _required_dependency(value, name: str):
    if value is None:
        raise ValueError(f"{name} is unavailable")
    return value


def _datetime(payload: Mapping[str, object], name: str) -> datetime:
    return _optional_datetime(_text(payload, name))  # type: ignore[return-value]


def _optional_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(str(value))
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _aware_now(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
