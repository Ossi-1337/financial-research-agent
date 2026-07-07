from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter_ns
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from financial_research_agent.background import (
    BackgroundResearchJob,
    BackgroundResearchRunner,
    BackgroundResearchStatus,
)
from financial_research_agent.context_analysis import (
    ContextScope,
    ContextSourceItem,
    ContextSourceType,
    NewsMacroSectorAgent,
    SourceReliability,
)
from financial_research_agent.entities import (
    CompanySearchError,
    CompanySearchErrorCode,
    CompanySearchProvider,
    create_default_company_search_provider,
)
from financial_research_agent.filings import (
    FilingCompany,
    FilingError,
    FilingErrorCode,
    FilingProvider,
    FilingStore,
    create_default_filing_provider,
)
from financial_research_agent.interop import (
    InteropAccessDecision,
    InteropAccessPolicy,
    MCPReadOnlyDispatcher,
    create_agent_card,
    create_sanitized_status_payload,
)
from financial_research_agent.llm import (
    ChatMessage,
    ChatRequest,
    MessageRole,
    ModelMetadata,
    ProviderCapability,
    ProviderError,
    ProviderErrorCode,
    StreamEventType,
)
from financial_research_agent.llm.local_openai import OpenAICompatibleLocalProvider
from financial_research_agent.llm.offline import OfflineTestProvider
from financial_research_agent.llm.openai import OpenAIProvider
from financial_research_agent.llm.registry import ProviderRegistry, create_default_provider_registry
from financial_research_agent.market_data import (
    MarketDataError,
    MarketDataErrorCode,
    MarketDataProvider,
    MarketDataStore,
    MarketSecurity,
    create_default_market_data_provider,
)
from financial_research_agent.observability import (
    RedactionPolicy,
    build_debug_bundle,
    build_replay_plan,
    build_trace_from_orchestrator_run,
)
from financial_research_agent.orchestration import (
    OrchestratedResearchRun,
    OrchestratorResearchInput,
    OrchestratorRunStatus,
    OrchestratorRunStore,
    ResearchOrchestrator,
    default_orchestrator_plan,
)
from financial_research_agent.performance import (
    CachingEmbeddingProvider,
    LocalEmbeddingCache,
    ProviderCallKind,
    call_metrics_from_response,
    default_local_model_profiles,
    measured_chat,
    prompt_budgets_for_limits,
)
from financial_research_agent.report_analysis import (
    FinancialReportAnalysisAgent,
    FinancialReportAnalysisCompany,
)
from financial_research_agent.reports import (
    CitedResearchRun,
    CitedResearchRunStatus,
    CitedResearchRunStore,
    build_rag_messages,
    citation_artifacts_from_retrieval,
    ensure_citation_marker,
    missing_evidence_limitation,
)
from financial_research_agent.retrieval import (
    LocalVectorIndex,
    RetrievalError,
    RetrievalErrorCode,
    RetrievalQuery,
    index_filing_result,
    search_index,
)
from financial_research_agent.runtime_settings import RuntimeSettingsOverrides, RuntimeSettingsStore
from financial_research_agent.settings import ProviderTask, Settings
from financial_research_agent.statements import (
    FinancialStatementCompany,
    FinancialStatementError,
    FinancialStatementErrorCode,
    FinancialStatementProvider,
    FinancialStatementStore,
    create_default_financial_statement_provider,
)
from financial_research_agent.stock_analysis import (
    StockPriceAnalysisAgent,
    StockPriceAnalysisSecurity,
)
from financial_research_agent.storage import LocalStorageManager
from financial_research_agent.web.sessions import ChatMention, ChatSessionStore

SYSTEM_PROMPT = (
    "You are a local financial research chat assistant. Sidebar data fetches may exist, but "
    "this chat endpoint does not automatically receive live financial data, RAG evidence, "
    "or agent orchestration. Do not invent identifiers, prices, financial facts, source URLs, "
    "or citations. If the user asks for current company or market facts, explain that they "
    "must fetch and inspect source data first. Do not provide buy, sell, or hold "
    "recommendations."
)


class MentionRequest(BaseModel):
    id: str = Field(min_length=1, max_length=200)
    label: str = Field(min_length=1, max_length=200)
    company_id: str = Field(min_length=1, max_length=200)
    legal_name: str = Field(min_length=1, max_length=300)
    ticker: str | None = Field(default=None, max_length=32)
    cik: str | None = Field(default=None, max_length=10, pattern="^[0-9]+$")
    source_provider: str | None = Field(default=None, max_length=80)


class MessageRequest(BaseModel):
    content: str = Field(min_length=1, max_length=20_000)
    mentions: tuple[MentionRequest, ...] = ()


class MarketDataHistoryRequest(BaseModel):
    symbol: str = Field(min_length=1, max_length=32)
    security_id: str | None = None
    exchange_mic: str | None = None
    exchange_name: str | None = None
    currency: str | None = None
    outputsize: str = Field(default="compact", pattern="^(compact|full)$")
    refresh: bool = False


class FinancialStatementRequest(BaseModel):
    cik: str = Field(min_length=1, max_length=10, pattern="^[0-9]+$")
    company_id: str | None = None
    legal_name: str | None = None
    fiscal_years: int = Field(default=3, ge=1, le=10)
    refresh: bool = False


class FilingIngestionRequest(BaseModel):
    cik: str = Field(min_length=1, max_length=10, pattern="^[0-9]+$")
    company_id: str | None = None
    legal_name: str | None = None
    forms: tuple[str, ...] = ("10-K", "10-Q")
    limit: int = Field(default=1, ge=1, le=10)
    refresh: bool = False


class RetrievalIndexFilingRequest(BaseModel):
    cik: str = Field(min_length=1, max_length=10, pattern="^[0-9]+$")
    rebuild: bool = True


class RetrievalSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2_000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    min_score: float | None = Field(default=None, ge=-1.0, le=1.0)
    filters: dict[str, str] = Field(default_factory=dict)


class CitedAnswerRequest(BaseModel):
    content: str = Field(min_length=1, max_length=20_000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    min_score: float | None = Field(default=None, ge=-1.0, le=1.0)
    filters: dict[str, str] = Field(default_factory=dict)


class FinancialReportAnalysisRequest(BaseModel):
    cik: str = Field(min_length=1, max_length=10, pattern="^[0-9]+$")
    company_id: str | None = None
    legal_name: str | None = None


class StockPriceAnalysisRequest(BaseModel):
    symbol: str = Field(min_length=1, max_length=32)
    security_id: str | None = None
    exchange_mic: str | None = None
    exchange_name: str | None = None
    currency: str | None = None
    benchmark_symbol: str | None = Field(default=None, min_length=1, max_length=32)


class ContextSourceItemRequest(BaseModel):
    id: str = Field(min_length=1, max_length=200)
    title: str = Field(min_length=1, max_length=500)
    summary: str = Field(min_length=1, max_length=2_000)
    source_url: str = Field(min_length=1, max_length=2_000)
    source_name: str = Field(min_length=1, max_length=200)
    source_type: ContextSourceType
    reliability: SourceReliability
    scope: ContextScope
    retrieved_at: datetime
    published_at: datetime | None = None
    company_symbols: tuple[str, ...] = ()
    sector: str | None = None
    region: str | None = None
    topics: tuple[str, ...] = ()
    metadata: dict[str, str] = Field(default_factory=dict)


class ContextAnalysisRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2_000)
    company_symbols: tuple[str, ...] = ()
    sector: str | None = None
    region: str | None = None
    source_items: tuple[ContextSourceItemRequest, ...] = ()


class OrchestratorResearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2_000)
    refresh: bool = True
    company_search_limit: int = Field(default=3, ge=1, le=10)
    fiscal_years: int = Field(default=3, ge=1, le=10)
    filing_forms: tuple[str, ...] = ("10-K", "10-Q")
    filing_limit: int = Field(default=1, ge=1, le=5)
    market_outputsize: str = Field(default="compact", pattern="^(compact|full)$")
    benchmark_symbol: str | None = Field(default=None, min_length=1, max_length=32)
    context_source_items: tuple[ContextSourceItemRequest, ...] = ()


class RuntimeSettingsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    llm_provider: str | None = Field(default=None, min_length=1, max_length=80)
    llm_model: str | None = Field(default=None, min_length=1, max_length=200)
    llm_base_url: str | None = Field(default=None, max_length=500)
    llm_local_runtime: str | None = Field(default=None, min_length=1, max_length=80)
    llm_timeout_seconds: float | None = Field(default=None, gt=0, le=300)
    openai_api_key: str | None = Field(default=None, max_length=500)
    alpha_vantage_api_key: str | None = Field(default=None, max_length=500)
    interop_api_key: str | None = Field(default=None, max_length=500)
    api_key: str | None = Field(default=None, max_length=500)
    embedding_provider: str | None = Field(default=None, min_length=1, max_length=80)
    embedding_model: str | None = Field(default=None, max_length=200)
    chat_provider: str | None = Field(default=None, max_length=80)
    chat_model: str | None = Field(default=None, max_length=200)
    streaming_provider: str | None = Field(default=None, max_length=80)
    streaming_model: str | None = Field(default=None, max_length=200)
    tool_calling_provider: str | None = Field(default=None, max_length=80)
    tool_calling_model: str | None = Field(default=None, max_length=200)
    structured_output_provider: str | None = Field(default=None, max_length=80)
    structured_output_model: str | None = Field(default=None, max_length=200)
    chat_history_recent_turns: int | None = Field(default=None, ge=1, le=100)
    chat_history_summary_max_chars: int | None = Field(default=None, ge=100, le=20_000)
    company_lookup_provider: str | None = Field(default=None, min_length=1, max_length=80)
    company_lookup_cache_ttl_days: int | None = Field(default=None, ge=1, le=365)
    market_data_provider: str | None = Field(default=None, min_length=1, max_length=80)
    market_data_cache_ttl_days: int | None = Field(default=None, ge=1, le=365)
    financial_statement_provider: str | None = Field(default=None, min_length=1, max_length=80)
    financial_statement_cache_ttl_days: int | None = Field(default=None, ge=1, le=365)
    filing_provider: str | None = Field(default=None, min_length=1, max_length=80)
    filing_cache_ttl_days: int | None = Field(default=None, ge=1, le=365)
    filing_max_document_bytes: int | None = Field(default=None, ge=1_000, le=100_000_000)
    retrieval_provider: str | None = Field(default=None, min_length=1, max_length=80)
    retrieval_top_k: int | None = Field(default=None, ge=1, le=50)
    retrieval_min_score: float | None = Field(default=None, ge=-1.0, le=1.0)
    background_max_concurrent_research_runs: int | None = Field(default=None, ge=1, le=8)

    def overrides(self) -> RuntimeSettingsOverrides:
        return RuntimeSettingsOverrides.from_mapping(
            self.model_dump(exclude_none=True),
        )


def create_app(
    *,
    settings: Settings | None = None,
    registry: ProviderRegistry | None = None,
    session_store: ChatSessionStore | None = None,
    company_search_provider: CompanySearchProvider | None = None,
    market_data_provider: MarketDataProvider | None = None,
    market_data_store: MarketDataStore | None = None,
    financial_statement_provider: FinancialStatementProvider | None = None,
    financial_statement_store: FinancialStatementStore | None = None,
    filing_provider: FilingProvider | None = None,
    filing_store: FilingStore | None = None,
    storage_manager: LocalStorageManager | None = None,
    retrieval_index: LocalVectorIndex | None = None,
    report_run_store: CitedResearchRunStore | None = None,
    orchestrator_run_store: OrchestratorRunStore | None = None,
    background_runner: BackgroundResearchRunner | None = None,
    runtime_settings_store: RuntimeSettingsStore | None = None,
    embedding_cache: LocalEmbeddingCache | None = None,
) -> FastAPI:
    app_settings = settings or Settings.from_env()
    provider_registry = registry or create_default_provider_registry(app_settings.provider)
    sessions = session_store or ChatSessionStore.from_settings(app_settings)
    company_search = company_search_provider or create_default_company_search_provider(app_settings)
    market_provider = market_data_provider or create_default_market_data_provider(app_settings)
    market_store = market_data_store or MarketDataStore.from_settings(app_settings)
    statement_provider = (
        financial_statement_provider or create_default_financial_statement_provider(app_settings)
    )
    statement_store = financial_statement_store or FinancialStatementStore.from_settings(
        app_settings
    )
    filing_source = filing_provider or create_default_filing_provider(app_settings)
    filings = filing_store or FilingStore.from_settings(app_settings)
    storage = storage_manager or LocalStorageManager.from_settings(app_settings)
    retrieval = retrieval_index or LocalVectorIndex.from_settings(app_settings)
    report_runs = report_run_store or CitedResearchRunStore.from_settings(app_settings)
    orchestrator_runs = orchestrator_run_store or OrchestratorRunStore.from_settings(app_settings)
    runtime_settings = runtime_settings_store or RuntimeSettingsStore.from_settings(app_settings)
    embeddings_cache = embedding_cache or LocalEmbeddingCache.from_settings(app_settings)
    background_research = background_runner or BackgroundResearchRunner(
        max_concurrent_runs=app_settings.background.max_concurrent_research_runs,
    )
    financial_report_agent = FinancialReportAnalysisAgent(
        statement_store=statement_store,
        filing_store=filings,
        statement_provider=app_settings.data_sources.financial_statement_provider,
        filing_provider=app_settings.data_sources.filing_provider,
    )
    stock_price_agent = StockPriceAnalysisAgent(
        market_data_store=market_store,
        market_data_provider=app_settings.data_sources.market_data_provider,
    )
    context_agent = NewsMacroSectorAgent()
    orchestrator = ResearchOrchestrator(
        company_search_provider=company_search,
        market_data_provider=market_provider,
        market_data_store=market_store,
        financial_statement_provider=statement_provider,
        financial_statement_store=statement_store,
        filing_provider=filing_source,
        filing_store=filings,
        financial_report_agent=financial_report_agent,
        stock_price_agent=stock_price_agent,
        context_agent=context_agent,
        run_store=orchestrator_runs,
    )
    static_dir = Path(__file__).with_name("static")

    app = FastAPI(title="Financial Research Agent", version="0.1.0")
    app.state.settings = app_settings
    app.state.provider_registry = provider_registry
    app.state.sessions = sessions
    app.state.company_search = company_search
    app.state.market_data_provider = market_provider
    app.state.market_data_store = market_store
    app.state.financial_statement_provider = statement_provider
    app.state.financial_statement_store = statement_store
    app.state.filing_provider = filing_source
    app.state.filing_store = filings
    app.state.storage_manager = storage
    app.state.runtime_settings_store = runtime_settings
    app.state.embedding_cache = embeddings_cache
    app.state.retrieval_index = retrieval
    app.state.report_run_store = report_runs
    app.state.orchestrator_run_store = orchestrator_runs
    app.state.background_research = background_research
    app.state.financial_report_agent = financial_report_agent
    app.state.stock_price_agent = stock_price_agent
    app.state.context_agent = context_agent
    app.state.orchestrator = orchestrator

    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    def current_settings() -> Settings:
        return runtime_settings.settings(app_settings)

    def current_registry() -> ProviderRegistry:
        if registry is not None:
            return provider_registry
        return create_default_provider_registry(current_settings().provider)

    def sanitized_interop_status() -> dict[str, object]:
        settings_for_request = current_settings()
        registry_for_request = current_registry()
        selection = settings_for_request.provider.selection_for_task(ProviderTask.CHAT)
        return create_sanitized_status_payload(
            environment=settings_for_request.environment,
            chat_provider=selection.provider,
            chat_model=selection.model,
            chat_registered=registry_for_request.has_chat_provider(selection.provider),
            storage_provider=settings_for_request.storage.provider,
            retrieval_provider=settings_for_request.retrieval.provider,
            interop_policy=_interop_policy(settings_for_request),
        )

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/.well-known/agent.json")
    @app.get("/.well-known/agent-card.json")
    def a2a_agent_card(
        request: Request,
        authorization: str | None = Header(default=None),
        x_fra_interop_key: str | None = Header(default=None),
    ) -> dict[str, Any]:
        _require_interop_access(request, app_settings, authorization, x_fra_interop_key)
        return create_agent_card(
            base_url=str(request.base_url),
            version=app.version,
            api_key_required=app_settings.interoperability.api_key is not None,
        ).to_dict()

    @app.post("/api/interop/mcp")
    async def mcp_read_only_endpoint(
        request: Request,
        authorization: str | None = Header(default=None),
        x_fra_interop_key: str | None = Header(default=None),
    ) -> dict[str, Any]:
        _require_interop_access(request, app_settings, authorization, x_fra_interop_key)
        try:
            payload = await request.json()
        except json.JSONDecodeError:
            return {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "Parse error"},
            }
        if not isinstance(payload, dict):
            return {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32600, "message": "Invalid JSON-RPC request."},
            }
        dispatcher = MCPReadOnlyDispatcher(status_payload=sanitized_interop_status())
        return dispatcher.handle(payload)

    @app.get("/api/status")
    async def status() -> dict[str, Any]:
        settings_for_request = current_settings()
        registry_for_request = current_registry()
        selection = settings_for_request.provider.selection_for_task(ProviderTask.CHAT)
        embedding_selection = settings_for_request.provider.selection_for_task(
            ProviderTask.EMBEDDINGS
        )
        background_stats = await background_research.stats()
        return {
            "app": "financial-research-agent",
            "environment": settings_for_request.environment,
            "status": "ok",
            "chat": {
                "provider": selection.provider,
                "model": selection.model,
                "base_url": selection.base_url,
                "registered": registry_for_request.has_chat_provider(selection.provider),
            },
            "history": {
                "recent_turns": sessions.recent_turns,
                "summary_max_chars": sessions.summary_max_chars,
                "session_count": sessions.count(),
                "persistent": sessions.storage_path is not None,
            },
            "company_search": {
                "provider": settings_for_request.data_sources.company_lookup_provider,
                "cache_ttl_days": (settings_for_request.data_sources.company_lookup_cache_ttl_days),
            },
            "market_data": {
                "provider": settings_for_request.data_sources.market_data_provider,
                "cache_ttl_days": settings_for_request.data_sources.market_data_cache_ttl_days,
                "alpha_vantage_api_key_configured": (
                    settings_for_request.data_sources.alpha_vantage_api_key is not None
                ),
                "stored_series_count": market_store.count(),
            },
            "financial_statements": {
                "provider": settings_for_request.data_sources.financial_statement_provider,
                "cache_ttl_days": (
                    settings_for_request.data_sources.financial_statement_cache_ttl_days
                ),
                "stored_result_count": statement_store.count(),
            },
            "filings": {
                "provider": settings_for_request.data_sources.filing_provider,
                "cache_ttl_days": settings_for_request.data_sources.filing_cache_ttl_days,
                "max_document_bytes": settings_for_request.data_sources.filing_max_document_bytes,
                "stored_result_count": filings.count(),
            },
            "retrieval": {
                "provider": settings_for_request.retrieval.provider,
                "top_k": settings_for_request.retrieval.top_k,
                "min_score": settings_for_request.retrieval.min_score,
                "index": retrieval.metadata().to_dict(),
                "embedding_provider": embedding_selection.provider,
                "embedding_model": embedding_selection.model,
                "embedding_provider_registered": registry_for_request.has_embedding_provider(
                    embedding_selection.provider
                ),
            },
            "report_runs": {
                "stored_run_count": report_runs.count(),
                "persistent": report_runs.storage_path is not None,
            },
            "financial_report_analysis": {
                "source": "stored_financial_statements_and_filings",
                "recommendations": "disabled",
            },
            "stock_price_analysis": {
                "source": "stored_market_data",
                "recommendations": "disabled",
            },
            "context_analysis": {
                "source": "explicit_source_items",
                "recommendations": "disabled",
            },
            "orchestration": {
                "execution_policy": "sequential_local_safe",
                "stored_run_count": orchestrator_runs.count(),
                "recommendations": "disabled",
            },
            "background_research": {
                **background_stats,
                "queue": "in_process",
                "scheduled_monitoring": "disabled",
            },
            "synthesis": {
                "source": "orchestrator_specialist_handoffs",
                "recommendations": "disabled",
            },
            "observability": {
                "source": "stored_orchestrator_runs",
                "hosted_telemetry": "disabled",
                "debug_bundle": "redacted_local_json",
            },
            "interoperability": settings_for_request.interoperability.to_dict(),
            "storage": {
                "provider": settings_for_request.storage.provider,
                "app_home": str(settings_for_request.local_paths.app_home),
                "dataset_count": len(storage.dataset_specs),
            },
            "performance": _performance_status_payload(settings_for_request, embeddings_cache),
        }

    @app.get("/api/settings")
    def get_runtime_settings() -> dict[str, Any]:
        settings_for_request = current_settings()
        registry_for_request = current_registry()
        return _settings_payload(
            settings_for_request,
            runtime_settings.get(),
            registry_for_request,
            storage,
            embeddings_cache,
        )

    @app.put("/api/settings")
    def update_runtime_settings(request: RuntimeSettingsRequest) -> dict[str, Any]:
        try:
            runtime_settings.update(request.overrides(), base_settings=app_settings)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail={"error": "invalid_runtime_settings", "message": str(exc)},
            ) from exc
        settings_for_request = current_settings()
        registry_for_request = current_registry()
        return _settings_payload(
            settings_for_request,
            runtime_settings.get(),
            registry_for_request,
            storage,
            embeddings_cache,
        )

    @app.delete("/api/settings")
    def clear_runtime_settings() -> dict[str, Any]:
        runtime_settings.clear()
        settings_for_request = current_settings()
        registry_for_request = current_registry()
        return _settings_payload(
            settings_for_request,
            runtime_settings.get(),
            registry_for_request,
            storage,
            embeddings_cache,
        )

    @app.get("/api/settings/provider-health")
    async def runtime_provider_health(
        provider: str | None = Query(default=None, min_length=1, max_length=80),
    ) -> dict[str, Any]:
        settings_for_request = current_settings()
        selected_provider = provider or settings_for_request.provider.llm_provider
        return {
            "provider_health": await _provider_health_payload(
                selected_provider,
                settings_for_request,
            )
        }

    @app.get("/api/storage")
    def storage_status() -> dict[str, Any]:
        return {"storage": storage.inspect().to_dict()}

    @app.post("/api/storage/migrate")
    def migrate_storage() -> dict[str, Any]:
        return {"result": storage.migrate().to_dict()}

    @app.delete("/api/storage/cache")
    def clear_storage_cache() -> dict[str, Any]:
        return {"result": storage.clear_cache().to_dict()}

    @app.post("/api/sessions")
    def create_session() -> dict[str, Any]:
        return {"session": sessions.create().to_dict()}

    @app.get("/api/sessions")
    def list_sessions() -> dict[str, Any]:
        return {"sessions": [session.to_dict() for session in sessions.list()]}

    @app.delete("/api/sessions")
    def clear_sessions() -> dict[str, Any]:
        return {"deleted": sessions.clear()}

    @app.get("/api/sessions/{session_id}")
    def get_session(session_id: str) -> dict[str, Any]:
        session = sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail={"error": "session_not_found"})
        return {"session": session.to_dict()}

    @app.delete("/api/sessions/{session_id}")
    def delete_session(session_id: str) -> dict[str, Any]:
        if not sessions.delete(session_id):
            raise HTTPException(status_code=404, detail={"error": "session_not_found"})
        return {"deleted": True}

    @app.get("/api/company-search")
    async def search_companies(
        query: str = Query(min_length=1, max_length=200),
        limit: int = Query(default=10, ge=1, le=50),
    ) -> dict[str, Any]:
        content = _message_content(query)
        try:
            result = await company_search.search(content, limit=limit)
        except CompanySearchError as exc:
            raise HTTPException(
                status_code=_status_for_company_search_error(exc.code),
                detail=exc.to_dict(),
            ) from exc
        return {"result": result.to_dict()}

    @app.get("/api/market-data/history/{symbol}")
    def get_market_history(symbol: str) -> dict[str, Any]:
        stored = market_store.get_history(
            symbol=symbol,
            provider=app_settings.data_sources.market_data_provider,
        )
        if stored is None:
            raise HTTPException(status_code=404, detail={"error": "market_data_not_found"})
        return {"history": stored.to_dict(), "stored": True}

    @app.post("/api/market-data/history")
    async def fetch_market_history(request: MarketDataHistoryRequest) -> dict[str, Any]:
        security = MarketSecurity(
            symbol=request.symbol,
            security_id=request.security_id,
            exchange_mic=request.exchange_mic,
            exchange_name=request.exchange_name,
            currency=request.currency,
        )
        if not request.refresh:
            stored = market_store.get_history(
                symbol=security.symbol,
                provider=app_settings.data_sources.market_data_provider,
            )
            if stored is not None:
                return {"history": stored.to_dict(), "stored": True}
        try:
            history = await market_provider.fetch_daily_prices(
                security,
                outputsize=request.outputsize,
            )
        except MarketDataError as exc:
            raise HTTPException(
                status_code=_status_for_market_data_error(exc.code),
                detail=exc.to_dict(),
            ) from exc
        stored_history = market_store.save_history(_history_with_missing_metadata_warnings(history))
        return {"history": stored_history.to_dict(), "stored": False}

    @app.get("/api/financial-statements/{cik}")
    def get_financial_statements(cik: str) -> dict[str, Any]:
        stored = statement_store.get_result(
            cik=cik,
            provider=app_settings.data_sources.financial_statement_provider,
        )
        if stored is None:
            raise HTTPException(status_code=404, detail={"error": "financial_statements_not_found"})
        return {"statements": stored.to_dict(), "stored": True}

    @app.post("/api/financial-statements")
    async def fetch_financial_statements(
        request: FinancialStatementRequest,
    ) -> dict[str, Any]:
        company = FinancialStatementCompany(
            cik=request.cik,
            company_id=request.company_id,
            legal_name=request.legal_name,
        )
        if not request.refresh:
            stored = statement_store.get_result(
                cik=company.cik,
                provider=app_settings.data_sources.financial_statement_provider,
            )
            if stored is not None:
                return {"statements": stored.to_dict(), "stored": True}
        try:
            result = await statement_provider.fetch_statements(
                company,
                fiscal_years=request.fiscal_years,
            )
        except FinancialStatementError as exc:
            raise HTTPException(
                status_code=_status_for_financial_statement_error(exc.code),
                detail=exc.to_dict(),
            ) from exc
        stored_result = statement_store.save_result(result)
        return {"statements": stored_result.to_dict(), "stored": False}

    @app.get("/api/filings/{cik}")
    def get_filings(cik: str) -> dict[str, Any]:
        stored = filings.get_result(
            cik=cik,
            provider=app_settings.data_sources.filing_provider,
        )
        if stored is None:
            raise HTTPException(status_code=404, detail={"error": "filings_not_found"})
        return {"filings": stored.to_dict(), "stored": True}

    @app.post("/api/filings/ingest")
    async def ingest_filings(request: FilingIngestionRequest) -> dict[str, Any]:
        company = FilingCompany(
            cik=request.cik,
            company_id=request.company_id,
            legal_name=request.legal_name,
        )
        if not request.refresh:
            stored = filings.get_result(
                cik=company.cik,
                provider=app_settings.data_sources.filing_provider,
            )
            if stored is not None:
                return {"filings": stored.to_dict(), "stored": True}
        try:
            result = await filing_source.ingest_latest(
                company,
                forms=request.forms,
                limit=request.limit,
            )
        except FilingError as exc:
            raise HTTPException(
                status_code=_status_for_filing_error(exc.code),
                detail=exc.to_dict(),
            ) from exc
        stored_result = filings.save_result(result)
        return {"filings": stored_result.to_dict(), "stored": False}

    @app.post("/api/financial-report-analysis")
    def analyze_financial_report(request: FinancialReportAnalysisRequest) -> dict[str, Any]:
        company = FinancialReportAnalysisCompany(
            cik=request.cik,
            company_id=request.company_id,
            legal_name=request.legal_name,
        )
        result = financial_report_agent.analyze(company)
        return {"analysis": result.to_dict()}

    @app.post("/api/stock-price-analysis")
    def analyze_stock_price(request: StockPriceAnalysisRequest) -> dict[str, Any]:
        security = StockPriceAnalysisSecurity(
            symbol=request.symbol,
            security_id=request.security_id,
            exchange_mic=request.exchange_mic,
            exchange_name=request.exchange_name,
            currency=request.currency,
        )
        result = stock_price_agent.analyze(
            security,
            benchmark_symbol=request.benchmark_symbol,
        )
        return {"analysis": result.to_dict()}

    @app.post("/api/context-analysis")
    def analyze_context(request: ContextAnalysisRequest) -> dict[str, Any]:
        try:
            result = context_agent.analyze(
                query=request.query,
                source_items=tuple(_context_source_item(item) for item in request.source_items),
                company_symbols=request.company_symbols,
                sector=request.sector,
                region=request.region,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail={"error": "invalid_context_source", "message": str(exc)},
            ) from exc
        return {"analysis": result.to_dict()}

    @app.post("/api/orchestrator/research")
    async def run_orchestrator_research(
        request: OrchestratorResearchRequest,
    ) -> dict[str, Any]:
        try:
            run = await orchestrator.run(_orchestrator_input(request))
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail={"error": "invalid_orchestrator_request", "message": str(exc)},
            ) from exc
        return {"run": run.to_dict(), "synthesis_report": _synthesis_report_from_run(run)}

    @app.post("/api/background/research-runs", status_code=202)
    async def enqueue_background_research(
        request: OrchestratorResearchRequest,
    ) -> dict[str, Any]:
        try:
            research_input = _orchestrator_input(request)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail={"error": "invalid_orchestrator_request", "message": str(exc)},
            ) from exc
        job = await background_research.submit(research_input, run=orchestrator.run)
        return _background_job_payload(job, orchestrator_runs)

    @app.get("/api/background/research-runs")
    async def list_background_research_runs() -> dict[str, Any]:
        return {
            "jobs": [
                _background_job_payload(job, orchestrator_runs)["job"]
                for job in await background_research.list()
            ],
            "limits": await background_research.stats(),
        }

    @app.get("/api/background/research-runs/{job_id}")
    async def get_background_research_run(job_id: str) -> dict[str, Any]:
        job = await background_research.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail={"error": "background_job_not_found"})
        return _background_job_payload(job, orchestrator_runs)

    @app.post("/api/background/research-runs/{job_id}/cancel")
    async def cancel_background_research_run(job_id: str) -> dict[str, Any]:
        job = await background_research.cancel(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail={"error": "background_job_not_found"})
        _mark_cancelled_orchestrator_run(job, orchestrator_runs)
        return _background_job_payload(job, orchestrator_runs)

    @app.post("/api/sessions/{session_id}/synthesis-report")
    async def post_session_synthesis_report(
        session_id: str,
        request: OrchestratorResearchRequest,
    ) -> dict[str, Any]:
        if sessions.get(session_id) is None:
            raise HTTPException(status_code=404, detail={"error": "session_not_found"})
        try:
            run = await orchestrator.run(_orchestrator_input(request))
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail={"error": "invalid_orchestrator_request", "message": str(exc)},
            ) from exc
        report = _synthesis_report_from_run(run)
        assistant_content = _synthesis_message_content(run, report)
        updated_session = sessions.append_exchange(
            session_id=session_id,
            user_content=request.query,
            assistant_content=assistant_content,
            provider="orchestrator",
            model=run.execution_policy.value,
            research_run_id=run.id,
            synthesis_report=report,
        )
        return {
            "session": updated_session.to_dict(),
            "assistant_message": updated_session.messages[-1].to_dict(),
            "run": run.to_dict(),
            "synthesis_report": report,
            "provider": "orchestrator",
            "model": run.execution_policy.value,
        }

    @app.post("/api/sessions/{session_id}/synthesis-report/background", status_code=202)
    async def enqueue_session_synthesis_report(
        session_id: str,
        request: OrchestratorResearchRequest,
    ) -> dict[str, Any]:
        if sessions.get(session_id) is None:
            raise HTTPException(status_code=404, detail={"error": "session_not_found"})
        try:
            research_input = _orchestrator_input(request)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail={"error": "invalid_orchestrator_request", "message": str(exc)},
            ) from exc

        async def run_and_append(
            input_request: OrchestratorResearchInput,
        ) -> OrchestratedResearchRun:
            run = await orchestrator.run(input_request)
            report = _synthesis_report_from_run(run)
            assistant_content = _synthesis_message_content(run, report)
            sessions.append_exchange(
                session_id=session_id,
                user_content=request.query,
                assistant_content=assistant_content,
                provider="orchestrator",
                model=run.execution_policy.value,
                research_run_id=run.id,
                synthesis_report=report,
            )
            return run

        job = await background_research.submit(
            research_input,
            run=run_and_append,
            metadata={"session_id": session_id},
        )
        return _background_job_payload(job, orchestrator_runs)

    @app.get("/api/orchestrator/runs")
    def list_orchestrator_runs() -> dict[str, Any]:
        return {
            "runs": [
                {
                    **run.to_dict(),
                    "synthesis_report": _synthesis_report_from_run(run),
                }
                for run in orchestrator_runs.list()
            ]
        }

    @app.get("/api/orchestrator/runs/{run_id}")
    def get_orchestrator_run(run_id: str) -> dict[str, Any]:
        run = orchestrator_runs.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail={"error": "orchestrator_run_not_found"})
        return {"run": run.to_dict(), "synthesis_report": _synthesis_report_from_run(run)}

    @app.get("/api/orchestrator/runs/{run_id}/trace")
    def get_orchestrator_run_trace(run_id: str) -> dict[str, Any]:
        run = _orchestrator_run_or_404(orchestrator_runs, run_id)
        trace = build_trace_from_orchestrator_run(
            run,
            redaction_policy=RedactionPolicy.from_settings(app_settings),
        )
        return {"trace": trace.to_dict()}

    @app.post("/api/orchestrator/runs/{run_id}/replay")
    def replay_orchestrator_run(run_id: str) -> dict[str, Any]:
        run = _orchestrator_run_or_404(orchestrator_runs, run_id)
        replay = build_replay_plan(
            run,
            redaction_policy=RedactionPolicy.from_settings(app_settings),
        )
        return {"replay": replay.to_dict()}

    @app.get("/api/orchestrator/runs/{run_id}/debug-bundle")
    def get_orchestrator_debug_bundle(run_id: str) -> dict[str, Any]:
        run = _orchestrator_run_or_404(orchestrator_runs, run_id)
        bundle = build_debug_bundle(run, settings=app_settings)
        return {"debug_bundle": bundle.to_dict()}

    @app.get("/api/retrieval/index")
    def get_retrieval_index() -> dict[str, Any]:
        return {"index": retrieval.metadata().to_dict()}

    @app.post("/api/retrieval/index/filings")
    async def index_stored_filings(request: RetrievalIndexFilingRequest) -> dict[str, Any]:
        stored = filings.get_result(
            cik=request.cik,
            provider=app_settings.data_sources.filing_provider,
        )
        if stored is None:
            raise HTTPException(status_code=404, detail={"error": "filings_not_found"})
        settings_for_request = current_settings()
        registry_for_request = current_registry()
        selection = settings_for_request.provider.selection_for_task(ProviderTask.EMBEDDINGS)
        try:
            embedding_provider = _embedding_provider_for_request(
                registry_for_request,
                settings_for_request,
                embeddings_cache,
                selection.provider,
            )
            result = await index_filing_result(
                stored,
                index=retrieval,
                embedding_provider=embedding_provider,
                embedding_model=selection.model,
                replace_company=request.rebuild,
            )
        except RetrievalError as exc:
            raise HTTPException(
                status_code=_status_for_retrieval_error(exc.code),
                detail=exc.to_dict(),
            ) from exc
        except ProviderError as exc:
            raise HTTPException(
                status_code=_status_for_provider_error(exc.code),
                detail=_provider_error_detail(exc),
            ) from exc
        return {"result": result.to_dict(), "index": retrieval.metadata().to_dict()}

    @app.delete("/api/retrieval/index")
    def clear_retrieval_index() -> dict[str, Any]:
        cleared_records = retrieval.clear()
        return {"cleared_records": cleared_records, "index": retrieval.metadata().to_dict()}

    @app.post("/api/retrieval/search")
    async def search_retrieval_index(request: RetrievalSearchRequest) -> dict[str, Any]:
        settings_for_request = current_settings()
        registry_for_request = current_registry()
        selection = settings_for_request.provider.selection_for_task(ProviderTask.EMBEDDINGS)
        query = RetrievalQuery(
            query=_message_content(request.query),
            top_k=request.top_k or settings_for_request.retrieval.top_k,
            min_score=(
                request.min_score
                if request.min_score is not None
                else settings_for_request.retrieval.min_score
            ),
            filters=request.filters,
        )
        try:
            embedding_provider = _embedding_provider_for_request(
                registry_for_request,
                settings_for_request,
                embeddings_cache,
                selection.provider,
            )
            result = await search_index(
                query,
                index=retrieval,
                embedding_provider=embedding_provider,
                embedding_model=selection.model,
            )
        except RetrievalError as exc:
            raise HTTPException(
                status_code=_status_for_retrieval_error(exc.code),
                detail=exc.to_dict(),
            ) from exc
        except ProviderError as exc:
            raise HTTPException(
                status_code=_status_for_provider_error(exc.code),
                detail=_provider_error_detail(exc),
            ) from exc
        return {"result": result.to_dict()}

    @app.get("/api/research-runs/{run_id}")
    def get_research_run(run_id: str) -> dict[str, Any]:
        run = report_runs.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail={"error": "research_run_not_found"})
        return {"research_run": run.to_dict()}

    @app.post("/api/sessions/{session_id}/cited-answer")
    async def post_cited_answer(
        session_id: str,
        request: CitedAnswerRequest,
    ) -> dict[str, Any]:
        session = sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail={"error": "session_not_found"})
        content = _message_content(request.content)
        settings_for_request = current_settings()
        registry_for_request = current_registry()
        retrieval_query = RetrievalQuery(
            query=content,
            top_k=request.top_k or settings_for_request.retrieval.top_k,
            min_score=(
                request.min_score
                if request.min_score is not None
                else settings_for_request.retrieval.min_score
            ),
            filters=request.filters,
        )
        try:
            embedding_selection = settings_for_request.provider.selection_for_task(
                ProviderTask.EMBEDDINGS
            )
            embedding_provider = _embedding_provider_for_request(
                registry_for_request,
                settings_for_request,
                embeddings_cache,
                embedding_selection.provider,
            )
            retrieval_result = await search_index(
                retrieval_query,
                index=retrieval,
                embedding_provider=embedding_provider,
                embedding_model=embedding_selection.model,
            )
        except RetrievalError as exc:
            if exc.code not in {RetrievalErrorCode.INDEX_EMPTY, RetrievalErrorCode.INDEX_NOT_FOUND}:
                raise HTTPException(
                    status_code=_status_for_retrieval_error(exc.code),
                    detail=exc.to_dict(),
                ) from exc
            return _append_limited_cited_answer(
                sessions=sessions,
                report_runs=report_runs,
                session_id=session_id,
                query=content,
                limitation=missing_evidence_limitation(content),
            )
        except ProviderError as exc:
            raise HTTPException(
                status_code=_status_for_provider_error(exc.code),
                detail=_provider_error_detail(exc),
            ) from exc

        citations, evidence = citation_artifacts_from_retrieval(retrieval_result)
        if not citations:
            return _append_limited_cited_answer(
                sessions=sessions,
                report_runs=report_runs,
                session_id=session_id,
                query=content,
                limitation=missing_evidence_limitation(content),
            )

        chat_selection = settings_for_request.provider.selection_for_task(ProviderTask.CHAT)
        try:
            provider = registry_for_request.chat_provider(chat_selection.provider)
            chat_request = _budgeted_chat_request(
                ChatRequest(
                    messages=build_rag_messages(content, evidence),
                    model=chat_selection.model,
                ),
                settings_for_request,
                budget_name="cited_answer",
            )
            measured = await measured_chat(provider, chat_request)
            response = measured.value
        except ProviderError as exc:
            raise HTTPException(
                status_code=_status_for_provider_error(exc.code),
                detail=_provider_error_detail(exc),
            ) from exc

        answer = ensure_citation_marker(response.message.content, citations)
        run = report_runs.save(
            CitedResearchRun(
                id=_new_research_run_id(),
                query=content,
                answer=answer,
                status=CitedResearchRunStatus.ANSWERED,
                created_at=datetime.now(UTC),
                citations=citations,
                evidence=evidence,
                provider=response.provider,
                model=response.model,
                usage={"provider_call": measured.metrics.to_dict()},
            )
        )
        updated_session = sessions.append_exchange(
            session_id=session_id,
            user_content=content,
            assistant_content=answer,
            provider=response.provider,
            model=response.model,
            research_run_id=run.id,
            citations=citations,
            evidence_snippets=evidence,
        )
        return {
            "session": updated_session.to_dict(),
            "assistant_message": updated_session.messages[-1].to_dict(),
            "research_run": run.to_dict(),
            "provider": response.provider,
            "model": response.model,
            "finish_reason": response.finish_reason.value,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            "performance": measured.metrics.to_dict(),
        }

    @app.post("/api/sessions/{session_id}/messages")
    async def post_message(session_id: str, request: MessageRequest) -> dict[str, Any]:
        session = sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail={"error": "session_not_found"})
        content = _message_content(request.content)
        mentions = _chat_mentions(request.mentions)
        settings_for_request = current_settings()
        registry_for_request = current_registry()
        selection = settings_for_request.provider.selection_for_task(ProviderTask.CHAT)
        try:
            provider = registry_for_request.chat_provider(selection.provider)
            chat_request = _budgeted_chat_request(
                ChatRequest(
                    messages=_request_messages(
                        session.context_messages(
                            recent_turns=sessions.recent_turns,
                            summary_max_chars=sessions.summary_max_chars,
                        ),
                        content,
                        mentions,
                    ),
                    model=selection.model,
                ),
                settings_for_request,
                budget_name="chat",
            )
            measured = await measured_chat(provider, chat_request)
            response = measured.value
        except ProviderError as exc:
            raise HTTPException(
                status_code=_status_for_provider_error(exc.code),
                detail=_provider_error_detail(exc),
            ) from exc

        updated_session = sessions.append_exchange(
            session_id=session_id,
            user_content=content,
            assistant_content=response.message.content,
            provider=response.provider,
            model=response.model,
            mentions=mentions,
        )
        return {
            "session": updated_session.to_dict(),
            "assistant_message": updated_session.messages[-1].to_dict(),
            "provider": response.provider,
            "model": response.model,
            "finish_reason": response.finish_reason.value,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            "performance": measured.metrics.to_dict(),
        }

    @app.post("/api/sessions/{session_id}/messages/stream")
    async def stream_message(
        session_id: str,
        request: MessageRequest,
    ) -> StreamingResponse:
        session = sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail={"error": "session_not_found"})
        content = _message_content(request.content)
        mentions = _chat_mentions(request.mentions)
        settings_for_request = current_settings()
        registry_for_request = current_registry()
        selection = settings_for_request.provider.selection_for_task(ProviderTask.STREAMING)
        try:
            provider = registry_for_request.chat_provider(selection.provider)
        except ProviderError as exc:
            raise HTTPException(
                status_code=_status_for_provider_error(exc.code),
                detail=_provider_error_detail(exc),
            ) from exc
        chat_request = _budgeted_chat_request(
            ChatRequest(
                messages=_request_messages(
                    session.context_messages(
                        recent_turns=sessions.recent_turns,
                        summary_max_chars=sessions.summary_max_chars,
                    ),
                    content,
                    mentions,
                ),
                model=selection.model,
            ),
            settings_for_request,
            budget_name="chat",
        )

        async def events() -> AsyncIterator[str]:
            collected_content: list[str] = []
            started_ns = perf_counter_ns()
            try:
                async for event in provider.stream_chat(chat_request):
                    if event.event_type == StreamEventType.MESSAGE_DELTA:
                        delta = event.delta or ""
                        collected_content.append(delta)
                        yield _stream_line({"type": "delta", "delta": delta})
                    elif event.event_type == StreamEventType.ERROR and event.error is not None:
                        yield _stream_error_line(event.error)
                        return
                    elif event.event_type == StreamEventType.COMPLETED:
                        if event.response is None:
                            yield _stream_error_line(
                                ProviderError(
                                    code=ProviderErrorCode.MALFORMED_RESPONSE,
                                    message="Streaming provider completed without a response.",
                                    provider=selection.provider,
                                    model=selection.model,
                                )
                            )
                            return
                        response = event.response
                        completed_ns = perf_counter_ns()
                        metrics = call_metrics_from_response(
                            response,
                            call_kind=ProviderCallKind.STREAMING_CHAT,
                            started_ns=started_ns,
                            completed_ns=completed_ns,
                        )
                        assistant_content = response.message.content or "".join(collected_content)
                        updated_session = sessions.append_exchange(
                            session_id=session_id,
                            user_content=content,
                            assistant_content=assistant_content,
                            provider=response.provider,
                            model=response.model,
                            mentions=mentions,
                        )
                        yield _stream_line(
                            {
                                "type": "completed",
                                "session": updated_session.to_dict(),
                                "assistant_message": updated_session.messages[-1].to_dict(),
                                "provider": response.provider,
                                "model": response.model,
                                "finish_reason": response.finish_reason.value,
                                "usage": {
                                    "input_tokens": response.usage.input_tokens,
                                    "output_tokens": response.usage.output_tokens,
                                    "total_tokens": response.usage.total_tokens,
                                },
                                "performance": metrics.to_dict(),
                            }
                        )
                        return
                yield _stream_error_line(
                    ProviderError(
                        code=ProviderErrorCode.MALFORMED_RESPONSE,
                        message="Streaming provider ended without a completion event.",
                        provider=selection.provider,
                        model=selection.model,
                    )
                )
            except ProviderError as exc:
                yield _stream_error_line(exc)

        return StreamingResponse(
            events(),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-cache"},
        )

    return app


def _settings_payload(
    settings: Settings,
    overrides: RuntimeSettingsOverrides,
    registry: ProviderRegistry,
    storage: LocalStorageManager,
    embedding_cache: LocalEmbeddingCache,
) -> dict[str, Any]:
    return {
        "settings": {
            "provider": settings.provider.to_dict(),
            "chat": settings.chat.to_dict(),
            "data_sources": settings.data_sources.to_dict(),
            "retrieval": settings.retrieval.to_dict(),
            "background": settings.background.to_dict(),
            "performance": settings.performance.to_dict(),
        },
        "overrides": overrides.to_dict(),
        "providers": _provider_options_payload(settings, registry),
        "secrets": {
            "strategy": "environment_only",
            "plaintext_storage": "disabled",
            "openai_api_key_configured": settings.provider.openai_api_key is not None,
            "alpha_vantage_api_key_configured": (
                settings.data_sources.alpha_vantage_api_key is not None
            ),
            "message": (
                "Secrets are not stored in the local settings override file. Configure API "
                "keys through environment variables."
            ),
        },
        "management": {
            "cache_clear_endpoint": "/api/storage/cache",
            "storage_status_endpoint": "/api/storage",
            "data_reset_command": "python -m financial_research_agent data-reset --yes --pretty",
            "settings_storage_path": str(settings.local_paths.data_dir / "settings_overrides.json"),
            "storage_dataset_count": len(storage.dataset_specs),
            "embedding_cache": embedding_cache.to_dict(),
        },
        "performance": _performance_status_payload(settings, embedding_cache),
    }


def _performance_status_payload(
    settings: Settings,
    embedding_cache: LocalEmbeddingCache,
) -> dict[str, Any]:
    return {
        **settings.performance.to_dict(),
        "prompt_budgets": {
            name: budget.to_dict()
            for name, budget in prompt_budgets_for_limits(
                max_input_tokens=settings.performance.prompt_budget_input_tokens,
                max_output_tokens=settings.performance.prompt_budget_output_tokens,
            ).items()
        },
        "local_model_profiles": [profile.to_dict() for profile in default_local_model_profiles()],
        "embedding_cache": embedding_cache.to_dict(),
        "cost_tracking": {
            "local_and_offline_costs": "zero_dollar_provider_call_estimate",
            "hosted_costs": "usage_tracked_without_default_price_card",
        },
    }


def _provider_options_payload(
    settings: Settings, registry: ProviderRegistry
) -> list[dict[str, Any]]:
    providers = (
        OfflineTestProvider().metadata,
        OpenAICompatibleLocalProvider.from_settings(settings.provider).metadata,
        OpenAIProvider.from_settings(settings.provider).metadata,
    )
    return [
        _provider_metadata_payload(
            metadata,
            chat_registered=registry.has_chat_provider(metadata.provider),
            embedding_registered=registry.has_embedding_provider(metadata.provider),
        )
        for metadata in providers
    ]


def _provider_metadata_payload(
    metadata: ModelMetadata,
    *,
    chat_registered: bool,
    embedding_registered: bool,
) -> dict[str, Any]:
    capabilities = {capability.value for capability in metadata.capabilities}
    return {
        "provider": metadata.provider,
        "model": metadata.model,
        "registered": {
            "chat": chat_registered,
            "embeddings": embedding_registered,
        },
        "capabilities": sorted(capabilities),
        "capability_status": {
            capability.value: capability.value in capabilities for capability in ProviderCapability
        },
        "context_window": metadata.context_window,
        "max_output_tokens": metadata.max_output_tokens,
        "metadata": dict(metadata.metadata),
    }


async def _provider_health_payload(provider: str, settings: Settings) -> dict[str, Any]:
    normalized = provider.strip()
    if normalized == "offline-test":
        metadata = OfflineTestProvider().metadata
        return {
            "provider": metadata.provider,
            "model": metadata.model,
            "reachable": True,
            "authenticated": True,
            "status": "ok",
            "capabilities": [capability.value for capability in metadata.capabilities],
            "limitations": ["Deterministic offline provider; no network or real model calls."],
        }
    if normalized == "local-openai":
        return (
            await OpenAICompatibleLocalProvider.from_settings(settings.provider).check_health()
        ).to_dict()
    if normalized == "openai":
        return (await OpenAIProvider.from_settings(settings.provider).check_health()).to_dict()
    raise HTTPException(
        status_code=404,
        detail={"error": "provider_not_supported", "provider": normalized},
    )


def _embedding_provider_for_request(
    registry: ProviderRegistry,
    settings: Settings,
    embedding_cache: LocalEmbeddingCache,
    provider_name: str,
):
    provider = registry.embedding_provider(provider_name)
    if not settings.performance.embedding_cache_enabled:
        return provider
    return CachingEmbeddingProvider(provider, embedding_cache)


def _budgeted_chat_request(
    request: ChatRequest,
    settings: Settings,
    *,
    budget_name: str,
) -> ChatRequest:
    budgets = prompt_budgets_for_limits(
        max_input_tokens=settings.performance.prompt_budget_input_tokens,
        max_output_tokens=settings.performance.prompt_budget_output_tokens,
    )
    base_budget = budgets.get(budget_name, budgets["chat"])
    check = base_budget.check(request)
    if check.over_budget:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "prompt_budget_exceeded",
                "budget": check.to_dict(),
            },
        )
    if request.max_output_tokens is not None:
        return request
    return replace(request, max_output_tokens=check.recommended_max_output_tokens)


def _interop_policy(settings: Settings) -> InteropAccessPolicy:
    return InteropAccessPolicy(
        enabled=settings.interoperability.enabled,
        local_only=settings.interoperability.local_only,
        api_key=settings.interoperability.api_key,
    )


def _require_interop_access(
    request: Request,
    settings: Settings,
    authorization: str | None,
    api_key_header: str | None,
) -> None:
    policy = _interop_policy(settings)
    result = policy.evaluate(
        client_host=request.client.host if request.client is not None else None,
        authorization=authorization,
        api_key_header=api_key_header,
    )
    if result.allowed:
        return
    if result.decision == InteropAccessDecision.DISABLED:
        raise HTTPException(status_code=404, detail={"error": result.reason})
    raise HTTPException(status_code=401, detail={"error": result.reason})


def _orchestrator_input(request: OrchestratorResearchRequest) -> OrchestratorResearchInput:
    return OrchestratorResearchInput(
        query=_message_content(request.query),
        refresh=request.refresh,
        company_search_limit=request.company_search_limit,
        fiscal_years=request.fiscal_years,
        filing_forms=request.filing_forms,
        filing_limit=request.filing_limit,
        market_outputsize=request.market_outputsize,
        benchmark_symbol=request.benchmark_symbol,
        context_source_items=tuple(
            _context_source_item(item) for item in request.context_source_items
        ),
    )


def _synthesis_report_from_run(run: OrchestratedResearchRun) -> dict[str, object] | None:
    for handoff in reversed(run.handoffs):
        if handoff.kind.value != "synthesis":
            continue
        report = handoff.output.get("report")
        return report if isinstance(report, dict) else None
    return None


def _orchestrator_run_or_404(
    orchestrator_runs: OrchestratorRunStore,
    run_id: str,
) -> OrchestratedResearchRun:
    run = orchestrator_runs.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail={"error": "orchestrator_run_not_found"})
    return run


def _background_job_payload(
    job: BackgroundResearchJob,
    orchestrator_runs: OrchestratorRunStore,
) -> dict[str, Any]:
    run = orchestrator_runs.get(job.orchestrator_run_id)
    progress = _orchestrator_progress(run) if run is not None else _empty_progress()
    return {
        "job": {
            **job.to_dict(),
            "progress": progress,
            "orchestrator_run": run.to_dict() if run is not None else None,
            "synthesis_report": _synthesis_report_from_run(run) if run is not None else None,
        }
    }


def _empty_progress() -> dict[str, object]:
    return {
        "completed_steps": 0,
        "total_steps": len(default_orchestrator_plan()),
        "current_step": None,
    }


def _orchestrator_progress(run: OrchestratedResearchRun) -> dict[str, object]:
    completed_steps = {handoff.step_id for handoff in run.handoffs}
    remaining_steps = tuple(step.id for step in run.plan if step.id not in completed_steps)
    return {
        "completed_steps": len(completed_steps),
        "total_steps": len(run.plan),
        "current_step": remaining_steps[0] if remaining_steps else None,
    }


def _mark_cancelled_orchestrator_run(
    job: BackgroundResearchJob,
    orchestrator_runs: OrchestratorRunStore,
) -> None:
    if job.status != BackgroundResearchStatus.CANCELLED:
        return
    run = orchestrator_runs.get(job.orchestrator_run_id)
    if run is None or run.status != OrchestratorRunStatus.RUNNING:
        return
    limitation = "Background research job was cancelled before the workflow completed."
    orchestrator_runs.save(
        replace(
            run,
            status=OrchestratorRunStatus.PARTIAL,
            limitations=tuple(dict.fromkeys((*run.limitations, limitation))),
            updated_at=datetime.now(UTC),
        )
    )


def _synthesis_message_content(
    run: OrchestratedResearchRun,
    report: dict[str, object] | None,
) -> str:
    if report is None:
        return run.synthesis_summary or "Synthesis report is unavailable."
    summary = report.get("summary")
    notice = report.get("no_recommendation_notice")
    parts = [str(summary)] if isinstance(summary, str) and summary.strip() else []
    if isinstance(notice, str) and notice.strip():
        parts.append(notice)
    return "\n\n".join(parts) if parts else "Synthesis report generated."


def _message_content(content: str) -> str:
    text = content.strip()
    if text == "":
        raise HTTPException(status_code=422, detail={"error": "message_content_required"})
    return text


def _request_messages(
    history: tuple[ChatMessage, ...],
    user_content: str,
    mentions: tuple[ChatMention, ...] = (),
) -> tuple[ChatMessage, ...]:
    mention_context = _mention_context_messages(mentions)
    return (
        ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
        *history,
        *mention_context,
        ChatMessage(role=MessageRole.USER, content=user_content),
    )


def _chat_mentions(mentions: tuple[MentionRequest, ...]) -> tuple[ChatMention, ...]:
    return tuple(
        ChatMention(
            id=mention.id,
            label=mention.label,
            company_id=mention.company_id,
            legal_name=mention.legal_name,
            ticker=mention.ticker,
            cik=mention.cik,
            source_provider=mention.source_provider,
        )
        for mention in mentions
    )


def _mention_context_messages(mentions: tuple[ChatMention, ...]) -> tuple[ChatMessage, ...]:
    if not mentions:
        return ()
    lines = [
        "Resolved @company mentions for this user message.",
        "These are identifier context only, not live financial evidence.",
        "Do not invent prices, statements, filings, or citations from these identifiers.",
    ]
    for index, mention in enumerate(mentions, start=1):
        fields = [
            f"label={mention.label}",
            f"legal_name={mention.legal_name}",
            f"company_id={mention.company_id}",
        ]
        if mention.ticker is not None:
            fields.append(f"ticker={mention.ticker}")
        if mention.cik is not None:
            fields.append(f"cik={mention.cik}")
        if mention.source_provider is not None:
            fields.append(f"source_provider={mention.source_provider}")
        lines.append(f"{index}. " + "; ".join(fields))
    return (ChatMessage(role=MessageRole.SYSTEM, content="\n".join(lines)),)


def _context_source_item(request: ContextSourceItemRequest) -> ContextSourceItem:
    return ContextSourceItem(
        id=request.id,
        title=request.title,
        summary=request.summary,
        source_url=request.source_url,
        source_name=request.source_name,
        source_type=request.source_type,
        reliability=request.reliability,
        scope=request.scope,
        retrieved_at=request.retrieved_at,
        published_at=request.published_at,
        company_symbols=request.company_symbols,
        sector=request.sector,
        region=request.region,
        topics=request.topics,
        metadata=request.metadata,
    )


def _append_limited_cited_answer(
    *,
    sessions: ChatSessionStore,
    report_runs: CitedResearchRunStore,
    session_id: str,
    query: str,
    limitation: str,
) -> dict[str, Any]:
    run = report_runs.save(
        CitedResearchRun(
            id=_new_research_run_id(),
            query=query,
            answer=limitation,
            status=CitedResearchRunStatus.LIMITED,
            created_at=datetime.now(UTC),
            provider="retrieval",
            model="no-evidence",
            usage={
                "provider_call": {
                    "call_kind": "retrieval",
                    "provider": "retrieval",
                    "model": "no-evidence",
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "latency_ms": 0,
                    "estimated_cost_usd": "0.000000",
                    "cost_source": "no_provider_call",
                    "warnings": [],
                }
            },
            limitations=(limitation,),
        )
    )
    updated_session = sessions.append_exchange(
        session_id=session_id,
        user_content=query,
        assistant_content=limitation,
        provider="retrieval",
        model="no-evidence",
        research_run_id=run.id,
    )
    return {
        "session": updated_session.to_dict(),
        "assistant_message": updated_session.messages[-1].to_dict(),
        "research_run": run.to_dict(),
        "provider": "retrieval",
        "model": "no-evidence",
        "finish_reason": "stop",
        "usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        },
        "performance": run.usage["provider_call"],
    }


def _new_research_run_id() -> str:
    return f"research_run_{uuid4().hex}"


def _status_for_provider_error(code: ProviderErrorCode) -> int:
    if code == ProviderErrorCode.AUTHENTICATION_FAILED:
        return 401
    if code == ProviderErrorCode.RATE_LIMITED:
        return 429
    if code == ProviderErrorCode.TIMEOUT:
        return 504
    if code in {
        ProviderErrorCode.INVALID_REQUEST,
        ProviderErrorCode.UNSUPPORTED_FEATURE,
        ProviderErrorCode.CONTEXT_LENGTH_EXCEEDED,
    }:
        return 400
    return 503


def _provider_error_detail(error: ProviderError) -> dict[str, Any]:
    return {
        "error": "provider_error",
        "code": error.code.value,
        "message": error.message,
        "provider": error.provider,
        "model": error.model,
        "retryable": error.retryable,
    }


def _stream_line(payload: dict[str, Any]) -> str:
    return json.dumps(payload, separators=(",", ":")) + "\n"


def _stream_error_line(error: ProviderError) -> str:
    return _stream_line(
        {
            "type": "error",
            "status": _status_for_provider_error(error.code),
            "detail": _provider_error_detail(error),
        }
    )


def _status_for_company_search_error(code: CompanySearchErrorCode) -> int:
    if code == CompanySearchErrorCode.INVALID_REQUEST:
        return 400
    if code == CompanySearchErrorCode.RATE_LIMITED:
        return 429
    if code == CompanySearchErrorCode.TIMEOUT:
        return 504
    return 503


def _status_for_market_data_error(code: MarketDataErrorCode) -> int:
    if code == MarketDataErrorCode.AUTHENTICATION_FAILED:
        return 401
    if code == MarketDataErrorCode.RATE_LIMITED:
        return 429
    if code == MarketDataErrorCode.TIMEOUT:
        return 504
    if code == MarketDataErrorCode.INVALID_REQUEST:
        return 400
    if code == MarketDataErrorCode.NOT_FOUND:
        return 404
    return 503


def _status_for_financial_statement_error(code: FinancialStatementErrorCode) -> int:
    if code == FinancialStatementErrorCode.RATE_LIMITED:
        return 429
    if code == FinancialStatementErrorCode.TIMEOUT:
        return 504
    if code == FinancialStatementErrorCode.INVALID_REQUEST:
        return 400
    if code == FinancialStatementErrorCode.NOT_FOUND:
        return 404
    return 503


def _status_for_filing_error(code: FilingErrorCode) -> int:
    if code == FilingErrorCode.RATE_LIMITED:
        return 429
    if code == FilingErrorCode.TIMEOUT:
        return 504
    if code in {
        FilingErrorCode.INVALID_REQUEST,
        FilingErrorCode.UNSUPPORTED_FORMAT,
        FilingErrorCode.DOCUMENT_TOO_LARGE,
    }:
        return 400
    if code == FilingErrorCode.NOT_FOUND:
        return 404
    return 503


def _status_for_retrieval_error(code: RetrievalErrorCode) -> int:
    if code == RetrievalErrorCode.INVALID_REQUEST:
        return 400
    if code in {RetrievalErrorCode.INDEX_EMPTY, RetrievalErrorCode.INDEX_NOT_FOUND}:
        return 404
    if code == RetrievalErrorCode.VECTOR_DIMENSION_MISMATCH:
        return 400
    return 503


def _history_with_missing_metadata_warnings(history):
    warnings = list(history.warnings)
    if history.security.currency is None:
        warnings.append("Currency metadata is unavailable for this security.")
    if history.security.exchange_mic is None and history.security.exchange_name is None:
        warnings.append("Exchange metadata is unavailable for this security.")
    return replace(history, warnings=tuple(dict.fromkeys(warnings)))
