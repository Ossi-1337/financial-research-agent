from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from time import perf_counter_ns
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from financial_research_agent.a2a import (
    SQLiteA2ADelegationStore,
    create_a2a_dispatcher,
)
from financial_research_agent.agents import (
    AgentDecisionMode,
    AgentRuntimeError,
    AgentRuntimeResolver,
)
from financial_research_agent.background import BackgroundResearchRunner
from financial_research_agent.entities import (
    CompanySearchError,
    CompanySearchErrorCode,
    CompanySearchProvider,
    create_default_company_search_provider,
)
from financial_research_agent.filings import FilingStore
from financial_research_agent.llm import (
    ModelMetadata,
    ProviderCapability,
    ProviderError,
    ProviderErrorCode,
    StreamEventType,
)
from financial_research_agent.llm.anthropic import AnthropicProvider
from financial_research_agent.llm.gemini import GeminiProvider
from financial_research_agent.llm.litellm import LiteLLMGatewayProvider
from financial_research_agent.llm.local_openai import OpenAICompatibleLocalProvider
from financial_research_agent.llm.offline import OfflineTestProvider
from financial_research_agent.llm.openai import OpenAIProvider
from financial_research_agent.llm.registry import ProviderRegistry, create_default_provider_registry
from financial_research_agent.market_data import MarketDataStore
from financial_research_agent.observability import (
    RedactionPolicy,
)
from financial_research_agent.orchestration import (
    OrchestratedResearchRun,
    OrchestratorResearchInput,
    OrchestratorRunStore,
    ResearchOrchestrator,
    ResearchStepDispatcher,
    UnavailableResearchStepDispatcher,
)
from financial_research_agent.performance import (
    LocalEmbeddingCache,
    ProviderCallKind,
    call_metrics_from_response,
    default_local_model_profiles,
    prompt_budgets_for_limits,
)
from financial_research_agent.persistence import (
    PersistenceError,
    create_persistence,
    create_storage_manager,
)
from financial_research_agent.report_exports import (
    ReportExportService,
    ReportExportStore,
)
from financial_research_agent.reports import (
    CitedResearchRunStore,
)
from financial_research_agent.retrieval import (
    LocalVectorIndex,
)
from financial_research_agent.runtime_settings import RuntimeSettingsOverrides, RuntimeSettingsStore
from financial_research_agent.scenarios import create_default_scenario_catalog
from financial_research_agent.security import ConversationPolicy
from financial_research_agent.settings import ProviderTask, Settings
from financial_research_agent.statements import FinancialStatementStore
from financial_research_agent.storage import LocalStorageManager
from financial_research_agent.web.conversation import AgentConversationService
from financial_research_agent.web.report_routes import create_report_router
from financial_research_agent.web.research_routes import (
    background_job_payload,
    create_research_router,
    synthesis_report_from_run,
)
from financial_research_agent.web.sessions import ChatMention, ChatSessionStore


class MentionRequest(BaseModel):
    id: str = Field(min_length=1, max_length=200)
    label: str = Field(min_length=1, max_length=200)
    company_id: str = Field(min_length=1, max_length=200)
    legal_name: str = Field(min_length=1, max_length=300)
    ticker: str | None = Field(default=None, max_length=32)
    cik: str | None = Field(default=None, max_length=10, pattern="^[0-9]+$")
    source_provider: str | None = Field(default=None, max_length=80)


class MessageRequest(BaseModel):
    content: str = Field(min_length=1, max_length=ConversationPolicy.MAX_INPUT_CHARS)
    mentions: tuple[MentionRequest, ...] = Field(
        default=(),
        max_length=ConversationPolicy.MAX_MENTIONS,
    )


class RuntimeSettingsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    llm_provider: str | None = Field(default=None, min_length=1, max_length=80)
    llm_model: str | None = Field(default=None, min_length=1, max_length=200)
    llm_base_url: str | None = Field(default=None, max_length=500)
    llm_local_runtime: str | None = Field(default=None, min_length=1, max_length=80)
    llm_timeout_seconds: float | None = Field(default=None, gt=0, le=300)
    openai_api_key: str | None = Field(default=None, max_length=500)
    anthropic_api_key: str | None = Field(default=None, max_length=500)
    gemini_api_key: str | None = Field(default=None, max_length=500)
    litellm_api_key: str | None = Field(default=None, max_length=500)
    alpha_vantage_api_key: str | None = Field(default=None, max_length=500)
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
    market_data_store: MarketDataStore | None = None,
    financial_statement_store: FinancialStatementStore | None = None,
    filing_store: FilingStore | None = None,
    storage_manager: LocalStorageManager | None = None,
    retrieval_index: LocalVectorIndex | None = None,
    report_run_store: CitedResearchRunStore | None = None,
    orchestrator_run_store: OrchestratorRunStore | None = None,
    report_export_store: ReportExportStore | None = None,
    background_runner: BackgroundResearchRunner | None = None,
    runtime_settings_store: RuntimeSettingsStore | None = None,
    embedding_cache: LocalEmbeddingCache | None = None,
    research_dispatcher: ResearchStepDispatcher | None = None,
) -> FastAPI:
    app_settings = settings or Settings.from_env()
    provider_registry = registry or create_default_provider_registry(app_settings.provider)
    uses_default_research_runtime = (
        company_search_provider is None and orchestrator_run_store is None
    )
    all_stores_injected = all(
        store is not None
        for store in (
            session_store,
            market_data_store,
            financial_statement_store,
            filing_store,
            report_run_store,
            orchestrator_run_store,
            runtime_settings_store,
        )
    )
    persistence = None if all_stores_injected else create_persistence(app_settings)
    sessions = session_store or persistence.sessions
    company_search = company_search_provider or create_default_company_search_provider(app_settings)
    market_store = market_data_store or persistence.market_data
    statement_store = financial_statement_store or persistence.financial_statements
    filings = filing_store or persistence.filings
    storage = storage_manager or create_storage_manager(app_settings)
    retrieval = retrieval_index or LocalVectorIndex.from_settings(app_settings)
    report_runs = report_run_store or persistence.cited_runs
    orchestrator_runs = orchestrator_run_store or persistence.orchestrator_runs
    report_exports = report_export_store or ReportExportStore.from_settings(app_settings)
    report_export_service = ReportExportService(
        store=report_exports,
        redaction_policy=RedactionPolicy.from_settings(app_settings),
    )
    scenario_catalog = create_default_scenario_catalog()
    runtime_settings = runtime_settings_store or persistence.runtime_settings
    embeddings_cache = embedding_cache or LocalEmbeddingCache.from_settings(app_settings)

    def current_settings() -> Settings:
        return runtime_settings.settings(app_settings)

    def current_registry() -> ProviderRegistry:
        if registry is not None:
            return provider_registry
        return create_default_provider_registry(current_settings().provider)

    agent_runtime = AgentRuntimeResolver(
        settings=current_settings,
        registry=lambda _current: current_registry(),
    )

    def current_agent_runtime_selection() -> tuple[str, str]:
        selection = agent_runtime.resolve(require_research=True)
        return selection.provider_name, selection.model

    background_research = background_runner or BackgroundResearchRunner(
        max_concurrent_runs=app_settings.background.max_concurrent_research_runs,
        job_store=persistence.background_jobs if persistence is not None else None,
    )
    dispatcher = research_dispatcher
    if (
        dispatcher is None
        and uses_default_research_runtime
        and persistence is not None
        and persistence.database is not None
    ):
        dispatcher = create_a2a_dispatcher(
            app_settings,
            delegation_store=SQLiteA2ADelegationStore(persistence.database),
        )
    a2a_available = dispatcher is not None
    dispatcher = dispatcher or UnavailableResearchStepDispatcher()
    orchestrator = ResearchOrchestrator(
        company_search_provider=company_search,
        run_store=orchestrator_runs,
        step_dispatcher=dispatcher,
        agent_runtime_selection=current_agent_runtime_selection,
    )
    static_dir = Path(__file__).with_name("static")

    app = FastAPI(title="Financial Research Agent", version="0.1.0")
    app.state.settings = app_settings
    app.state.persistence = persistence
    app.state.provider_registry = provider_registry
    app.state.sessions = sessions
    app.state.company_search = company_search
    app.state.market_data_store = market_store
    app.state.financial_statement_store = statement_store
    app.state.filing_store = filings
    app.state.storage_manager = storage
    app.state.runtime_settings_store = runtime_settings
    app.state.agent_runtime = agent_runtime
    app.state.embedding_cache = embeddings_cache
    app.state.retrieval_index = retrieval
    app.state.report_run_store = report_runs
    app.state.orchestrator_run_store = orchestrator_runs
    app.state.report_export_store = report_exports
    app.state.report_export_service = report_export_service
    app.state.background_research = background_research
    app.state.research_dispatcher = dispatcher
    app.state.orchestrator = orchestrator
    app.state.scenario_catalog = scenario_catalog

    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    conversation_policy = ConversationPolicy()
    conversation = AgentConversationService(
        settings=current_settings,
        registry=current_registry,
        agent_runtime=agent_runtime,
        policy=conversation_policy,
    )
    app.include_router(
        create_research_router(
            background_research=background_research,
            orchestrator_runs=orchestrator_runs,
            settings=current_settings,
        )
    )
    app.include_router(
        create_report_router(
            orchestrator_runs=orchestrator_runs,
            report_exports=report_exports,
            report_export_service=report_export_service,
        )
    )

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/api/status")
    async def status() -> dict[str, Any]:
        settings_for_request = current_settings()
        registry_for_request = current_registry()
        selection = settings_for_request.provider.selection_for_task(ProviderTask.CHAT)
        embedding_selection = settings_for_request.provider.selection_for_task(
            ProviderTask.EMBEDDINGS
        )
        a2a_settings = settings_for_request.a2a.to_dict()
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
            "research_agent_runtime": _agent_runtime_status_payload(agent_runtime),
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
                "execution_policy": "distributed_a2a",
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
            "a2a": {
                "enabled": a2a_available,
                "role": "orchestrator",
                "protocol_version": a2a_settings["protocol_version"],
                "protocol_binding": a2a_settings["protocol_binding"],
                "delegation_timeout_seconds": (settings_for_request.a2a.delegation_timeout_seconds),
                "delegation_max_attempts": (settings_for_request.a2a.delegation_max_attempts),
                "specialists": {
                    "financial_report": settings_for_request.a2a.financial_report_url,
                    "stock": settings_for_request.a2a.stock_url,
                    "context": settings_for_request.a2a.context_url,
                    "synthesis": settings_for_request.a2a.synthesis_url,
                },
            },
            "security": settings_for_request.security.to_dict(),
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
            agent_runtime,
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
            agent_runtime,
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
            agent_runtime,
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

    @app.get("/api/storage/integrity")
    def storage_integrity(full: bool = False) -> dict[str, Any]:
        if persistence is None or persistence.database is None:
            return {
                "integrity": {
                    "provider": "local-json",
                    "healthy": True,
                    "warning": "SQLite integrity checks are unavailable for local-json.",
                }
            }
        try:
            report = persistence.database.integrity(full=full)
        except PersistenceError as exc:
            raise HTTPException(status_code=503, detail=exc.to_dict()) from exc
        return {"integrity": report.to_dict()}

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

    @app.post("/api/sessions/{session_id}/messages")
    async def post_message(session_id: str, request: MessageRequest) -> dict[str, Any]:
        session = sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail={"error": "session_not_found"})
        content = _message_content(request.content)
        mentions = _chat_mentions(request.mentions)
        context_messages = session.context_messages(
            recent_turns=sessions.recent_turns,
            summary_max_chars=sessions.summary_max_chars,
        )
        references = tuple(mention.to_dict() for mention in mentions)
        try:
            plan = await conversation.plan(
                content=content,
                context_messages=context_messages,
                company_references=references,
            )
            conversation.ensure_research_available(plan)
            if plan.decision.mode == AgentDecisionMode.RESEARCH:
                run = await orchestrator.run(
                    OrchestratorResearchInput(
                        query=content,
                        company_query=plan.decision.company_query,
                        specialist_roles=plan.decision.specialist_roles,
                        agent_provider=plan.provider.metadata.provider,
                        agent_model=plan.model,
                        orchestrator_skill_references=tuple(
                            f"{skill.id}@{skill.version.value}" for skill in plan.decision.skills
                        ),
                    )
                )
                report = synthesis_report_from_run(run)
                assistant_content = _synthesis_message_content(run, report)
                updated_session = sessions.append_exchange(
                    session_id=session_id,
                    user_content=content,
                    assistant_content=assistant_content,
                    provider=plan.provider.metadata.provider,
                    model=plan.model,
                    research_run_id=run.id,
                    mentions=mentions,
                    synthesis_report=report,
                )
                return {
                    "session": updated_session.to_dict(),
                    "assistant_message": updated_session.messages[-1].to_dict(),
                    "provider": plan.provider.metadata.provider,
                    "model": plan.model,
                    "research_run": run.to_dict(),
                }
            response = await conversation.direct_response(
                plan,
                content=content,
                context_messages=context_messages,
                company_references=references,
            )
        except AgentRuntimeError as exc:
            raise HTTPException(
                status_code=503,
                detail={"error": exc.code, "message": exc.message},
            ) from exc
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
        context_messages = session.context_messages(
            recent_turns=sessions.recent_turns,
            summary_max_chars=sessions.summary_max_chars,
        )
        references = tuple(mention.to_dict() for mention in mentions)
        try:
            plan = await conversation.plan(
                content=content,
                context_messages=context_messages,
                company_references=references,
            )
            conversation.ensure_research_available(plan)
        except AgentRuntimeError as exc:
            raise HTTPException(
                status_code=503,
                detail={"error": exc.code, "message": exc.message},
            ) from exc
        except ProviderError as exc:
            raise HTTPException(
                status_code=_status_for_provider_error(exc.code),
                detail=_provider_error_detail(exc),
            ) from exc
        if plan.decision.mode == AgentDecisionMode.RESEARCH:

            async def run_and_append(
                research_input: OrchestratorResearchInput,
            ) -> OrchestratedResearchRun:
                run = await orchestrator.run(research_input)
                report = synthesis_report_from_run(run)
                sessions.append_exchange(
                    session_id=session_id,
                    user_content=content,
                    assistant_content=_synthesis_message_content(run, report),
                    provider=plan.provider.metadata.provider,
                    model=plan.model,
                    research_run_id=run.id,
                    mentions=mentions,
                    synthesis_report=report,
                )
                return run

            job = await background_research.submit(
                OrchestratorResearchInput(
                    query=content,
                    company_query=plan.decision.company_query,
                    specialist_roles=plan.decision.specialist_roles,
                    agent_provider=plan.provider.metadata.provider,
                    agent_model=plan.model,
                    orchestrator_skill_references=tuple(
                        f"{skill.id}@{skill.version.value}" for skill in plan.decision.skills
                    ),
                ),
                run=run_and_append,
                metadata={"session_id": session_id},
            )

            async def research_event() -> AsyncIterator[str]:
                yield _stream_line(
                    {
                        "type": "research",
                        **background_job_payload(job, orchestrator_runs),
                    }
                )

            return StreamingResponse(
                research_event(),
                media_type="application/x-ndjson",
                headers={"Cache-Control": "no-cache"},
            )

        if plan.decision.mode != AgentDecisionMode.DIRECT_ANSWER:
            response = await conversation.direct_response(
                plan,
                content=content,
                context_messages=context_messages,
                company_references=references,
            )
            updated_session = sessions.append_exchange(
                session_id=session_id,
                user_content=content,
                assistant_content=response.message.content,
                provider=response.provider,
                model=response.model,
                mentions=mentions,
            )

            async def decision_event() -> AsyncIterator[str]:
                yield _stream_line({"type": "delta", "delta": response.message.content})
                yield _stream_line(
                    {
                        "type": "completed",
                        "session": updated_session.to_dict(),
                        "assistant_message": updated_session.messages[-1].to_dict(),
                        "provider": response.provider,
                        "model": response.model,
                        "finish_reason": response.finish_reason.value,
                    }
                )

            return StreamingResponse(
                decision_event(),
                media_type="application/x-ndjson",
                headers={"Cache-Control": "no-cache"},
            )

        async def events() -> AsyncIterator[str]:
            collected_content: list[str] = []
            started_ns = perf_counter_ns()
            try:
                async for event in conversation.stream_direct_response(
                    plan,
                    content=content,
                    context_messages=context_messages,
                    company_references=references,
                ):
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
                                    provider=plan.provider.metadata.provider,
                                    model=plan.model,
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
                        provider=plan.provider.metadata.provider,
                        model=plan.model,
                    )
                )
            except ProviderError as exc:
                yield _stream_error_line(exc)
            except AgentRuntimeError as exc:
                yield _stream_agent_error_line(exc)

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
    agent_runtime: AgentRuntimeResolver,
) -> dict[str, Any]:
    return {
        "settings": {
            "provider": settings.provider.to_dict(),
            "chat": settings.chat.to_dict(),
            "data_sources": _public_data_source_settings(settings),
            "retrieval": settings.retrieval.to_dict(),
            "background": settings.background.to_dict(),
            "performance": settings.performance.to_dict(),
            "security": settings.security.to_dict(),
        },
        "overrides": overrides.to_dict(),
        "providers": _provider_options_payload(settings, registry),
        "research_agent_runtime": _agent_runtime_status_payload(agent_runtime),
        "secrets": {
            "strategy": "environment_only",
            "plaintext_storage": "disabled",
            "openai_api_key_configured": settings.provider.openai_api_key is not None,
            "anthropic_api_key_configured": settings.provider.anthropic_api_key is not None,
            "gemini_api_key_configured": settings.provider.gemini_api_key is not None,
            "litellm_api_key_configured": settings.provider.litellm_api_key is not None,
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


def _agent_runtime_status_payload(
    resolver: AgentRuntimeResolver,
) -> dict[str, Any]:
    try:
        selection = resolver.resolve()
    except AgentRuntimeError as exc:
        return {
            "provider": None,
            "model": None,
            "compatible": False,
            "error_code": exc.code,
            "message": exc.message,
            "required_capabilities": ["chat", "tool_calls", "structured_output"],
        }
    try:
        resolver.validate_research(selection)
    except AgentRuntimeError as exc:
        compatible = False
        error_code = exc.code
        message = exc.message
    else:
        compatible = True
        error_code = None
        message = "Configured provider can run research agents."
    return {
        "provider": selection.provider_name,
        "model": selection.model,
        "compatible": compatible,
        "error_code": error_code,
        "message": message,
        "required_capabilities": ["chat", "tool_calls", "structured_output"],
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
        AnthropicProvider.from_settings(settings.provider).metadata,
        GeminiProvider.from_settings(settings.provider).metadata,
        LiteLLMGatewayProvider.from_settings(settings.provider).metadata,
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


def _public_data_source_settings(settings: Settings) -> dict[str, Any]:
    payload = settings.data_sources.to_dict()
    payload.pop("sec_user_agent", None)
    payload["sec_user_agent_configured"] = bool(settings.data_sources.sec_user_agent)
    return payload


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
            "available_models": [metadata.model],
            "capabilities": [capability.value for capability in metadata.capabilities],
            "limitations": ["Deterministic offline provider; no network or real model calls."],
        }
    if normalized == "local-openai":
        return (
            await OpenAICompatibleLocalProvider.from_settings(settings.provider).check_health()
        ).to_dict()
    if normalized == "openai":
        return (await OpenAIProvider.from_settings(settings.provider).check_health()).to_dict()
    if normalized == "anthropic":
        return (await AnthropicProvider.from_settings(settings.provider).check_health()).to_dict()
    if normalized == "gemini":
        return (await GeminiProvider.from_settings(settings.provider).check_health()).to_dict()
    if normalized == "litellm":
        return (
            await LiteLLMGatewayProvider.from_settings(settings.provider).check_health()
        ).to_dict()
    raise HTTPException(
        status_code=404,
        detail={"error": "provider_not_supported", "provider": normalized},
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
    text = ConversationPolicy().normalize_input(content)
    if text == "":
        raise HTTPException(status_code=422, detail={"error": "message_content_required"})
    return text


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


def _stream_agent_error_line(error: AgentRuntimeError) -> str:
    return _stream_line(
        {
            "type": "error",
            "status": 503,
            "detail": {"error": error.code, "message": error.message},
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
