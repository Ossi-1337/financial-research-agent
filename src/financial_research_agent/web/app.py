from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

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
from financial_research_agent.llm import (
    ChatMessage,
    ChatRequest,
    MessageRole,
    ProviderError,
    ProviderErrorCode,
    StreamEventType,
)
from financial_research_agent.llm.registry import ProviderRegistry, create_default_provider_registry
from financial_research_agent.market_data import (
    MarketDataError,
    MarketDataErrorCode,
    MarketDataProvider,
    MarketDataStore,
    MarketSecurity,
    create_default_market_data_provider,
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
from financial_research_agent.settings import ProviderTask, Settings
from financial_research_agent.statements import (
    FinancialStatementCompany,
    FinancialStatementError,
    FinancialStatementErrorCode,
    FinancialStatementProvider,
    FinancialStatementStore,
    create_default_financial_statement_provider,
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
    app.state.retrieval_index = retrieval
    app.state.report_run_store = report_runs

    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        selection = app_settings.provider.selection_for_task(ProviderTask.CHAT)
        embedding_selection = app_settings.provider.selection_for_task(ProviderTask.EMBEDDINGS)
        return {
            "app": "financial-research-agent",
            "environment": app_settings.environment,
            "status": "ok",
            "chat": {
                "provider": selection.provider,
                "model": selection.model,
                "base_url": selection.base_url,
                "registered": provider_registry.has_chat_provider(selection.provider),
            },
            "history": {
                "recent_turns": sessions.recent_turns,
                "summary_max_chars": sessions.summary_max_chars,
                "session_count": sessions.count(),
                "persistent": sessions.storage_path is not None,
            },
            "company_search": {
                "provider": app_settings.data_sources.company_lookup_provider,
                "cache_ttl_days": app_settings.data_sources.company_lookup_cache_ttl_days,
            },
            "market_data": {
                "provider": app_settings.data_sources.market_data_provider,
                "cache_ttl_days": app_settings.data_sources.market_data_cache_ttl_days,
                "alpha_vantage_api_key_configured": (
                    app_settings.data_sources.alpha_vantage_api_key is not None
                ),
                "stored_series_count": market_store.count(),
            },
            "financial_statements": {
                "provider": app_settings.data_sources.financial_statement_provider,
                "cache_ttl_days": (app_settings.data_sources.financial_statement_cache_ttl_days),
                "stored_result_count": statement_store.count(),
            },
            "filings": {
                "provider": app_settings.data_sources.filing_provider,
                "cache_ttl_days": app_settings.data_sources.filing_cache_ttl_days,
                "max_document_bytes": app_settings.data_sources.filing_max_document_bytes,
                "stored_result_count": filings.count(),
            },
            "retrieval": {
                "provider": app_settings.retrieval.provider,
                "top_k": app_settings.retrieval.top_k,
                "min_score": app_settings.retrieval.min_score,
                "index": retrieval.metadata().to_dict(),
                "embedding_provider": embedding_selection.provider,
                "embedding_model": embedding_selection.model,
                "embedding_provider_registered": provider_registry.has_embedding_provider(
                    embedding_selection.provider
                ),
            },
            "report_runs": {
                "stored_run_count": report_runs.count(),
                "persistent": report_runs.storage_path is not None,
            },
            "storage": {
                "provider": app_settings.storage.provider,
                "app_home": str(app_settings.local_paths.app_home),
                "dataset_count": len(storage.dataset_specs),
            },
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
        selection = app_settings.provider.selection_for_task(ProviderTask.EMBEDDINGS)
        try:
            embedding_provider = provider_registry.embedding_provider(selection.provider)
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
        selection = app_settings.provider.selection_for_task(ProviderTask.EMBEDDINGS)
        query = RetrievalQuery(
            query=_message_content(request.query),
            top_k=request.top_k or app_settings.retrieval.top_k,
            min_score=(
                request.min_score
                if request.min_score is not None
                else app_settings.retrieval.min_score
            ),
            filters=request.filters,
        )
        try:
            embedding_provider = provider_registry.embedding_provider(selection.provider)
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
        retrieval_query = RetrievalQuery(
            query=content,
            top_k=request.top_k or app_settings.retrieval.top_k,
            min_score=(
                request.min_score
                if request.min_score is not None
                else app_settings.retrieval.min_score
            ),
            filters=request.filters,
        )
        try:
            embedding_selection = app_settings.provider.selection_for_task(ProviderTask.EMBEDDINGS)
            embedding_provider = provider_registry.embedding_provider(embedding_selection.provider)
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

        chat_selection = app_settings.provider.selection_for_task(ProviderTask.CHAT)
        try:
            provider = provider_registry.chat_provider(chat_selection.provider)
            response = await provider.chat(
                ChatRequest(
                    messages=build_rag_messages(content, evidence),
                    model=chat_selection.model,
                )
            )
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
        }

    @app.post("/api/sessions/{session_id}/messages")
    async def post_message(session_id: str, request: MessageRequest) -> dict[str, Any]:
        session = sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail={"error": "session_not_found"})
        content = _message_content(request.content)
        mentions = _chat_mentions(request.mentions)
        selection = app_settings.provider.selection_for_task(ProviderTask.CHAT)
        try:
            provider = provider_registry.chat_provider(selection.provider)
            response = await provider.chat(
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
                )
            )
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
        selection = app_settings.provider.selection_for_task(ProviderTask.STREAMING)
        try:
            provider = provider_registry.chat_provider(selection.provider)
        except ProviderError as exc:
            raise HTTPException(
                status_code=_status_for_provider_error(exc.code),
                detail=_provider_error_detail(exc),
            ) from exc
        chat_request = ChatRequest(
            messages=_request_messages(
                session.context_messages(
                    recent_turns=sessions.recent_turns,
                    summary_max_chars=sessions.summary_max_chars,
                ),
                content,
                mentions,
            ),
            model=selection.model,
        )

        async def events() -> AsyncIterator[str]:
            collected_content: list[str] = []
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
