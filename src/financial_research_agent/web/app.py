from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from financial_research_agent.llm import (
    ChatMessage,
    ChatRequest,
    MessageRole,
    ProviderError,
    ProviderErrorCode,
)
from financial_research_agent.llm.registry import ProviderRegistry, create_default_provider_registry
from financial_research_agent.settings import ProviderTask, Settings
from financial_research_agent.web.sessions import ChatSessionStore

SYSTEM_PROMPT = (
    "You are a local financial research chat assistant. This milestone has no live financial "
    "data ingestion, database, RAG, or agent orchestration. Do not invent identifiers, prices, "
    "financial facts, source URLs, or citations. If the user asks for current company or market "
    "facts, explain that real data tools are not connected yet. Do not provide buy, sell, or "
    "hold recommendations."
)


class MessageRequest(BaseModel):
    content: str = Field(min_length=1, max_length=20_000)


def create_app(
    *,
    settings: Settings | None = None,
    registry: ProviderRegistry | None = None,
    session_store: ChatSessionStore | None = None,
) -> FastAPI:
    app_settings = settings or Settings.from_env()
    provider_registry = registry or create_default_provider_registry(app_settings.provider)
    sessions = session_store or ChatSessionStore.from_settings(app_settings)
    static_dir = Path(__file__).with_name("static")

    app = FastAPI(title="Financial Research Agent", version="0.1.0")
    app.state.settings = app_settings
    app.state.provider_registry = provider_registry
    app.state.sessions = sessions

    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        selection = app_settings.provider.selection_for_task(ProviderTask.CHAT)
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
        }

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

    @app.post("/api/sessions/{session_id}/messages")
    async def post_message(session_id: str, request: MessageRequest) -> dict[str, Any]:
        session = sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail={"error": "session_not_found"})
        content = _message_content(request.content)
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

    return app


def _message_content(content: str) -> str:
    text = content.strip()
    if text == "":
        raise HTTPException(status_code=422, detail={"error": "message_content_required"})
    return text


def _request_messages(
    history: tuple[ChatMessage, ...],
    user_content: str,
) -> tuple[ChatMessage, ...]:
    return (
        ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
        *history,
        ChatMessage(role=MessageRole.USER, content=user_content),
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
