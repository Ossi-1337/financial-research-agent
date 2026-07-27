from __future__ import annotations

import hashlib
import json
import secrets
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid5

from a2a.server.agent_execution import (
    RequestContext,
    RequestContextBuilder,
)
from a2a.server.context import ServerCallContext
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import (
    add_a2a_routes_to_fastapi,
    create_agent_card_routes,
    create_rest_routes,
)
from a2a.types import (
    AgentCard,
    Message,
    Part,
    Role,
    SendMessageRequest,
    SendMessageResponse,
    StreamResponse,
    Task,
    TaskState,
)
from a2a.utils.constants import PROTOCOL_VERSION_1_0, TransportProtocol
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from google.protobuf.json_format import MessageToDict, ParseDict
from google.protobuf.timestamp_pb2 import Timestamp

from financial_research_agent import __version__
from financial_research_agent.a2a.runtime import (
    A2AResearchRuntime,
    create_default_a2a_runtime,
)
from financial_research_agent.a2a.specialist_executor import SpecialistAgentExecutor
from financial_research_agent.observability import RedactionPolicy
from financial_research_agent.orchestration import AgentRole
from financial_research_agent.settings import Settings

_ID_NAMESPACE = UUID("911036e9-e57d-496c-b428-e5c29f5e219f")
_ALLOWED_REST_ROUTES = {
    ("/message:send", "POST"),
    ("/message:stream", "POST"),
    ("/tasks/{id}", "GET"),
    ("/tasks", "GET"),
    ("/tasks/{id}:subscribe", "POST"),
}
_TERMINAL_TASK_STATES = {
    TaskState.TASK_STATE_COMPLETED,
    TaskState.TASK_STATE_CANCELED,
    TaskState.TASK_STATE_FAILED,
    TaskState.TASK_STATE_REJECTED,
}


class DeterministicRequestContextBuilder(RequestContextBuilder):
    async def build(
        self,
        context: ServerCallContext,
        params: SendMessageRequest | None = None,
        task_id: str | None = None,
        context_id: str | None = None,
        task: Task | None = None,
    ) -> RequestContext:
        original_task_id = task_id
        context.state["fra_original_task_id"] = original_task_id
        if params is not None and task_id is None:
            message_id = params.message.message_id
            if message_id:
                task_id = f"a2a_task_{uuid5(_ID_NAMESPACE, message_id).hex}"
                if context_id is None:
                    context_id = f"a2a_context_{uuid5(_ID_NAMESPACE, f'context:{message_id}').hex}"
        return RequestContext(
            call_context=context,
            request=params,
            task_id=task_id,
            context_id=context_id,
            task=task,
        )


def create_agent_card(
    settings: Settings,
    role: AgentRole,
) -> AgentCard:
    profile = _agent_card_profile(role)
    payload: dict[str, Any] = {
        "name": profile["name"],
        "description": profile["description"],
        "version": __version__,
        "supportedInterfaces": [
            {
                "url": settings.a2a.public_base_url,
                "protocolBinding": TransportProtocol.HTTP_JSON.value,
                "protocolVersion": PROTOCOL_VERSION_1_0,
            }
        ],
        "capabilities": {
            "streaming": True,
            "pushNotifications": False,
            "extendedAgentCard": False,
        },
        "defaultInputModes": [profile["input_mode"]],
        "defaultOutputModes": ["text/plain", "application/json"],
        "skills": [profile["skill"]],
    }
    if not settings.a2a.local_only:
        payload["securitySchemes"] = {
            "bearer": {
                "httpAuthSecurityScheme": {
                    "description": "Bearer key configured out of band.",
                    "scheme": "bearer",
                    "bearerFormat": "API key",
                }
            }
        }
        payload["securityRequirements"] = [{"schemes": {"bearer": {"list": []}}}]
    return ParseDict(payload, AgentCard())


def create_a2a_app(
    *,
    settings: Settings | None = None,
    runtime: A2AResearchRuntime | None = None,
    role: AgentRole | None = None,
) -> FastAPI:
    app_settings = settings or Settings.from_env()
    if not app_settings.a2a.enabled:
        raise ValueError("A2A server is disabled. Set FRA_A2A_ENABLED=true.")
    server_role = role or (runtime.role if runtime is not None else None)
    if server_role is None:
        raise ValueError("specialist A2A role is required")
    research_runtime = runtime or create_default_a2a_runtime(app_settings, role=server_role)
    card = create_agent_card(app_settings, server_role)
    redaction_policy = RedactionPolicy.from_settings(app_settings)
    executor = SpecialistAgentExecutor(
        role=server_role,
        service=research_runtime.specialist_service,
        task_store=research_runtime.task_store,
        redaction_policy=redaction_policy,
    )
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=research_runtime.task_store,
        agent_card=card,
        request_context_builder=DeterministicRequestContextBuilder(),
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await research_runtime.task_store.reconcile_restarted_tasks()
        yield
        await handler.aclose()

    app = FastAPI(
        title="Financial Research Agent A2A",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.settings = app_settings
    app.state.a2a_runtime = research_runtime
    app.state.a2a_handler = handler
    app.state.agent_card = card

    @app.post("/tasks/{task_id}:cancel")
    async def cancel_task(task_id: str) -> Response:
        return await _cancel_persisted_task(task_id, research_runtime)

    card_routes = create_agent_card_routes(card)
    rest_routes = [
        route for route in create_rest_routes(handler) if _route_key(route) in _ALLOWED_REST_ROUTES
    ]
    add_a2a_routes_to_fastapi(
        app,
        agent_card_routes=card_routes,
        rest_routes=rest_routes,
    )
    _install_security_middleware(app, app_settings, research_runtime, card)
    return app


def _agent_card_profile(role: AgentRole) -> dict[str, Any]:
    profiles = {
        AgentRole.FINANCIAL_REPORT: (
            "financial-report-agent",
            "financial_report_analysis",
            "Analyze stored financial statements and filing evidence.",
        ),
        AgentRole.STOCK: (
            "stock-analysis-agent",
            "stock_price_analysis",
            "Analyze stored market and benchmark data deterministically.",
        ),
        AgentRole.CONTEXT: (
            "context-analysis-agent",
            "context_analysis",
            "Analyze bounded source-linked company, macro, and sector context.",
        ),
        AgentRole.SYNTHESIS: (
            "synthesis-agent",
            "research_synthesis",
            "Build deterministic synthesis from persisted specialist handoffs.",
        ),
    }
    name, skill_id, description = profiles[role]
    return {
        "name": name,
        "description": description,
        "input_mode": "application/json",
        "skill": {
            "id": skill_id,
            "name": skill_id.replace("_", " ").title(),
            "description": description,
            "tags": ["finance", "specialist", "source-backed", "local-first"],
            "examples": [],
            "inputModes": ["application/json"],
            "outputModes": ["application/json"],
        },
    }


def _install_security_middleware(
    app: FastAPI,
    settings: Settings,
    runtime: A2AResearchRuntime,
    card: AgentCard,
) -> None:
    card_bytes = card.SerializeToString(deterministic=True)
    card_etag = f'"{hashlib.sha256(card_bytes).hexdigest()}"'
    max_request_bytes = settings.a2a.max_input_chars * 2 + 10_000

    @app.middleware("http")
    async def a2a_security(request: Request, call_next):
        if request.url.path == "/.well-known/agent-card.json":
            if request.headers.get("if-none-match") == card_etag:
                return Response(status_code=304, headers={"ETag": card_etag})
            response = await call_next(request)
            response.headers["Cache-Control"] = "public, max-age=300"
            response.headers["ETag"] = card_etag
            return response

        if not settings.a2a.local_only and not _valid_bearer(
            request.headers.get("authorization"),
            settings.a2a.api_key,
        ):
            return JSONResponse(
                status_code=401,
                content={"error": "a2a_authentication_required"},
                headers={"WWW-Authenticate": "Bearer"},
            )
        content_length = request.headers.get("content-length")
        try:
            request_size = int(content_length) if content_length else 0
        except ValueError:
            return JSONResponse(
                status_code=400,
                content={"error": "invalid_content_length"},
            )
        if request_size > max_request_bytes:
            return JSONResponse(
                status_code=413,
                content={"error": "a2a_request_too_large"},
            )
        if request.method == "POST" and request.url.path in {
            "/message:send",
            "/message:stream",
        }:
            duplicate = await _idempotent_response(request, runtime)
            if duplicate is not None:
                return duplicate
        if request.method == "POST" and request.url.path.endswith(":subscribe"):
            snapshot = await _terminal_subscription_response(request, runtime)
            if snapshot is not None:
                return snapshot
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        return response


async def _idempotent_response(
    request: Request,
    runtime: A2AResearchRuntime,
) -> Response | None:
    try:
        payload = await request.json()
    except json.JSONDecodeError, UnicodeDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    message = payload.get("message")
    message_id = message.get("messageId") if isinstance(message, dict) else None
    if not isinstance(message_id, str) or not message_id:
        return None
    existing = await runtime.task_store.find_by_message_id(message_id)
    if existing is None:
        return None
    if request.url.path == "/message:stream":
        event = json.dumps(
            MessageToDict(StreamResponse(task=existing)),
            separators=(",", ":"),
        )
        return StreamingResponse(
            iter((f"data: {event}\n\n",)),
            media_type="text/event-stream",
            headers={"A2A-Version": PROTOCOL_VERSION_1_0},
        )
    return JSONResponse(
        MessageToDict(SendMessageResponse(task=existing)),
        headers={"A2A-Version": PROTOCOL_VERSION_1_0},
    )


async def _terminal_subscription_response(
    request: Request,
    runtime: A2AResearchRuntime,
) -> Response | None:
    task_id = request.url.path.removeprefix("/tasks/").removesuffix(":subscribe")
    task = await runtime.task_store.get(task_id, ServerCallContext(state={}))
    if task is None or task.status.state not in _TERMINAL_TASK_STATES:
        return None
    event = json.dumps(
        MessageToDict(StreamResponse(task=task)),
        separators=(",", ":"),
    )
    return StreamingResponse(
        iter((f"data: {event}\n\n",)),
        media_type="text/event-stream",
        headers={"A2A-Version": PROTOCOL_VERSION_1_0},
    )


async def _cancel_persisted_task(
    task_id: str,
    runtime: A2AResearchRuntime,
) -> Response:
    context = ServerCallContext(state={})
    task = await runtime.task_store.get(task_id, context)
    if task is None:
        return JSONResponse(status_code=404, content={"error": "a2a_task_not_found"})
    if task.status.state in _TERMINAL_TASK_STATES:
        return JSONResponse(status_code=409, content={"error": "a2a_task_not_cancelable"})
    timestamp = Timestamp()
    timestamp.FromDatetime(datetime.now(UTC))
    task.status.state = TaskState.TASK_STATE_CANCELED
    task.status.timestamp.CopyFrom(timestamp)
    task.status.message.CopyFrom(
        Message(
            message_id=f"cancel-{task.id}",
            task_id=task.id,
            context_id=task.context_id,
            role=Role.ROLE_AGENT,
            parts=[Part(text="Research task cancelled.")],
        )
    )
    await runtime.task_store.save(task, context)
    await runtime.task_store.append_event(
        task_id,
        "cancelled",
        {"error_code": "cancelled"},
    )
    return JSONResponse(
        MessageToDict(task),
        headers={"A2A-Version": PROTOCOL_VERSION_1_0},
    )


def _route_key(route: object) -> tuple[str, str] | None:
    path = getattr(route, "path", None)
    methods = getattr(route, "methods", None)
    if not isinstance(path, str) or not methods:
        return None
    normalized = set(methods)
    if normalized == {"GET", "HEAD"}:
        return path, "GET"
    if len(normalized) == 1:
        return path, next(iter(normalized))
    return None


def _valid_bearer(header: str | None, expected: str | None) -> bool:
    if expected is None or header is None:
        return False
    scheme, separator, value = header.partition(" ")
    return (
        bool(separator)
        and scheme.lower() == "bearer"
        and secrets.compare_digest(value.strip(), expected)
    )
