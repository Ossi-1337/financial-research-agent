from __future__ import annotations

import asyncio
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import replace
from datetime import UTC, datetime
from uuid import NAMESPACE_URL, uuid5

import httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.client.errors import A2AClientError
from a2a.types import AgentCard, SendMessageRequest, Task, TaskState
from a2a.utils.constants import TransportProtocol
from google.protobuf.json_format import MessageToDict, ParseDict

from financial_research_agent.a2a.delegations import (
    SQLiteA2ADelegationStore,
    delegation_record,
)
from financial_research_agent.orchestration import (
    AgentEndpoint,
    AgentExecutionMetadata,
    AgentExecutionMode,
    AgentHandoff,
    AgentRole,
    DelegationRequest,
    DelegationResult,
    HandoffConfidence,
    OrchestratedResearchRun,
    OrchestratorHandoffStatus,
    handoff_from_dict,
)


class A2AResearchStepDispatcher:
    def __init__(
        self,
        *,
        endpoints: Mapping[AgentRole, AgentEndpoint],
        timeout_seconds: float = 60.0,
        max_attempts: int = 2,
        api_key: str | None = None,
        delegation_store: SQLiteA2ADelegationStore | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        self.endpoints = dict(endpoints)
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self.api_key = api_key
        self.delegation_store = delegation_store
        self.transport = transport
        self._active: dict[str, tuple[object, str]] = {}
        self._active_lock = asyncio.Lock()

    async def dispatch(
        self,
        request: DelegationRequest,
        *,
        run: OrchestratedResearchRun | None = None,
    ) -> DelegationResult:
        del run
        endpoint = self.endpoints.get(request.role)
        if endpoint is None:
            return self._failure(request, "a2a_agent_unconfigured", 1)
        last_code = "a2a_agent_unavailable"
        for attempt in range(1, self.max_attempts + 1):
            try:
                handoff, task_id = await self._send(endpoint, request)
                if handoff.kind != request.expected_kind or handoff.step_id != request.step_id:
                    raise ValueError("specialist artifact does not match requested step")
                metadata = AgentExecutionMetadata(
                    mode=AgentExecutionMode.A2A,
                    agent_role=request.role.value,
                    correlation_id=request.correlation_id,
                    delegation_id=_delegation_id(request),
                    remote_task_id=task_id,
                    service_id=endpoint.service_id,
                    attempt_count=attempt,
                    prompt_id=(
                        handoff.execution.prompt_id if handoff.execution is not None else None
                    ),
                    prompt_version=(
                        handoff.execution.prompt_version if handoff.execution is not None else None
                    ),
                    provider=handoff.execution.provider if handoff.execution is not None else None,
                    model=handoff.execution.model if handoff.execution is not None else None,
                    tool_status=(
                        handoff.execution.tool_status if handoff.execution is not None else None
                    ),
                    reasoning_summary=(
                        handoff.execution.reasoning_summary
                        if handoff.execution is not None
                        else None
                    ),
                )
                result = DelegationResult(
                    handoff=replace(handoff, execution=metadata),
                    remote_task_id=task_id,
                    attempt_count=attempt,
                )
                await self._persist(request, endpoint, result)
                return result
            except asyncio.CancelledError:
                await self.cancel_correlation(request.correlation_id)
                raise
            except TimeoutError:
                last_code = "a2a_agent_timeout"
            except httpx.HTTPStatusError as exc:
                last_code = (
                    "a2a_agent_unavailable"
                    if exc.response.status_code >= 500
                    else "a2a_agent_invalid_response"
                )
                if exc.response.status_code < 500:
                    break
            except httpx.HTTPError:
                last_code = "a2a_agent_unavailable"
            except A2AClientError as exc:
                last_code, retryable = _classify_a2a_client_error(exc)
                if not retryable:
                    break
            except KeyError, TypeError, ValueError:
                last_code = "a2a_agent_malformed_artifact"
                break
        result = self._failure(request, last_code, self.max_attempts)
        await self._persist(request, endpoint, result)
        return result

    async def cancel_correlation(self, correlation_id: str) -> None:
        async with self._active_lock:
            active = tuple(
                (delegation_id, client, task_id)
                for delegation_id, (client, task_id) in self._active.items()
                if delegation_id.startswith(f"{correlation_id}:")
            )
        for delegation_id, client, task_id in active:
            try:
                await client.cancel_task(ParseDict({"id": task_id}, _cancel_request()))
            except Exception:
                pass
            finally:
                async with self._active_lock:
                    self._active.pop(delegation_id, None)

    async def _send(
        self,
        endpoint: AgentEndpoint,
        request: DelegationRequest,
    ) -> tuple[AgentHandoff, str]:
        headers = {
            "A2A-Version": "1.0",
            "X-FRA-Correlation-ID": request.correlation_id,
            "X-FRA-Delegation-ID": _delegation_id(request),
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        timeout = httpx.Timeout(self.timeout_seconds)
        async with httpx.AsyncClient(
            headers=headers,
            timeout=timeout,
            transport=self.transport,
        ) as http_client:
            card = await A2ACardResolver(http_client, endpoint.base_url).get_agent_card()
            _validate_card(card, endpoint)
            client = ClientFactory(
                ClientConfig(
                    streaming=False,
                    polling=False,
                    httpx_client=http_client,
                    supported_protocol_bindings=[TransportProtocol.HTTP_JSON.value],
                    accepted_output_modes=["application/json"],
                )
            ).create(card)
            task: Task | None = None
            send_request = ParseDict(
                {
                    "message": {
                        "messageId": _message_id(request),
                        "role": "ROLE_USER",
                        "parts": [
                            {
                                "data": request.to_dict(),
                                "mediaType": "application/json",
                            }
                        ],
                    }
                },
                SendMessageRequest(),
            )
            try:
                async with asyncio.timeout(self.timeout_seconds):
                    async for response in client.send_message(send_request):
                        if response.HasField("task"):
                            task = response.task
                            if task.id:
                                async with self._active_lock:
                                    self._active[_active_key(request)] = (client, task.id)
            except asyncio.CancelledError:
                if task is not None and task.id:
                    with suppress(Exception):
                        await client.cancel_task(ParseDict({"id": task.id}, _cancel_request()))
                raise
            finally:
                if task is not None:
                    async with self._active_lock:
                        self._active.pop(_active_key(request), None)
                await client.close()
            if task is None:
                raise ValueError("specialist returned no task")
            if task.status.state != TaskState.TASK_STATE_COMPLETED:
                raise ValueError("specialist task did not complete")
            return _handoff_from_task(task), task.id

    def _failure(
        self,
        request: DelegationRequest,
        error_code: str,
        attempt_count: int,
    ) -> DelegationResult:
        now = datetime.now(UTC)
        endpoint = self.endpoints.get(request.role)
        handoff = AgentHandoff(
            id=(
                f"handoff_{request.expected_kind.value}_"
                f"{uuid5(NAMESPACE_URL, _active_key(request)).hex}"
            ),
            step_id=request.step_id,
            kind=request.expected_kind,
            status=OrchestratorHandoffStatus.FAILED,
            started_at=now,
            completed_at=now,
            limitations=(f"{request.role.value} A2A service was unavailable.",),
            confidence=HandoffConfidence.UNKNOWN,
            error_code=error_code,
            error_message="Specialist delegation failed safely.",
            execution=AgentExecutionMetadata(
                mode=AgentExecutionMode.A2A,
                agent_role=request.role.value,
                correlation_id=request.correlation_id,
                delegation_id=_delegation_id(request),
                service_id=endpoint.service_id if endpoint else None,
                attempt_count=attempt_count,
            ),
        )
        return DelegationResult(handoff=handoff, attempt_count=attempt_count)

    async def _persist(
        self,
        request: DelegationRequest,
        endpoint: AgentEndpoint,
        result: DelegationResult,
    ) -> None:
        if self.delegation_store is None:
            return
        await self.delegation_store.save(
            delegation_record(
                delegation_id=_delegation_id(request),
                run_id=request.run_id,
                correlation_id=request.correlation_id,
                role=request.role,
                service_id=endpoint.service_id,
                status=result.handoff.status.value,
                attempt_count=result.attempt_count,
                remote_task_id=result.remote_task_id,
                error_code=result.handoff.error_code,
            )
        )


def _validate_card(card: AgentCard, endpoint: AgentEndpoint) -> None:
    skill_ids = {skill.id for skill in card.skills}
    if endpoint.skill_id not in skill_ids:
        raise ValueError("specialist Agent Card does not advertise required skill")
    if not any(
        interface.protocol_binding == TransportProtocol.HTTP_JSON.value
        and interface.protocol_version == "1.0"
        for interface in card.supported_interfaces
    ):
        raise ValueError("specialist Agent Card does not support A2A 1.0 HTTP+JSON")


def _handoff_from_task(task: Task) -> AgentHandoff:
    payload = MessageToDict(task)
    for artifact in reversed(payload.get("artifacts", [])):
        if artifact.get("metadata", {}).get("kind") != "specialist_handoff":
            continue
        for part in artifact.get("parts", []):
            data = part.get("data")
            if isinstance(data, dict):
                return handoff_from_dict(data)
    raise ValueError("specialist handoff artifact is missing")


def _delegation_id(request: DelegationRequest) -> str:
    return f"delegation_{uuid5(NAMESPACE_URL, _active_key(request)).hex}"


def _message_id(request: DelegationRequest) -> str:
    return f"message_{uuid5(NAMESPACE_URL, _active_key(request)).hex}"


def _active_key(request: DelegationRequest) -> str:
    return f"{request.correlation_id}:{request.run_id}:{request.step_id}"


def _cancel_request():
    from a2a.types import CancelTaskRequest

    return CancelTaskRequest()


def _classify_a2a_client_error(error: A2AClientError) -> tuple[str, bool]:
    cause: BaseException | None = error
    while cause is not None:
        if isinstance(cause, httpx.HTTPStatusError):
            if cause.response.status_code >= 500:
                return "a2a_agent_unavailable", True
            return "a2a_agent_invalid_response", False
        if isinstance(cause, httpx.TimeoutException):
            return "a2a_agent_timeout", True
        if isinstance(cause, httpx.NetworkError):
            return "a2a_agent_unavailable", True
        cause = cause.__cause__
    return "a2a_agent_invalid_response", False
