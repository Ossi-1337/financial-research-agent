from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

import httpx

from .contracts import McpErrorCode, McpResultEnvelope

DEFAULT_MCP_APP_BASE_URL = "http://127.0.0.1:8000"
MAX_MCP_MESSAGE_CHARS = 4000
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,160}$")
_TERMINAL_JOB_STATUSES = {"succeeded", "failed", "cancelled"}


class McpApplicationClient:
    def __init__(
        self,
        *,
        base_url: str = DEFAULT_MCP_APP_BASE_URL,
        timeout_seconds: float = 30,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = _local_base_url(base_url)
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.timeout_seconds = timeout_seconds
        self._transport = transport

    async def send_message(
        self,
        *,
        content: str,
        session_id: str | None = None,
    ) -> McpResultEnvelope:
        try:
            message = _bounded_text("content", content, MAX_MCP_MESSAGE_CHARS)
            resolved_session_id = _safe_id("session_id", session_id) if session_id else None
        except ValueError as exc:
            return _failure("send_message", McpErrorCode.INVALID_ARGUMENTS, str(exc))

        async with self._http_client() as client:
            if resolved_session_id is None:
                response = await self._request(
                    client,
                    "POST",
                    "/api/sessions",
                    capability_id="send_message",
                )
                if isinstance(response, McpResultEnvelope):
                    return response
                session = response.get("session")
                if not isinstance(session, Mapping):
                    return _malformed("send_message")
                try:
                    resolved_session_id = _safe_id("session_id", session.get("id"))
                except ValueError:
                    return _malformed("send_message")

            response = await self._request(
                client,
                "POST",
                f"/api/sessions/{resolved_session_id}/messages/stream",
                capability_id="send_message",
                json_payload={"content": message, "mentions": []},
                ndjson=True,
            )
        if isinstance(response, McpResultEnvelope):
            return response
        return _message_result(resolved_session_id, response)

    async def get_research_status(self, *, job_id: str) -> McpResultEnvelope:
        return await self._job_request("get_research_status", job_id)

    async def get_research_result(self, *, job_id: str) -> McpResultEnvelope:
        result = await self._job_request("get_research_result", job_id)
        if result.status.value == "failed":
            return result
        job = result.data.get("job")
        if not isinstance(job, Mapping):
            return _malformed("get_research_result")
        status = str(job.get("status", ""))
        if status not in _TERMINAL_JOB_STATUSES:
            return McpResultEnvelope.accepted(
                capability_id="get_research_result",
                data={"job": _bounded_job(job)},
                warnings=("Research is not complete yet.",),
            )
        if status != "succeeded":
            return _failure(
                "get_research_result",
                McpErrorCode.RESEARCH_FAILED,
                f"Research ended with status {status}.",
            )
        run_id = job.get("orchestrator_run_id")
        try:
            safe_run_id = _safe_id("run_id", run_id)
        except ValueError:
            return _malformed("get_research_result")
        async with self._http_client() as client:
            evidence = await self._request(
                client,
                "GET",
                f"/api/orchestrator/runs/{safe_run_id}/evidence",
                capability_id="get_research_result",
            )
        if isinstance(evidence, McpResultEnvelope):
            return evidence
        return McpResultEnvelope.succeeded(
            capability_id="get_research_result",
            data={
                "job": _bounded_job(job),
                "research_run": job.get("research_run"),
                "synthesis_report": job.get("synthesis_report"),
                "evidence": evidence.get("evidence"),
            },
        )

    async def cancel_research(self, *, job_id: str) -> McpResultEnvelope:
        try:
            safe_job_id = _safe_id("job_id", job_id)
        except ValueError as exc:
            return _failure("cancel_research", McpErrorCode.INVALID_ARGUMENTS, str(exc))
        async with self._http_client() as client:
            response = await self._request(
                client,
                "POST",
                f"/api/background/research-runs/{safe_job_id}/cancel",
                capability_id="cancel_research",
            )
        if isinstance(response, McpResultEnvelope):
            return response
        job = response.get("job")
        if not isinstance(job, Mapping):
            return _malformed("cancel_research")
        return McpResultEnvelope.succeeded(
            capability_id="cancel_research",
            data={"job": _bounded_job(job)},
        )

    async def _job_request(self, capability_id: str, job_id: str) -> McpResultEnvelope:
        try:
            safe_job_id = _safe_id("job_id", job_id)
        except ValueError as exc:
            return _failure(capability_id, McpErrorCode.INVALID_ARGUMENTS, str(exc))
        async with self._http_client() as client:
            response = await self._request(
                client,
                "GET",
                f"/api/background/research-runs/{safe_job_id}",
                capability_id=capability_id,
            )
        if isinstance(response, McpResultEnvelope):
            return response
        job = response.get("job")
        if not isinstance(job, Mapping):
            return _malformed(capability_id)
        status = str(job.get("status", ""))
        constructor = (
            McpResultEnvelope.succeeded
            if status in _TERMINAL_JOB_STATUSES
            else McpResultEnvelope.accepted
        )
        return constructor(
            capability_id=capability_id,
            data={
                "job": _bounded_job(
                    job,
                    include_result=capability_id == "get_research_result",
                )
            },
        )

    async def _request(
        self,
        client: httpx.AsyncClient,
        method: str,
        path: str,
        *,
        capability_id: str,
        json_payload: Mapping[str, Any] | None = None,
        ndjson: bool = False,
    ) -> dict[str, Any] | list[dict[str, Any]] | McpResultEnvelope:
        try:
            response = await client.request(method, path, json=json_payload)
        except httpx.TimeoutException, httpx.RequestError:
            return _failure(
                capability_id,
                McpErrorCode.APP_UNAVAILABLE,
                "The local Financial Research Agent application is unavailable.",
            )
        if response.status_code == 404:
            return _failure(
                capability_id,
                McpErrorCode.NOT_FOUND,
                "The requested local application resource was not found.",
            )
        if response.status_code == 409:
            return _failure(
                capability_id,
                McpErrorCode.CONFLICT,
                "The local application rejected the request because of its current state.",
            )
        if response.status_code >= 400:
            return _failure(
                capability_id,
                McpErrorCode.APP_UNAVAILABLE,
                "The local application could not complete the request.",
            )
        try:
            if ndjson:
                return [
                    parsed
                    for line in response.text.splitlines()
                    if line.strip() and isinstance((parsed := json.loads(line)), dict)
                ]
            payload = response.json()
        except json.JSONDecodeError, ValueError:
            return _malformed(capability_id)
        return payload if isinstance(payload, dict) else _malformed(capability_id)

    def _http_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout_seconds,
            transport=self._transport,
        )


def _message_result(
    session_id: str,
    events: list[dict[str, Any]],
) -> McpResultEnvelope:
    research = next((event for event in events if event.get("type") == "research"), None)
    if research is not None:
        job = research.get("job")
        if not isinstance(job, Mapping):
            return _malformed("send_message")
        return McpResultEnvelope.accepted(
            capability_id="send_message",
            data={"session_id": session_id, "job": _bounded_job(job)},
        )
    completed = next(
        (event for event in reversed(events) if event.get("type") == "completed"),
        None,
    )
    if completed is None:
        return _malformed("send_message")
    assistant = completed.get("assistant_message")
    if not isinstance(assistant, Mapping):
        return _malformed("send_message")
    return McpResultEnvelope.succeeded(
        capability_id="send_message",
        data={
            "session_id": session_id,
            "assistant_message": dict(assistant),
            "provider": completed.get("provider"),
            "model": completed.get("model"),
        },
    )


def _bounded_job(
    job: Mapping[str, Any],
    *,
    include_result: bool = False,
) -> dict[str, Any]:
    allowed = (
        "id",
        "query",
        "status",
        "created_at",
        "updated_at",
        "started_at",
        "completed_at",
        "orchestrator_run_id",
        "error_code",
        "error_message",
        "warnings",
        "progress",
    )
    payload = {key: job.get(key) for key in allowed if key in job}
    if include_result:
        run = job.get("orchestrator_run")
        payload["research_run"] = _bounded_run(run) if isinstance(run, Mapping) else None
        payload["synthesis_report"] = job.get("synthesis_report")
    return payload


def _bounded_run(run: Mapping[str, Any]) -> dict[str, Any]:
    allowed = (
        "id",
        "query",
        "status",
        "created_at",
        "updated_at",
        "specialist_roles",
        "agent_provider",
        "agent_model",
        "selected_company",
        "selected_security",
        "warnings",
        "limitations",
        "no_recommendation_notice",
    )
    return {key: run.get(key) for key in allowed if key in run}


def _local_base_url(value: str) -> str:
    text = _bounded_text("base_url", value, 500).rstrip("/")
    parsed = urlsplit(text)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("MCP application base URL must be a local HTTP URL")
    return text


def _safe_id(name: str, value: object) -> str:
    text = str(value or "").strip()
    if _SAFE_ID.fullmatch(text) is None:
        raise ValueError(f"{name} is invalid")
    return text


def _bounded_text(name: str, value: object, maximum: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    text = value.strip()
    if not text:
        raise ValueError(f"{name} is required")
    if len(text) > maximum:
        raise ValueError(f"{name} must be at most {maximum} characters")
    return text


def _malformed(capability_id: str) -> McpResultEnvelope:
    return _failure(
        capability_id,
        McpErrorCode.MALFORMED_RESPONSE,
        "The local application returned an invalid response.",
    )


def _failure(
    capability_id: str,
    code: McpErrorCode,
    message: str,
) -> McpResultEnvelope:
    return McpResultEnvelope.failed(
        capability_id=capability_id,
        error_code=code,
        message=message,
    )
