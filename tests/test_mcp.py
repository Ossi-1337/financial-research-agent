from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import httpx
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from financial_research_agent.mcp import (
    McpApplicationClient,
    McpErrorCode,
    McpResultEnvelope,
    McpResultStatus,
)
from financial_research_agent.settings import Settings
from financial_research_agent.web import create_app

ROOT = Path(__file__).resolve().parents[1]
TOOL_NAMES = (
    "send_message",
    "get_research_status",
    "get_research_result",
    "cancel_research",
)


def test_mcp_envelope_is_immutable_and_versioned() -> None:
    result = McpResultEnvelope.succeeded(
        capability_id="send_message",
        data={"items": ("one", "two")},
    )

    assert result.to_dict()["schema_version"] == 1
    assert result.to_dict()["data"] == {"items": ["one", "two"]}
    with pytest.raises(FrozenInstanceError):
        result.status = McpResultStatus.FAILED  # type: ignore[misc]
    with pytest.raises(TypeError):
        result.data["new"] = "value"  # type: ignore[index]


def test_mcp_client_sends_direct_message_through_application() -> None:
    requests: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, request.url.path))
        if request.url.path == "/api/sessions":
            return httpx.Response(200, json={"session": {"id": "session_test"}})
        return httpx.Response(
            200,
            text=(
                '{"type":"delta","delta":"Hello"}\n'
                '{"type":"completed","assistant_message":{"role":"assistant",'
                '"content":"Hello"},"provider":"local-openai","model":"test-model"}\n'
            ),
        )

    result = asyncio.run(_client(handler).send_message(content="Hello"))

    assert result.status == McpResultStatus.SUCCEEDED
    assert result.data["session_id"] == "session_test"
    assert result.data["assistant_message"]["content"] == "Hello"
    assert requests == [
        ("POST", "/api/sessions"),
        ("POST", "/api/sessions/session_test/messages/stream"),
    ]


def test_mcp_client_returns_research_job_from_canonical_message_flow() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/sessions/session_existing/messages/stream"
        return httpx.Response(
            200,
            text=(
                '{"type":"research","job":{"id":"job_test","status":"queued",'
                '"orchestrator_run_id":"run_test","progress":{"completed_steps":0}}}\n'
            ),
        )

    result = asyncio.run(
        _client(handler).send_message(
            content="Research Tesla",
            session_id="session_existing",
        )
    )

    assert result.status == McpResultStatus.ACCEPTED
    assert result.data["job"]["id"] == "job_test"
    assert result.data["job"]["orchestrator_run_id"] == "run_test"


def test_mcp_client_uses_real_application_conversation_service(tmp_path: Path) -> None:
    settings = Settings.from_env(
        {
            "FRA_HOME": str(tmp_path),
            "FRA_STORAGE_PROVIDER": "local-json",
            "FRA_LLM_PROVIDER": "offline-test",
            "FRA_LLM_MODEL": "offline-test",
        }
    )
    app = create_app(settings=settings)
    client = McpApplicationClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://127.0.0.1:8000",
    )

    result = asyncio.run(client.send_message(content="Hello from MCP"))

    assert result.status == McpResultStatus.SUCCEEDED
    assert result.data["provider"] == "offline-test"
    assert result.data["assistant_message"]["content"] == ("offline-test response: Hello from MCP")


def test_mcp_client_gets_status_result_and_evidence() -> None:
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path.endswith("/evidence"):
            return httpx.Response(
                200,
                json={"evidence": {"sources": [{"evidence_id": "evidence:one"}]}},
            )
        return httpx.Response(
            200,
            json={
                "job": {
                    "id": "job_test",
                    "status": "succeeded",
                    "orchestrator_run_id": "run_test",
                    "orchestrator_run": {"id": "run_test", "status": "complete"},
                    "synthesis_report": {"id": "report_test"},
                }
            },
        )

    client = _client(handler)
    status = asyncio.run(client.get_research_status(job_id="job_test"))
    result = asyncio.run(client.get_research_result(job_id="job_test"))

    assert status.status == McpResultStatus.SUCCEEDED
    assert result.status == McpResultStatus.SUCCEEDED
    assert result.data["synthesis_report"]["id"] == "report_test"
    assert result.data["evidence"]["sources"][0]["evidence_id"] == "evidence:one"
    assert "handoffs" not in result.data["research_run"]
    assert paths[-1] == "/api/orchestrator/runs/run_test/evidence"


def test_mcp_client_returns_pending_result_without_evidence_request() -> None:
    requests = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(
            200,
            json={
                "job": {
                    "id": "job_test",
                    "status": "running",
                    "orchestrator_run_id": "run_test",
                }
            },
        )

    result = asyncio.run(_client(handler).get_research_result(job_id="job_test"))

    assert result.status == McpResultStatus.ACCEPTED
    assert result.warnings == ("Research is not complete yet.",)
    assert requests == 1


def test_mcp_client_cancels_through_application() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/background/research-runs/job_test/cancel"
        return httpx.Response(
            200,
            json={"job": {"id": "job_test", "status": "cancelled"}},
        )

    result = asyncio.run(_client(handler).cancel_research(job_id="job_test"))

    assert result.status == McpResultStatus.SUCCEEDED
    assert result.data["job"]["status"] == "cancelled"


def test_mcp_client_rejects_untrusted_url_and_invalid_input() -> None:
    with pytest.raises(ValueError, match="local HTTP URL"):
        McpApplicationClient(base_url="https://example.com")

    client = _client(lambda _request: httpx.Response(500))
    invalid_message = asyncio.run(client.send_message(content="x" * 4001))
    invalid_job = asyncio.run(client.get_research_status(job_id="../secret"))

    assert invalid_message.error_code == McpErrorCode.INVALID_ARGUMENTS
    assert invalid_job.error_code == McpErrorCode.INVALID_ARGUMENTS


def test_mcp_client_maps_unavailable_and_malformed_responses_safely() -> None:
    def unavailable(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("secret upstream detail")

    unavailable_result = asyncio.run(_client(unavailable).get_research_status(job_id="job_test"))
    malformed_result = asyncio.run(
        _client(lambda _request: httpx.Response(200, text="not-json")).get_research_status(
            job_id="job_test"
        )
    )

    assert unavailable_result.error_code == McpErrorCode.APP_UNAVAILABLE
    assert "secret upstream detail" not in unavailable_result.message
    assert malformed_result.error_code == McpErrorCode.MALFORMED_RESPONSE


def test_official_mcp_stdio_handshake_exposes_only_application_tools() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")

    async def run_client() -> None:
        parameters = StdioServerParameters(
            command=sys.executable,
            args=["-m", "financial_research_agent", "mcp-serve"],
            cwd=str(ROOT),
            env=env,
        )
        async with (
            stdio_client(parameters) as (read_stream, write_stream),
            ClientSession(read_stream, write_stream) as session,
        ):
            await session.initialize()
            tools = await session.list_tools()
            resources = await session.list_resources()
            templates = await session.list_resource_templates()

        assert tuple(tool.name for tool in tools.tools) == TOOL_NAMES
        assert resources.resources == []
        assert templates.resourceTemplates == []

    asyncio.run(run_client())


def _client(handler) -> McpApplicationClient:
    return McpApplicationClient(
        transport=httpx.MockTransport(handler),
        base_url="http://127.0.0.1:8000",
    )
