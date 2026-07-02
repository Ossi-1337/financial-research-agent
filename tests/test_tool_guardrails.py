from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from financial_research_agent.llm import ToolCall
from financial_research_agent.tools import (
    ToolContext,
    ToolErrorCode,
    ToolPermission,
    ToolRegistry,
    ToolResult,
    ToolResultStatus,
    ToolSpec,
)


async def _ok_handler(_context: ToolContext, _arguments: Mapping[str, Any]) -> ToolResult:
    return ToolResult.succeeded(tool_call_id="placeholder", tool_name="ok", data={"ok": True})


def test_unknown_tool_returns_structured_denial() -> None:
    result = asyncio.run(ToolRegistry().execute(ToolCall(id="call_1", name="missing_tool")))

    assert result.status == ToolResultStatus.DENIED
    assert result.error_code == ToolErrorCode.UNKNOWN_TOOL


def test_permission_mismatch_returns_structured_denial() -> None:
    registry = ToolRegistry(
        [
            ToolSpec(
                name="local_read",
                description="Read local.",
                input_schema={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                permissions=(ToolPermission.LOCAL_READ,),
                handler=_ok_handler,
            )
        ]
    )
    context = ToolContext(allowed_permissions=(ToolPermission.CALCULATION,))

    result = asyncio.run(registry.execute(ToolCall(id="call_1", name="local_read"), context))

    assert result.status == ToolResultStatus.DENIED
    assert result.error_code == ToolErrorCode.PERMISSION_DENIED


def test_invalid_arguments_return_structured_failure() -> None:
    registry = ToolRegistry(
        [
            ToolSpec(
                name="needs_text",
                description="Needs text.",
                input_schema={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                    "additionalProperties": False,
                },
                permissions=(ToolPermission.CALCULATION,),
                handler=_ok_handler,
            )
        ]
    )

    result = asyncio.run(
        registry.execute(
            ToolCall(id="call_1", name="needs_text", arguments={"text": 123}),
        )
    )

    assert result.status == ToolResultStatus.FAILED
    assert result.error_code == ToolErrorCode.INVALID_ARGUMENTS
    assert any("text" in error for error in result.errors)


def test_handler_exception_returns_structured_failure() -> None:
    async def failing_handler(_context: ToolContext, _arguments: Mapping[str, Any]) -> ToolResult:
        raise RuntimeError("boom")

    registry = ToolRegistry(
        [
            ToolSpec(
                name="fails",
                description="Fails.",
                input_schema={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                permissions=(ToolPermission.CALCULATION,),
                handler=failing_handler,
            )
        ]
    )

    result = asyncio.run(registry.execute(ToolCall(id="call_1", name="fails")))

    assert result.status == ToolResultStatus.FAILED
    assert result.error_code == ToolErrorCode.EXECUTION_FAILED
    assert "boom" in result.errors[0]


def test_timeout_returns_structured_failure() -> None:
    async def slow_handler(_context: ToolContext, _arguments: Mapping[str, Any]) -> ToolResult:
        await asyncio.sleep(0.05)
        return ToolResult.succeeded(tool_call_id="placeholder", tool_name="slow", data={})

    registry = ToolRegistry(
        [
            ToolSpec(
                name="slow",
                description="Slow.",
                input_schema={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                permissions=(ToolPermission.CALCULATION,),
                timeout_seconds=0.001,
                handler=slow_handler,
            )
        ]
    )

    result = asyncio.run(registry.execute(ToolCall(id="call_1", name="slow")))

    assert result.status == ToolResultStatus.FAILED
    assert result.error_code == ToolErrorCode.TIMEOUT
