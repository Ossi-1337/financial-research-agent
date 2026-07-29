from __future__ import annotations

import asyncio
import json
from dataclasses import FrozenInstanceError

import pytest

from financial_research_agent.llm import ToolCall
from financial_research_agent.tools import (
    ToolContext,
    ToolErrorCode,
    ToolPermission,
    ToolRegistry,
    ToolResult,
    ToolResultStatus,
    ToolSpec,
    validate_tool_arguments,
    validate_tool_schema,
)


async def _ok_handler(_context: ToolContext, _arguments):
    return ToolResult.succeeded(
        tool_call_id="placeholder",
        tool_name="echo",
        data={"ok": True},
        source="test",
    )


def test_tool_spec_is_immutable_and_converts_to_llm_definition() -> None:
    spec = ToolSpec(
        name="echo",
        description="Echo test.",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
        permissions=(ToolPermission.CALCULATION,),
        handler=_ok_handler,
    )

    definition = spec.to_llm_definition()

    assert definition.name == "echo"
    assert dict(definition.input_schema)["type"] == "object"
    with pytest.raises(FrozenInstanceError):
        spec.name = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        spec.input_schema["type"] = "changed"  # type: ignore[index]


def test_tool_schema_validation_rejects_invalid_schema() -> None:
    errors = validate_tool_schema({"type": "string"})

    assert "schema.type must be object" in errors
    with pytest.raises(ValueError, match="Invalid tool schema"):
        ToolSpec(
            name="bad",
            description="Bad.",
            input_schema={"type": "object", "properties": {"x": {"type": "unknown"}}},
            permissions=(ToolPermission.CALCULATION,),
            handler=_ok_handler,
        )


def test_tool_argument_validation_supports_nested_subset() -> None:
    schema = {
        "type": "object",
        "properties": {
            "mode": {"type": "string", "enum": ["fast", "slow"]},
            "items": {"type": "array", "items": {"type": "integer"}},
            "options": {
                "type": "object",
                "properties": {"enabled": {"type": "boolean"}},
                "required": ["enabled"],
                "additionalProperties": False,
            },
        },
        "required": ["mode", "items", "options"],
        "additionalProperties": False,
    }

    assert (
        validate_tool_arguments(
            schema,
            {"mode": "fast", "items": [1, 2], "options": {"enabled": True}},
        )
        == ()
    )
    errors = validate_tool_arguments(
        schema,
        {"mode": "other", "items": ["x"], "options": {}, "extra": True},
    )

    assert any("mode" in error for error in errors)
    assert any("items[0]" in error for error in errors)
    assert any("options.enabled is required" in error for error in errors)
    assert any("extra is not allowed" in error for error in errors)


def test_tool_argument_validation_enforces_string_and_array_bounds() -> None:
    schema = {
        "type": "object",
        "properties": {
            "label": {"type": "string", "maxLength": 4},
            "items": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 2,
            },
        },
        "required": ["label", "items"],
        "additionalProperties": False,
    }

    assert validate_tool_arguments(schema, {"label": "test", "items": ["a", "b"]}) == ()
    errors = validate_tool_arguments(
        schema,
        {"label": "too long", "items": ["a", "b", "c"]},
    )

    assert any("at most 4 characters" in error for error in errors)
    assert any("at most 2 items" in error for error in errors)


def test_registry_rejects_duplicates_and_executes_with_call_metadata() -> None:
    spec = ToolSpec(
        name="echo",
        description="Echo test.",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        permissions=(ToolPermission.CALCULATION,),
        handler=_ok_handler,
    )
    registry = ToolRegistry().register(spec)

    with pytest.raises(ValueError, match="already registered"):
        registry.register(spec)

    result = asyncio.run(
        registry.execute(
            ToolCall(id="call_1", name="echo"),
            ToolContext(
                allowed_permissions=(ToolPermission.CALCULATION,),
                allowed_tools=("echo",),
            ),
        )
    )

    assert result.status == ToolResultStatus.SUCCEEDED
    assert result.tool_call_id == "call_1"
    assert result.tool_name == "echo"


def test_tool_result_serializes_safe_json() -> None:
    result = ToolResult.failed(
        tool_call_id="call_1",
        tool_name="tool",
        error_code=ToolErrorCode.INVALID_ARGUMENTS,
        errors=("bad input",),
        data={"nested": {"values": (1, 2)}},
    )

    payload = json.loads(result.to_message_content())

    assert payload["status"] == "failed"
    assert payload["error_code"] == "invalid_arguments"
    assert payload["data"]["nested"]["values"] == [1, 2]
