from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Self

from financial_research_agent.llm import ToolCall, ToolDefinition
from financial_research_agent.tools.schema import validate_tool_arguments, validate_tool_schema

ToolHandler = Callable[["ToolContext", Mapping[str, Any]], "ToolResult | Awaitable[ToolResult]"]


class ToolPermission(StrEnum):
    CLOCK = "clock"
    CALCULATION = "calculation"
    ENTITY_LOOKUP = "entity_lookup"
    LOCAL_READ = "local_read"


class ToolResultStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    DENIED = "denied"


class ToolErrorCode(StrEnum):
    UNKNOWN_TOOL = "unknown_tool"
    PERMISSION_DENIED = "permission_denied"
    INVALID_SCHEMA = "invalid_schema"
    INVALID_ARGUMENTS = "invalid_arguments"
    EXECUTION_FAILED = "execution_failed"
    TIMEOUT = "timeout"
    NOT_FOUND = "not_found"
    DIVISION_BY_ZERO = "division_by_zero"
    MAX_ROUNDS_EXCEEDED = "max_rounds_exceeded"


@dataclass(frozen=True, slots=True)
class ToolContext:
    allowed_permissions: tuple[ToolPermission, ...] = field(
        default_factory=lambda: tuple(ToolPermission)
    )
    local_evidence: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "allowed_permissions", _permission_tuple(self.allowed_permissions))
        object.__setattr__(self, "local_evidence", _nested_mapping(self.local_evidence))
        object.__setattr__(self, "metadata", _text_mapping("metadata", self.metadata))

    def allows(self, permissions: Iterable[ToolPermission | str]) -> bool:
        allowed = set(self.allowed_permissions)
        return all(ToolPermission(permission) in allowed for permission in permissions)


@dataclass(frozen=True, slots=True)
class ToolResult:
    tool_call_id: str
    tool_name: str
    status: ToolResultStatus
    data: Mapping[str, Any] = field(default_factory=dict)
    source: str | None = None
    freshness: str | None = None
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    error_code: ToolErrorCode | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "tool_call_id", _require_text("tool_call_id", self.tool_call_id))
        object.__setattr__(self, "tool_name", _require_text("tool_name", self.tool_name))
        object.__setattr__(self, "status", ToolResultStatus(self.status))
        object.__setattr__(self, "data", _freeze_mapping("data", self.data))
        object.__setattr__(self, "source", _optional_text(self.source))
        object.__setattr__(self, "freshness", _optional_text(self.freshness))
        object.__setattr__(self, "warnings", _text_tuple("warnings", self.warnings))
        object.__setattr__(self, "errors", _text_tuple("errors", self.errors))
        if self.error_code is not None:
            object.__setattr__(self, "error_code", ToolErrorCode(self.error_code))

    @classmethod
    def succeeded(
        cls,
        *,
        tool_call_id: str,
        tool_name: str,
        data: Mapping[str, Any],
        source: str | None = None,
        freshness: str | None = None,
        warnings: Iterable[str] = (),
    ) -> Self:
        return cls(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            status=ToolResultStatus.SUCCEEDED,
            data=data,
            source=source,
            freshness=freshness,
            warnings=tuple(warnings),
        )

    @classmethod
    def failed(
        cls,
        *,
        tool_call_id: str,
        tool_name: str,
        error_code: ToolErrorCode | str,
        errors: Iterable[str],
        data: Mapping[str, Any] | None = None,
    ) -> Self:
        return cls(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            status=ToolResultStatus.FAILED,
            data=data or {},
            errors=tuple(errors),
            error_code=ToolErrorCode(error_code),
        )

    @classmethod
    def denied(
        cls,
        *,
        tool_call_id: str,
        tool_name: str,
        error_code: ToolErrorCode | str,
        errors: Iterable[str],
    ) -> Self:
        return cls(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            status=ToolResultStatus.DENIED,
            errors=tuple(errors),
            error_code=ToolErrorCode(error_code),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "status": self.status.value,
            "data": _json_ready(self.data),
            "source": self.source,
            "freshness": self.freshness,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "error_code": self.error_code.value if self.error_code is not None else None,
        }

    def to_message_content(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    input_schema: Mapping[str, Any]
    permissions: tuple[ToolPermission, ...]
    handler: ToolHandler = field(compare=False, repr=False)
    timeout_seconds: float = 5.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _require_text("name", self.name))
        object.__setattr__(self, "description", _require_text("description", self.description))
        object.__setattr__(self, "input_schema", _freeze_mapping("input_schema", self.input_schema))
        object.__setattr__(self, "permissions", _permission_tuple(self.permissions))
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if not self.permissions:
            raise ValueError("permissions must contain at least one value")
        schema_errors = validate_tool_schema(self.input_schema)
        if schema_errors:
            raise ValueError(f"Invalid tool schema for {self.name}: {'; '.join(schema_errors)}")

    def to_llm_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            input_schema=self.input_schema,
        )


class ToolRegistry:
    def __init__(self, specs: Iterable[ToolSpec] = ()) -> None:
        self._specs: dict[str, ToolSpec] = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec: ToolSpec) -> Self:
        name = _require_text("name", spec.name)
        if name in self._specs:
            raise ValueError(f"Tool is already registered: {name}")
        self._specs[name] = spec
        return self

    def get(self, name: str) -> ToolSpec:
        return self._specs[_require_text("name", name)]

    def has(self, name: str) -> bool:
        return _require_text("name", name) in self._specs

    def specs(self, context: ToolContext | None = None) -> tuple[ToolSpec, ...]:
        values = tuple(self._specs.values())
        if context is None:
            return values
        return tuple(spec for spec in values if context.allows(spec.permissions))

    def tool_definitions(self, context: ToolContext | None = None) -> tuple[ToolDefinition, ...]:
        return tuple(spec.to_llm_definition() for spec in self.specs(context))

    async def execute(self, tool_call: ToolCall, context: ToolContext | None = None) -> ToolResult:
        tool_context = context or ToolContext()
        spec = self._specs.get(tool_call.name)
        if spec is None:
            return ToolResult.denied(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                error_code=ToolErrorCode.UNKNOWN_TOOL,
                errors=(f"Unknown tool: {tool_call.name}",),
            )
        if not tool_context.allows(spec.permissions):
            return ToolResult.denied(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                error_code=ToolErrorCode.PERMISSION_DENIED,
                errors=(f"Tool is not allowed: {tool_call.name}",),
            )
        argument_errors = validate_tool_arguments(spec.input_schema, tool_call.arguments)
        if argument_errors:
            return ToolResult.failed(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                error_code=ToolErrorCode.INVALID_ARGUMENTS,
                errors=argument_errors,
            )
        try:
            result = await asyncio.wait_for(
                _call_handler(spec.handler, tool_context, tool_call.arguments),
                timeout=spec.timeout_seconds,
            )
        except TimeoutError:
            return ToolResult.failed(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                error_code=ToolErrorCode.TIMEOUT,
                errors=(f"Tool timed out after {spec.timeout_seconds} seconds.",),
            )
        except Exception as exc:
            return ToolResult.failed(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                error_code=ToolErrorCode.EXECUTION_FAILED,
                errors=(f"Tool execution failed: {exc}",),
            )
        if not isinstance(result, ToolResult):
            return ToolResult.failed(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                error_code=ToolErrorCode.EXECUTION_FAILED,
                errors=("Tool handler returned an invalid result.",),
            )
        return replace(result, tool_call_id=tool_call.id, tool_name=tool_call.name)


async def _call_handler(
    handler: ToolHandler,
    context: ToolContext,
    arguments: Mapping[str, Any],
) -> ToolResult:
    result = handler(context, arguments)
    if inspect.isawaitable(result):
        return await result
    return result


def _require_text(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    text = value.strip()
    if text == "":
        raise ValueError(f"{name} is required")
    return text


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    return text or None


def _text_tuple(name: str, values: Iterable[str]) -> tuple[str, ...]:
    if isinstance(values, str):
        raise ValueError(f"{name} must be an iterable of strings, not a string")
    return tuple(_require_text(f"{name}[{index}]", value) for index, value in enumerate(values))


def _permission_tuple(values: Iterable[ToolPermission | str]) -> tuple[ToolPermission, ...]:
    if isinstance(values, str):
        raise ValueError("permissions must be an iterable, not a string")
    return tuple(ToolPermission(value) for value in values)


def _text_mapping(name: str, values: Mapping[str, str]) -> Mapping[str, str]:
    return MappingProxyType(
        {
            _require_text(f"{name}.key", key): _require_text(f"{name}[{key!r}]", value)
            for key, value in values.items()
        }
    )


def _nested_mapping(values: Mapping[str, Mapping[str, Any]]) -> Mapping[str, Mapping[str, Any]]:
    return MappingProxyType(
        {
            _require_text("local_evidence.key", key): _freeze_mapping("local_evidence", value)
            for key, value in values.items()
        }
    )


def _freeze_mapping(name: str, values: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(values, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return MappingProxyType(
        {_require_text(f"{name}.key", key): _freeze_value(value) for key, value in values.items()}
    )


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _freeze_mapping("value", value)
    if isinstance(value, str | bytes):
        return value
    if isinstance(value, tuple | list):
        return tuple(_freeze_value(item) for item in value)
    return value


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_ready(item) for item in value]
    if isinstance(value, ToolResultStatus | ToolErrorCode | ToolPermission):
        return value.value
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime | date):
        return value.isoformat()
    return value
