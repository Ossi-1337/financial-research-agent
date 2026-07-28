from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any

MCP_RESULT_SCHEMA_VERSION = 1


class McpResultStatus(StrEnum):
    ACCEPTED = "accepted"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class McpErrorCode(StrEnum):
    INVALID_ARGUMENTS = "invalid_arguments"
    APP_UNAVAILABLE = "app_unavailable"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    MALFORMED_RESPONSE = "malformed_response"
    RESEARCH_FAILED = "research_failed"
    INTERNAL_ERROR = "internal_error"


@dataclass(frozen=True, slots=True)
class McpResultEnvelope:
    capability_id: str
    status: McpResultStatus
    data: Mapping[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    error_code: McpErrorCode | None = None
    message: str | None = None
    generated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    schema_version: int = MCP_RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "capability_id",
            _require_text("capability_id", self.capability_id),
        )
        object.__setattr__(self, "status", McpResultStatus(self.status))
        object.__setattr__(self, "data", MappingProxyType(dict(self.data)))
        object.__setattr__(
            self,
            "warnings",
            tuple(dict.fromkeys(_require_text("warning", item) for item in self.warnings)),
        )
        if self.error_code is not None:
            object.__setattr__(self, "error_code", McpErrorCode(self.error_code))
        object.__setattr__(
            self,
            "message",
            self.message.strip() if self.message and self.message.strip() else None,
        )
        if self.generated_at.tzinfo is None:
            raise ValueError("generated_at must be timezone-aware")
        if self.schema_version != MCP_RESULT_SCHEMA_VERSION:
            raise ValueError("unsupported MCP result schema version")
        if self.status == McpResultStatus.FAILED and self.error_code is None:
            raise ValueError("failed MCP result requires error_code")
        if self.status != McpResultStatus.FAILED and self.error_code is not None:
            raise ValueError("successful MCP result cannot include error_code")

    @classmethod
    def accepted(
        cls,
        *,
        capability_id: str,
        data: Mapping[str, Any],
        warnings: tuple[str, ...] = (),
    ) -> McpResultEnvelope:
        return cls(
            capability_id=capability_id,
            status=McpResultStatus.ACCEPTED,
            data=data,
            warnings=warnings,
        )

    @classmethod
    def succeeded(
        cls,
        *,
        capability_id: str,
        data: Mapping[str, Any],
        warnings: tuple[str, ...] = (),
    ) -> McpResultEnvelope:
        return cls(
            capability_id=capability_id,
            status=McpResultStatus.SUCCEEDED,
            data=data,
            warnings=warnings,
        )

    @classmethod
    def failed(
        cls,
        *,
        capability_id: str,
        error_code: McpErrorCode,
        message: str,
    ) -> McpResultEnvelope:
        return cls(
            capability_id=capability_id,
            status=McpResultStatus.FAILED,
            error_code=error_code,
            message=message,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "capability_id": self.capability_id,
            "status": self.status.value,
            "data": _json_ready(self.data),
            "warnings": list(self.warnings),
            "error_code": self.error_code.value if self.error_code is not None else None,
            "message": self.message,
            "generated_at": self.generated_at.isoformat(),
        }


def _require_text(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    text = value.strip()
    if not text:
        raise ValueError(f"{name} is required")
    return text


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_ready(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, StrEnum):
        return value.value
    return value
