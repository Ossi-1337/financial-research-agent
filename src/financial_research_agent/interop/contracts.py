from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

from financial_research_agent import __version__

A2A_PROTOCOL_VERSION = "1.0.1"
MCP_PROTOCOL_VERSION = "2025-03-26"
INTEROP_DECISION = "defer_a2a_runtime_accept_local_mcp_read_only_spike"
READ_ONLY_STATUS_TOOL = "financial_research_agent.status"


class InteropProtocol(StrEnum):
    A2A_DISCOVERY = "a2a_discovery"
    MCP_READ_ONLY = "mcp_read_only"


class InteropAccessDecision(StrEnum):
    ALLOWED = "allowed"
    DISABLED = "disabled"
    DENIED = "denied"


@dataclass(frozen=True, slots=True)
class InteropAccessResult:
    decision: InteropAccessDecision
    reason: str

    @property
    def allowed(self) -> bool:
        return self.decision == InteropAccessDecision.ALLOWED


@dataclass(frozen=True, slots=True)
class InteropAccessPolicy:
    enabled: bool = False
    local_only: bool = True
    api_key: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "api_key", _optional_text(self.api_key))
        if self.enabled and not self.local_only and self.api_key is None:
            raise ValueError("api_key is required when interop is enabled for non-local clients")

    @property
    def protocols(self) -> tuple[InteropProtocol, ...]:
        return (InteropProtocol.A2A_DISCOVERY, InteropProtocol.MCP_READ_ONLY)

    def evaluate(
        self,
        *,
        client_host: str | None,
        authorization: str | None = None,
        api_key_header: str | None = None,
    ) -> InteropAccessResult:
        if not self.enabled:
            return InteropAccessResult(InteropAccessDecision.DISABLED, "interoperability_disabled")
        if self.api_key is not None:
            supplied_key = _bearer_token(authorization) or _optional_text(api_key_header)
            if supplied_key != self.api_key:
                return InteropAccessResult(InteropAccessDecision.DENIED, "invalid_interop_key")
            return InteropAccessResult(InteropAccessDecision.ALLOWED, "api_key")
        if self.local_only and _is_loopback_host(client_host):
            return InteropAccessResult(InteropAccessDecision.ALLOWED, "local_client")
        return InteropAccessResult(InteropAccessDecision.DENIED, "local_client_required")

    def to_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "local_only": self.local_only,
            "api_key_configured": self.api_key is not None,
            "protocols": [protocol.value for protocol in self.protocols],
        }


@dataclass(frozen=True, slots=True)
class A2ASkill:
    id: str
    name: str
    description: str
    tags: tuple[str, ...] = ()
    examples: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_text("id", self.id))
        object.__setattr__(self, "name", _require_text("name", self.name))
        object.__setattr__(self, "description", _require_text("description", self.description))
        object.__setattr__(self, "tags", _text_tuple("tags", self.tags))
        object.__setattr__(self, "examples", _text_tuple("examples", self.examples))

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "tags": list(self.tags),
            "examples": list(self.examples),
        }


@dataclass(frozen=True, slots=True)
class A2AAgentCard:
    name: str
    description: str
    url: str
    version: str
    protocol_version: str = A2A_PROTOCOL_VERSION
    skills: tuple[A2ASkill, ...] = ()
    capabilities: Mapping[str, object] = field(default_factory=dict)
    default_input_modes: tuple[str, ...] = ("application/json", "text/plain")
    default_output_modes: tuple[str, ...] = ("application/json", "text/plain")
    security: tuple[Mapping[str, object], ...] = ()
    security_schemes: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _require_text("name", self.name))
        object.__setattr__(self, "description", _require_text("description", self.description))
        object.__setattr__(self, "url", _require_text("url", self.url))
        object.__setattr__(self, "version", _require_text("version", self.version))
        object.__setattr__(
            self,
            "protocol_version",
            _require_text("protocol_version", self.protocol_version),
        )
        object.__setattr__(self, "skills", _skill_tuple(self.skills))
        object.__setattr__(self, "capabilities", _object_mapping(self.capabilities))
        object.__setattr__(
            self,
            "default_input_modes",
            _text_tuple("default_input_modes", self.default_input_modes),
        )
        object.__setattr__(
            self,
            "default_output_modes",
            _text_tuple("default_output_modes", self.default_output_modes),
        )
        object.__setattr__(self, "security", tuple(_object_mapping(item) for item in self.security))
        object.__setattr__(self, "security_schemes", _object_mapping(self.security_schemes))

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "url": self.url,
            "version": self.version,
            "protocolVersion": self.protocol_version,
            "capabilities": dict(self.capabilities),
            "defaultInputModes": list(self.default_input_modes),
            "defaultOutputModes": list(self.default_output_modes),
            "skills": [skill.to_dict() for skill in self.skills],
            "security": [dict(item) for item in self.security],
            "securitySchemes": dict(self.security_schemes),
        }


@dataclass(frozen=True, slots=True)
class MCPTool:
    name: str
    description: str
    input_schema: Mapping[str, object]
    annotations: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _require_text("name", self.name))
        object.__setattr__(self, "description", _require_text("description", self.description))
        object.__setattr__(self, "input_schema", _object_mapping(self.input_schema))
        object.__setattr__(self, "annotations", _object_mapping(self.annotations))

    def to_dict(self) -> dict[str, object]:
        payload = {
            "name": self.name,
            "description": self.description,
            "inputSchema": dict(self.input_schema),
        }
        if self.annotations:
            payload["annotations"] = dict(self.annotations)
        return payload


@dataclass(frozen=True, slots=True)
class MCPToolResult:
    content: tuple[Mapping[str, str], ...]
    is_error: bool = False

    @classmethod
    def text(cls, text: str, *, is_error: bool = False) -> MCPToolResult:
        return cls(content=(MappingProxyType({"type": "text", "text": text}),), is_error=is_error)

    def to_dict(self) -> dict[str, object]:
        return {
            "content": [dict(item) for item in self.content],
            "isError": self.is_error,
        }


def create_read_only_status_tool() -> MCPTool:
    return MCPTool(
        name=READ_ONLY_STATUS_TOOL,
        description=(
            "Read sanitized Financial Research Agent runtime status. This tool is read-only, "
            "does not call LLMs, does not fetch market data, and does not expose secrets."
        ),
        input_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        annotations={
            "readOnlyHint": True,
            "destructiveHint": False,
            "openWorldHint": False,
        },
    )


def create_agent_card(
    *,
    base_url: str,
    version: str,
    api_key_required: bool,
) -> A2AAgentCard:
    normalized_base_url = base_url.rstrip("/")
    security_schemes: dict[str, object]
    security: tuple[Mapping[str, object], ...]
    if api_key_required:
        security_schemes = {
            "fraInteropApiKey": {
                "type": "apiKey",
                "in": "header",
                "name": "X-FRA-Interop-Key",
            }
        }
        security = (MappingProxyType({"fraInteropApiKey": []}),)
    else:
        security_schemes = {
            "localOnly": {
                "type": "localOnly",
                "description": "Loopback-only access enforced by the application.",
            }
        }
        security = (MappingProxyType({"localOnly": []}),)
    return A2AAgentCard(
        name="financial-research-agent",
        description=(
            "Local-first financial research assistant. Milestone 24 exposes discovery and "
            "one read-only MCP-style status tool only."
        ),
        url=f"{normalized_base_url}/api/interop/mcp",
        version=version,
        capabilities={
            "streaming": False,
            "pushNotifications": False,
            "stateTransitionHistory": False,
        },
        skills=(
            A2ASkill(
                id="read_sanitized_status",
                name="Read sanitized status",
                description=(
                    "Return non-secret local runtime status and enabled research capabilities."
                ),
                tags=("status", "financial-research-agent", "read-only"),
                examples=("Check whether local orchestration is available.",),
            ),
        ),
        security=security,
        security_schemes=security_schemes,
    )


def create_sanitized_status_payload(
    *,
    environment: str,
    chat_provider: str,
    chat_model: str | None,
    chat_registered: bool,
    storage_provider: str,
    retrieval_provider: str,
    interop_policy: InteropAccessPolicy,
) -> dict[str, object]:
    return {
        "app": "financial-research-agent",
        "environment": environment,
        "status": "ok",
        "chat": {
            "provider": chat_provider,
            "model": chat_model,
            "registered": chat_registered,
        },
        "capabilities": {
            "orchestration": "sequential_local_safe",
            "synthesis": "source_backed_deterministic",
            "retrieval": retrieval_provider,
            "storage": storage_provider,
            "recommendations": "disabled",
        },
        "interoperability": interop_policy.to_dict(),
    }


class MCPReadOnlyDispatcher:
    def __init__(self, *, status_payload: Mapping[str, object]) -> None:
        self._status_payload = _plain_json_value(status_payload)
        self._tools = {READ_ONLY_STATUS_TOOL: create_read_only_status_tool()}

    def handle(self, request: Mapping[str, object]) -> dict[str, object]:
        request_id = request.get("id")
        if request.get("jsonrpc") != "2.0":
            return _json_rpc_error(request_id, -32600, "Invalid JSON-RPC request.")
        method = request.get("method")
        if method == "initialize":
            return _json_rpc_result(
                request_id,
                {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "financial-research-agent", "version": __version__},
                },
            )
        if method == "tools/list":
            return _json_rpc_result(
                request_id,
                {"tools": [tool.to_dict() for tool in self._tools.values()]},
            )
        if method == "tools/call":
            return self._call_tool(request_id, request.get("params"))
        return _json_rpc_error(request_id, -32601, f"Unknown method: {method}")

    def _call_tool(self, request_id: object, params: object) -> dict[str, object]:
        if not isinstance(params, Mapping):
            return _json_rpc_error(request_id, -32602, "tools/call params must be an object.")
        name = params.get("name")
        arguments = params.get("arguments", {})
        if name != READ_ONLY_STATUS_TOOL:
            return _json_rpc_error(request_id, -32602, f"Unknown tool: {name}")
        if not isinstance(arguments, Mapping):
            return _json_rpc_error(request_id, -32602, "Tool arguments must be an object.")
        if arguments:
            return _json_rpc_error(
                request_id,
                -32602,
                f"{READ_ONLY_STATUS_TOOL} does not accept arguments.",
            )
        return _json_rpc_result(
            request_id,
            MCPToolResult.text(json.dumps(self._status_payload, sort_keys=True)).to_dict(),
        )


def _json_rpc_result(request_id: object, result: Mapping[str, object]) -> dict[str, object]:
    return {"jsonrpc": "2.0", "id": request_id, "result": dict(result)}


def _json_rpc_error(request_id: object, code: int, message: str) -> dict[str, object]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def _bearer_token(value: str | None) -> str | None:
    text = _optional_text(value)
    if text is None:
        return None
    prefix = "bearer "
    if text.casefold().startswith(prefix):
        token = text[len(prefix) :].strip()
        return token or None
    return None


def _is_loopback_host(value: str | None) -> bool:
    host = (value or "").strip().casefold()
    return host in {"127.0.0.1", "::1", "localhost", "testclient"}


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


def _text_tuple(name: str, values) -> tuple[str, ...]:
    if isinstance(values, str):
        raise ValueError(f"{name} must be an iterable of strings, not a string")
    return tuple(_require_text(f"{name}[{index}]", value) for index, value in enumerate(values))


def _object_mapping(values: Mapping[str, object]) -> Mapping[str, object]:
    if not isinstance(values, Mapping):
        raise ValueError("value must be a mapping")
    return MappingProxyType({_require_text("key", str(key)): item for key, item in values.items()})


def _plain_json_value(value):
    if isinstance(value, Mapping):
        return {str(key): _plain_json_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_plain_json_value(item) for item in value]
    return value


def _skill_tuple(values) -> tuple[A2ASkill, ...]:
    skills = tuple(values)
    for index, skill in enumerate(skills):
        if not isinstance(skill, A2ASkill):
            raise ValueError(f"skills[{index}] must be an A2ASkill")
    return skills
