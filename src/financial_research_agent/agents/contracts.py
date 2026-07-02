from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Self

from financial_research_agent.llm import ResponseFormat, ResponseFormatType
from financial_research_agent.tools.schema import validate_tool_arguments, validate_tool_schema

SEMANTIC_VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")


class AgentRole(StrEnum):
    ORCHESTRATOR = "orchestrator"
    FINANCIAL_REPORT_ANALYST = "financial_report_analyst"
    STOCK_ANALYST = "stock_analyst"
    NEWS_MACRO_ANALYST = "news_macro_analyst"
    SYNTHESIS_AGENT = "synthesis_agent"


@dataclass(frozen=True, slots=True)
class PromptVersion:
    value: str

    def __post_init__(self) -> None:
        text = _require_text("value", self.value)
        if SEMANTIC_VERSION_PATTERN.fullmatch(text) is None:
            raise ValueError("prompt version must use semantic version format x.y.z")
        object.__setattr__(self, "value", text)


@dataclass(frozen=True, slots=True)
class AgentOutputSchema:
    name: str
    schema: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _require_text("name", self.name))
        object.__setattr__(self, "schema", _freeze_mapping("schema", self.schema))
        schema_errors = validate_tool_schema(self.schema)
        if schema_errors:
            message = f"Invalid agent output schema for {self.name}: {'; '.join(schema_errors)}"
            raise ValueError(message)

    def validate_output(self, output: Mapping[str, Any]) -> tuple[str, ...]:
        return validate_tool_arguments(self.schema, output)


@dataclass(frozen=True, slots=True)
class PromptContract:
    id: str
    role: AgentRole
    version: PromptVersion
    system_prompt: str
    description: str
    allowed_tools: tuple[str, ...]
    output_schema: AgentOutputSchema

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_text("id", self.id))
        object.__setattr__(self, "role", AgentRole(self.role))
        if not isinstance(self.version, PromptVersion):
            raise ValueError("version must be a PromptVersion")
        object.__setattr__(
            self,
            "system_prompt",
            _require_text("system_prompt", self.system_prompt),
        )
        object.__setattr__(self, "description", _require_text("description", self.description))
        object.__setattr__(self, "allowed_tools", _text_tuple("allowed_tools", self.allowed_tools))
        if not isinstance(self.output_schema, AgentOutputSchema):
            raise ValueError("output_schema must be an AgentOutputSchema")

    def response_format(self) -> ResponseFormat:
        return ResponseFormat(
            format_type=ResponseFormatType.JSON_SCHEMA,
            name=self.output_schema.name,
            json_schema=self.output_schema.schema,
        )


class PromptCatalog:
    def __init__(self, contracts: Iterable[PromptContract] = ()) -> None:
        self._by_id: dict[str, PromptContract] = {}
        self._by_role: dict[AgentRole, PromptContract] = {}
        for contract in contracts:
            self.register(contract)

    def register(self, contract: PromptContract) -> Self:
        if contract.id in self._by_id:
            raise ValueError(f"Prompt contract is already registered: {contract.id}")
        if contract.role in self._by_role:
            raise ValueError(f"Prompt role is already registered: {contract.role.value}")
        self._by_id[contract.id] = contract
        self._by_role[contract.role] = contract
        return self

    def by_id(self, prompt_id: str) -> PromptContract:
        return self._by_id[_require_text("prompt_id", prompt_id)]

    def by_role(self, role: AgentRole | str) -> PromptContract:
        return self._by_role[AgentRole(role)]

    def contracts(self) -> tuple[PromptContract, ...]:
        return tuple(self._by_id.values())


def _require_text(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    text = value.strip()
    if text == "":
        raise ValueError(f"{name} is required")
    return text


def _text_tuple(name: str, values: Iterable[str]) -> tuple[str, ...]:
    if isinstance(values, str):
        raise ValueError(f"{name} must be an iterable of strings, not a string")
    return tuple(_require_text(f"{name}[{index}]", value) for index, value in enumerate(values))


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
