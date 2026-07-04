"""Deterministic tool registry and function-calling loop."""

from financial_research_agent.tools.builtins import (
    calculate_ratio_tool,
    create_default_tool_registry,
    current_utc_datetime_tool,
    read_local_evidence_tool,
    resolve_company_stub_tool,
    resolve_company_tool,
)
from financial_research_agent.tools.contracts import (
    ToolContext,
    ToolErrorCode,
    ToolPermission,
    ToolRegistry,
    ToolResult,
    ToolResultStatus,
    ToolSpec,
)
from financial_research_agent.tools.runner import ToolCallingRunner, ToolLoopResult
from financial_research_agent.tools.schema import validate_tool_arguments, validate_tool_schema

__all__ = [
    "ToolCallingRunner",
    "ToolContext",
    "ToolErrorCode",
    "ToolLoopResult",
    "ToolPermission",
    "ToolRegistry",
    "ToolResult",
    "ToolResultStatus",
    "ToolSpec",
    "calculate_ratio_tool",
    "create_default_tool_registry",
    "current_utc_datetime_tool",
    "read_local_evidence_tool",
    "resolve_company_stub_tool",
    "resolve_company_tool",
    "validate_tool_arguments",
    "validate_tool_schema",
]
