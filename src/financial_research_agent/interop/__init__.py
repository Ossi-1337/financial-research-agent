"""Local-only interoperability spike contracts for A2A discovery and MCP-style tools."""

from financial_research_agent.interop.contracts import (
    A2A_PROTOCOL_VERSION,
    INTEROP_DECISION,
    MCP_PROTOCOL_VERSION,
    READ_ONLY_STATUS_TOOL,
    A2AAgentCard,
    A2ASkill,
    InteropAccessDecision,
    InteropAccessPolicy,
    InteropAccessResult,
    InteropProtocol,
    MCPReadOnlyDispatcher,
    MCPTool,
    MCPToolResult,
    create_agent_card,
    create_read_only_status_tool,
    create_sanitized_status_payload,
)

__all__ = [
    "A2A_PROTOCOL_VERSION",
    "INTEROP_DECISION",
    "MCP_PROTOCOL_VERSION",
    "READ_ONLY_STATUS_TOOL",
    "A2AAgentCard",
    "A2ASkill",
    "InteropAccessDecision",
    "InteropAccessPolicy",
    "InteropAccessResult",
    "InteropProtocol",
    "MCPReadOnlyDispatcher",
    "MCPTool",
    "MCPToolResult",
    "create_agent_card",
    "create_read_only_status_tool",
    "create_sanitized_status_payload",
]
