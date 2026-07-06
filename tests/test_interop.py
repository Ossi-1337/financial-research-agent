from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest

from financial_research_agent.interop import (
    INTEROP_DECISION,
    MCP_PROTOCOL_VERSION,
    READ_ONLY_STATUS_TOOL,
    A2ASkill,
    InteropAccessDecision,
    InteropAccessPolicy,
    MCPReadOnlyDispatcher,
    create_agent_card,
    create_read_only_status_tool,
    create_sanitized_status_payload,
)


def test_interop_contracts_are_immutable_and_describe_spike_decision() -> None:
    skill = A2ASkill(
        id="status",
        name="Status",
        description="Read status",
        tags=("read-only",),
        examples=("Check status",),
    )

    assert INTEROP_DECISION == "defer_a2a_runtime_accept_local_mcp_read_only_spike"
    assert skill.to_dict()["tags"] == ["read-only"]
    with pytest.raises(FrozenInstanceError):
        skill.name = "changed"  # type: ignore[misc]


def test_interop_access_policy_defaults_to_disabled_and_allows_local_when_enabled() -> None:
    disabled = InteropAccessPolicy()
    enabled = InteropAccessPolicy(enabled=True)

    assert disabled.evaluate(client_host="127.0.0.1").decision == InteropAccessDecision.DISABLED
    assert enabled.evaluate(client_host="127.0.0.1").allowed is True
    assert enabled.evaluate(client_host="::1").allowed is True
    assert enabled.evaluate(client_host="203.0.113.9").decision == InteropAccessDecision.DENIED


def test_interop_access_policy_supports_header_or_bearer_key_for_remote_mode() -> None:
    policy = InteropAccessPolicy(enabled=True, local_only=False, api_key="secret")

    assert policy.evaluate(client_host="203.0.113.9").allowed is False
    assert policy.evaluate(client_host="203.0.113.9", api_key_header="secret").allowed is True
    assert policy.evaluate(client_host="203.0.113.9", authorization="Bearer secret").allowed is True

    with pytest.raises(ValueError, match="api_key is required"):
        InteropAccessPolicy(enabled=True, local_only=False)


def test_a2a_agent_card_advertises_read_only_local_skill_without_secret() -> None:
    card = create_agent_card(
        base_url="http://127.0.0.1:8000/",
        version="0.1.0",
        api_key_required=True,
    ).to_dict()

    assert card["name"] == "financial-research-agent"
    assert card["url"] == "http://127.0.0.1:8000/api/interop/mcp"
    assert card["protocolVersion"]
    assert card["skills"][0]["id"] == "read_sanitized_status"
    assert card["securitySchemes"]["fraInteropApiKey"]["name"] == "X-FRA-Interop-Key"
    assert "secret-key" not in json.dumps(card).casefold()


def test_mcp_read_only_dispatcher_initializes_lists_and_calls_status_tool() -> None:
    policy = InteropAccessPolicy(enabled=True)
    status = create_sanitized_status_payload(
        environment="test",
        chat_provider="offline-test",
        chat_model="offline-test",
        chat_registered=True,
        storage_provider="local-json",
        retrieval_provider="local-vector",
        interop_policy=policy,
    )
    dispatcher = MCPReadOnlyDispatcher(status_payload=status)

    initialized = dispatcher.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    listed = dispatcher.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    called = dispatcher.handle(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": READ_ONLY_STATUS_TOOL, "arguments": {}},
        }
    )
    tool_text = called["result"]["content"][0]["text"]

    assert initialized["result"]["protocolVersion"] == MCP_PROTOCOL_VERSION
    assert listed["result"]["tools"][0]["name"] == READ_ONLY_STATUS_TOOL
    assert json.loads(tool_text)["capabilities"]["recommendations"] == "disabled"
    assert "secret-key" not in tool_text


def test_mcp_read_only_dispatcher_rejects_unknown_methods_and_mutating_arguments() -> None:
    dispatcher = MCPReadOnlyDispatcher(status_payload={"app": "financial-research-agent"})

    unknown_method = dispatcher.handle({"jsonrpc": "2.0", "id": 1, "method": "tasks/send"})
    unknown_tool = dispatcher.handle(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "shell.exec", "arguments": {}},
        }
    )
    unexpected_args = dispatcher.handle(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": READ_ONLY_STATUS_TOOL, "arguments": {"refresh": True}},
        }
    )

    assert unknown_method["error"]["code"] == -32601
    assert unknown_tool["error"]["code"] == -32602
    assert unexpected_args["error"]["code"] == -32602


def test_read_only_status_tool_schema_has_no_arguments() -> None:
    tool = create_read_only_status_tool()

    assert tool.input_schema["additionalProperties"] is False
    assert tool.annotations["readOnlyHint"] is True
    assert tool.annotations["destructiveHint"] is False
