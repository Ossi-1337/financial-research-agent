from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pytest

from financial_research_agent.agents import (
    AgentDecisionMode,
    AgentDecisionService,
    AgentOutputSchema,
    AgentRole,
    AgentRuntimeError,
    AgentRuntimeResolver,
    PromptContract,
    PromptVersion,
    StructuredAgentRunner,
)
from financial_research_agent.llm import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    MessageRole,
    ModelMetadata,
    OfflineTestProvider,
    ProviderCapability,
    ToolCall,
)
from financial_research_agent.llm.registry import ProviderRegistry
from financial_research_agent.settings import Settings
from financial_research_agent.tools import (
    ToolContext,
    ToolPermission,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)

EVIDENCE_ID = "evidence:test:1"


@pytest.mark.parametrize(
    "mode",
    tuple(AgentDecisionMode),
)
def test_orchestrator_accepts_each_valid_decision(mode: AgentDecisionMode) -> None:
    provider = DecisionProvider(mode)

    decision = asyncio.run(
        AgentDecisionService(provider, model="scripted-model").decide(
            content="TEST TOOL OUTPUT request",
        )
    )

    assert decision.mode == mode
    assert provider.requests[0].response_format is not None
    assert provider.requests[0].metadata["agent_role"] == "orchestrator"


def test_orchestrator_rejects_client_selected_specialist() -> None:
    provider = DecisionProvider(AgentDecisionMode.RESEARCH, roles=("external-agent",))

    with pytest.raises(AgentRuntimeError, match="invalid decision"):
        asyncio.run(
            AgentDecisionService(provider).decide(content="Research TEST TOOL OUTPUT company")
        )


def test_orchestrator_research_decision_requires_synthesis() -> None:
    provider = DecisionProvider(AgentDecisionMode.RESEARCH, roles=("stock",))

    with pytest.raises(AgentRuntimeError, match="invalid decision"):
        asyncio.run(
            AgentDecisionService(provider).decide(content="Research TEST TOOL OUTPUT company")
        )


def test_agent_runtime_rejects_provider_without_research_capabilities() -> None:
    provider = ScriptedAgentProvider(())
    registry = ProviderRegistry().register_chat_provider("scripted", provider)
    settings = Settings.from_env(
        {
            "FRA_LLM_PROVIDER": "scripted",
            "FRA_LLM_MODEL": "scripted-model",
        }
    )
    resolver = AgentRuntimeResolver(
        settings=lambda: settings,
        registry=lambda _current: registry,
    )

    with pytest.raises(AgentRuntimeError) as raised:
        resolver.resolve(require_research=True)

    assert raised.value.code == "agent_provider_incompatible"


def test_structured_agent_runs_allowlisted_tool_and_preserves_metadata() -> None:
    provider = ScriptedAgentProvider((_tool_call(), _agent_response(_valid_output())))
    runner = StructuredAgentRunner(provider, model="scripted-model")

    result = asyncio.run(
        runner.run(
            contract=_contract(),
            user_payload={"query": "TEST TOOL OUTPUT"},
            registry=_registry(),
            context=_context(),
            known_evidence_ids=(EVIDENCE_ID,),
        )
    )

    assert result.output["reasoning_summary"] == "Concise source-backed summary."
    assert result.tool_results[0].data["evidence_ids"] == (EVIDENCE_ID,)
    assert provider.requests[-1].metadata["prompt_id"] == "test.financial"
    assert provider.requests[-1].metadata["prompt_version"] == "1.0.0"


def test_structured_agent_preloads_required_tool_for_local_model() -> None:
    provider = ScriptedAgentProvider((_agent_response(_valid_output()),))

    result = asyncio.run(
        StructuredAgentRunner(provider).run(
            contract=_contract(),
            user_payload={
                "query": "TEST TOOL OUTPUT",
                "required_tool": "load_evidence",
            },
            registry=_registry(),
            context=_context(),
            known_evidence_ids=(EVIDENCE_ID,),
        )
    )

    payload = json.loads(provider.requests[0].messages[-1].content)

    assert payload["required_tool_result"]["data"]["evidence_ids"] == [EVIDENCE_ID]
    assert result.tool_results[0].tool_name == "load_evidence"
    assert len(provider.requests) == 1


def test_structured_agent_repairs_malformed_output_once() -> None:
    provider = ScriptedAgentProvider(
        (
            _tool_call(),
            ChatResponse(
                message=ChatMessage(role=MessageRole.ASSISTANT, content="{bad json"),
                provider="scripted",
                model="scripted-model",
            ),
            _agent_response(_valid_output()),
        )
    )

    result = asyncio.run(
        StructuredAgentRunner(provider).run(
            contract=_contract(),
            user_payload={"query": "TEST TOOL OUTPUT"},
            registry=_registry(),
            context=_context(),
            known_evidence_ids=(EVIDENCE_ID,),
        )
    )

    assert result.repaired is True
    assert provider.requests[-1].metadata["repair_attempt"] == "1"
    assert EVIDENCE_ID in provider.requests[-1].messages[-1].content


def test_structured_agent_rejects_unknown_evidence_after_repair() -> None:
    invalid = _valid_output(evidence_id="evidence:invented")
    provider = ScriptedAgentProvider(
        (_tool_call(), _agent_response(invalid), _agent_response(invalid))
    )

    with pytest.raises(AgentRuntimeError) as raised:
        asyncio.run(
            StructuredAgentRunner(provider).run(
                contract=_contract(),
                user_payload={"query": "TEST TOOL OUTPUT"},
                registry=_registry(),
                context=_context(),
                known_evidence_ids=(EVIDENCE_ID,),
            )
        )

    assert raised.value.code == "agent_output_invalid"
    assert len(provider.requests) == 3


def test_structured_agent_repairs_invalid_nested_schema_types() -> None:
    invalid = _valid_output()
    invalid["findings"] = ["not-an-object"]
    provider = ScriptedAgentProvider(
        (_tool_call(), _agent_response(invalid), _agent_response(_valid_output()))
    )

    result = asyncio.run(
        StructuredAgentRunner(provider).run(
            contract=_contract(),
            user_payload={"query": "TEST TOOL OUTPUT"},
            registry=_registry(),
            context=_context(),
            known_evidence_ids=(EVIDENCE_ID,),
        )
    )

    assert result.repaired is True
    assert provider.requests[-1].metadata["repair_attempt"] == "1"


def test_offline_provider_cannot_run_real_specialist() -> None:
    with pytest.raises(AgentRuntimeError) as raised:
        asyncio.run(
            StructuredAgentRunner(OfflineTestProvider()).run(
                contract=_contract(),
                user_payload={"query": "TEST TOOL OUTPUT"},
                registry=_registry(),
                context=_context(),
            )
        )

    assert raised.value.code == "agent_provider_unavailable"


@dataclass
class DecisionProvider:
    mode: AgentDecisionMode
    roles: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.requests: list[ChatRequest] = []

    @property
    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            provider="scripted",
            model="scripted-model",
            capabilities=(
                ProviderCapability.CHAT,
                ProviderCapability.TOOL_CALLS,
                ProviderCapability.STRUCTURED_OUTPUT,
            ),
        )

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        research = self.mode == AgentDecisionMode.RESEARCH
        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=""),
            provider="scripted",
            model="scripted-model",
            structured_output={
                "mode": self.mode.value,
                "answer": "Safe response." if not research else "",
                "company_query": "TEST TOOL OUTPUT COMPANY" if research else None,
                "specialist_roles": (
                    list(self.roles)
                    if self.roles
                    else (["financial-report", "stock", "context", "synthesis"] if research else [])
                ),
                "reasoning_summary": "Concise routing summary.",
            },
        )

    def stream_chat(self, _request: ChatRequest):
        raise NotImplementedError


@dataclass
class ScriptedAgentProvider:
    responses: tuple[ChatResponse, ...]

    def __post_init__(self) -> None:
        self.requests: list[ChatRequest] = []
        self._responses = list(self.responses)

    @property
    def metadata(self) -> ModelMetadata:
        return ModelMetadata(provider="scripted", model="scripted-model")

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        return self._responses.pop(0)

    def stream_chat(self, _request: ChatRequest):
        raise NotImplementedError


def _contract() -> PromptContract:
    return PromptContract(
        id="test.financial",
        role=AgentRole.FINANCIAL_REPORT_ANALYST,
        version=PromptVersion("1.0.0"),
        system_prompt="Use only tool evidence and return structured output.",
        description="Test specialist contract.",
        allowed_tools=("load_evidence",),
        output_schema=AgentOutputSchema(
            name="test_financial_output",
            schema={
                "type": "object",
                "properties": {
                    "agent_role": {"type": "string", "enum": ["financial_report_analyst"]},
                    "findings": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "statement": {"type": "string"},
                                "evidence_ids": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "minItems": 1,
                                },
                            },
                            "required": ["statement", "evidence_ids"],
                            "additionalProperties": False,
                        },
                    },
                    "reasoning_summary": {"type": "string"},
                },
                "required": ["agent_role", "findings", "reasoning_summary"],
                "additionalProperties": False,
            },
        ),
    )


def _valid_output(*, evidence_id: str = EVIDENCE_ID) -> dict[str, object]:
    return {
        "agent_role": "financial_report_analyst",
        "findings": [
            {
                "statement": "TEST TOOL OUTPUT source-backed finding.",
                "evidence_ids": [evidence_id],
            }
        ],
        "reasoning_summary": "Concise source-backed summary.",
    }


def _tool_call() -> ChatResponse:
    return ChatResponse(
        message=ChatMessage(role=MessageRole.ASSISTANT, content=""),
        provider="scripted",
        model="scripted-model",
        tool_calls=(ToolCall(id="call:evidence", name="load_evidence", arguments={}),),
    )


def _agent_response(output: Mapping[str, Any]) -> ChatResponse:
    return ChatResponse(
        message=ChatMessage(role=MessageRole.ASSISTANT, content=""),
        provider="scripted",
        model="scripted-model",
        structured_output=output,
    )


def _registry() -> ToolRegistry:
    async def load_evidence(
        _context: ToolContext,
        _arguments: Mapping[str, Any],
    ) -> ToolResult:
        return ToolResult.succeeded(
            tool_call_id="placeholder",
            tool_name="load_evidence",
            data={"evidence_ids": [EVIDENCE_ID]},
        )

    return ToolRegistry(
        (
            ToolSpec(
                name="load_evidence",
                description="Load bounded test evidence.",
                input_schema={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                permissions=(ToolPermission.FINANCIAL_DATA,),
                handler=load_evidence,
            ),
        )
    )


def _context() -> ToolContext:
    return ToolContext(
        allowed_permissions=(ToolPermission.FINANCIAL_DATA,),
        allowed_tools=("load_evidence",),
    )
