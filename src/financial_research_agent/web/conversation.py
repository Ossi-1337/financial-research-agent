from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterable, Mapping
from dataclasses import dataclass

from financial_research_agent.agents import (
    AgentDecision,
    AgentDecisionMode,
    AgentDecisionService,
    AgentRuntimeError,
)
from financial_research_agent.llm import (
    ChatMessage,
    ChatProvider,
    ChatRequest,
    ChatResponse,
    MessageRole,
    StreamEvent,
)
from financial_research_agent.llm.registry import ProviderRegistry
from financial_research_agent.settings import Settings

DIRECT_CHAT_PROMPT = """
You are the conversational entrypoint for a local financial research system.
Answer general and conceptual questions directly. Company-specific current research is handled
by specialist agents before this direct-answer path is used. Do not invent current prices,
financial facts, source URLs, or citations. Do not provide buy, sell, hold, price-target, or
personalized investment advice.
""".strip()


@dataclass(frozen=True, slots=True)
class ConversationPlan:
    decision: AgentDecision
    provider: ChatProvider
    model: str


class AgentConversationService:
    def __init__(
        self,
        *,
        settings: Callable[[], Settings],
        registry: Callable[[], ProviderRegistry],
    ) -> None:
        self._settings = settings
        self._registry = registry

    async def plan(
        self,
        *,
        content: str,
        context_messages: Iterable[ChatMessage],
        company_references: Iterable[Mapping[str, object]],
    ) -> ConversationPlan:
        settings = self._settings()
        provider_name = settings.provider.llm_provider
        model = settings.provider.llm_model
        provider = self._registry().chat_provider(provider_name)
        decision = await AgentDecisionService(provider, model=model).decide(
            content=content,
            context_messages=context_messages,
            company_references=company_references,
        )
        return ConversationPlan(decision=decision, provider=provider, model=model)

    def ensure_research_available(self, plan: ConversationPlan) -> None:
        if (
            plan.decision.mode == AgentDecisionMode.RESEARCH
            and plan.provider.metadata.provider == "offline-test"
        ):
            raise AgentRuntimeError(
                code="agent_provider_unavailable",
                message="Real research requires a configured local or hosted LLM provider.",
            )

    async def direct_response(
        self,
        plan: ConversationPlan,
        *,
        content: str,
        context_messages: Iterable[ChatMessage],
        company_references: Iterable[Mapping[str, object]],
    ) -> ChatResponse:
        if plan.decision.mode != AgentDecisionMode.DIRECT_ANSWER:
            return _decision_response(plan)
        return await plan.provider.chat(
            self._direct_request(
                plan,
                content=content,
                context_messages=context_messages,
                company_references=company_references,
            )
        )

    def stream_direct_response(
        self,
        plan: ConversationPlan,
        *,
        content: str,
        context_messages: Iterable[ChatMessage],
        company_references: Iterable[Mapping[str, object]],
    ) -> AsyncIterator[StreamEvent]:
        if plan.decision.mode != AgentDecisionMode.DIRECT_ANSWER:
            raise ValueError("stream_direct_response requires direct_answer decision")
        return plan.provider.stream_chat(
            self._direct_request(
                plan,
                content=content,
                context_messages=context_messages,
                company_references=company_references,
            )
        )

    def _direct_request(
        self,
        plan: ConversationPlan,
        *,
        content: str,
        context_messages: Iterable[ChatMessage],
        company_references: Iterable[Mapping[str, object]],
    ) -> ChatRequest:
        references = tuple(company_references)
        reference_note = ""
        if references:
            labels = ", ".join(
                str(item.get("legal_name") or item.get("ticker") or "unknown")
                for item in references
            )
            reference_note = (
                "\nResolved company identifiers are context only, not financial evidence: "
                f"{labels}."
            )
        return ChatRequest(
            messages=(
                ChatMessage(role=MessageRole.SYSTEM, content=DIRECT_CHAT_PROMPT),
                *tuple(context_messages),
                ChatMessage(role=MessageRole.USER, content=f"{content}{reference_note}"),
            ),
            model=plan.model,
            max_output_tokens=self._settings().performance.prompt_budget_output_tokens,
            metadata={"agent_role": "orchestrator", "decision": plan.decision.mode.value},
        )


def _decision_response(plan: ConversationPlan) -> ChatResponse:
    content = plan.decision.answer or (
        "Please clarify the company or research question."
        if plan.decision.mode == AgentDecisionMode.CLARIFICATION
        else "The request cannot be completed within the research safety policy."
    )
    return ChatResponse(
        message=ChatMessage(role=MessageRole.ASSISTANT, content=content),
        provider=plan.provider.metadata.provider,
        model=plan.model,
    )
