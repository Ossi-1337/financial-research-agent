from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterable, Mapping
from dataclasses import dataclass, replace

from financial_research_agent.agents import (
    AgentDecision,
    AgentDecisionMode,
    AgentDecisionService,
    AgentRuntimeResolver,
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
from financial_research_agent.settings import ProviderTask, Settings

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
        agent_runtime: AgentRuntimeResolver | None = None,
    ) -> None:
        self._settings = settings
        self._registry = registry
        self._agent_runtime = agent_runtime or AgentRuntimeResolver(
            settings=settings,
            registry=lambda _current: registry(),
        )

    async def plan(
        self,
        *,
        content: str,
        context_messages: Iterable[ChatMessage],
        company_references: Iterable[Mapping[str, object]],
    ) -> ConversationPlan:
        references = tuple(company_references)
        runtime = self._agent_runtime.resolve()
        decision = await AgentDecisionService(runtime.provider, model=runtime.model).decide(
            content=content,
            context_messages=context_messages,
            company_references=references,
        )
        decision = _canonicalize_research_company_query(decision, references)
        return ConversationPlan(
            decision=decision,
            provider=runtime.provider,
            model=runtime.model,
        )

    def ensure_research_available(self, plan: ConversationPlan) -> None:
        if plan.decision.mode == AgentDecisionMode.RESEARCH:
            self._agent_runtime.resolve_selection(
                provider_name=plan.provider.metadata.provider,
                model=plan.model,
                require_research=True,
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
        runtime = self._agent_runtime.resolve(task=ProviderTask.CHAT)
        return await runtime.provider.chat(
            self._direct_request(
                plan,
                model=runtime.model,
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
        runtime = self._agent_runtime.resolve(task=ProviderTask.STREAMING)
        return runtime.provider.stream_chat(
            self._direct_request(
                plan,
                model=runtime.model,
                content=content,
                context_messages=context_messages,
                company_references=company_references,
            )
        )

    def _direct_request(
        self,
        plan: ConversationPlan,
        *,
        model: str,
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
            model=model,
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


def _canonicalize_research_company_query(
    decision: AgentDecision,
    references: tuple[Mapping[str, object], ...],
) -> AgentDecision:
    if decision.mode != AgentDecisionMode.RESEARCH or not references:
        return decision

    reference = _matching_company_reference(decision.company_query, references)
    if reference is None and len(references) == 1:
        reference = references[0]
    if reference is None:
        return decision

    lookup_query = _company_reference_lookup_query(reference)
    if not lookup_query or lookup_query == decision.company_query:
        return decision
    return replace(decision, company_query=lookup_query)


def _matching_company_reference(
    company_query: str | None,
    references: tuple[Mapping[str, object], ...],
) -> Mapping[str, object] | None:
    normalized_query = _normalize_company_reference_value(company_query)
    if not normalized_query:
        return None
    for reference in references:
        values = (
            reference.get("id"),
            reference.get("company_id"),
            reference.get("legal_name"),
            reference.get("ticker"),
            reference.get("label"),
            reference.get("cik"),
        )
        if normalized_query in {
            normalized
            for value in values
            if (normalized := _normalize_company_reference_value(value))
        }:
            return reference
    return None


def _company_reference_lookup_query(reference: Mapping[str, object]) -> str | None:
    for key in ("legal_name", "ticker", "label"):
        value = reference.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _normalize_company_reference_value(value: object) -> str:
    normalized = str(value or "").strip().casefold()
    return normalized.removeprefix("@")
