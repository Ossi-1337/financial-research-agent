from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from financial_research_agent.agents.runtime import AgentRuntimeError
from financial_research_agent.llm import ChatProvider, ProviderCapability, ProviderError
from financial_research_agent.llm.registry import ProviderRegistry, create_default_provider_registry
from financial_research_agent.settings import ProviderTask, Settings

_RESEARCH_CAPABILITIES = (
    ProviderCapability.CHAT,
    ProviderCapability.TOOL_CALLS,
    ProviderCapability.STRUCTURED_OUTPUT,
)


@dataclass(frozen=True, slots=True)
class AgentRuntimeSelection:
    provider: ChatProvider
    model: str
    max_output_tokens: int

    @property
    def provider_name(self) -> str:
        return self.provider.metadata.provider


class AgentRuntimeResolver:
    """Resolves current provider settings at request time."""

    def __init__(
        self,
        *,
        settings: Callable[[], Settings],
        registry: Callable[[Settings], ProviderRegistry] | None = None,
    ) -> None:
        self._settings = settings
        self._registry = registry or (
            lambda current: create_default_provider_registry(current.provider)
        )

    def resolve(
        self,
        *,
        task: ProviderTask | str | None = None,
        require_research: bool = False,
    ) -> AgentRuntimeSelection:
        settings = self._settings()
        if task is None:
            provider_name = settings.provider.llm_provider
            model = settings.provider.llm_model
        else:
            task_selection = settings.provider.selection_for_task(task)
            provider_name = task_selection.provider
            model = task_selection.model
        return self.resolve_selection(
            provider_name=provider_name,
            model=model,
            require_research=require_research,
            settings=settings,
        )

    def resolve_selection(
        self,
        *,
        provider_name: str,
        model: str,
        require_research: bool = False,
        settings: Settings | None = None,
    ) -> AgentRuntimeSelection:
        current = settings or self._settings()
        try:
            provider = self._registry(current).chat_provider(provider_name)
        except ProviderError as exc:
            raise AgentRuntimeError(
                code="agent_provider_unavailable",
                message="Configured agent provider is unavailable.",
            ) from exc
        selection = AgentRuntimeSelection(
            provider=provider,
            model=model,
            max_output_tokens=current.performance.agent_max_output_tokens,
        )
        if require_research:
            self.validate_research(selection, settings=current)
        return selection

    def validate_research(
        self,
        selection: AgentRuntimeSelection,
        *,
        settings: Settings | None = None,
    ) -> None:
        current = settings or self._settings()
        if selection.provider_name == "offline-test":
            raise AgentRuntimeError(
                code="agent_provider_unavailable",
                message="Real research requires a configured local or hosted LLM provider.",
            )
        if not _credentials_configured(selection.provider_name, current):
            raise AgentRuntimeError(
                code="agent_provider_unavailable",
                message="Configured agent provider credentials are unavailable.",
            )
        missing = tuple(
            capability.value
            for capability in _RESEARCH_CAPABILITIES
            if not selection.provider.metadata.supports(capability)
        )
        if missing:
            raise AgentRuntimeError(
                code="agent_provider_incompatible",
                message=(
                    f"Configured agent provider lacks required capabilities: {', '.join(missing)}."
                ),
            )


def _credentials_configured(provider: str, settings: Settings) -> bool:
    if provider == "openai":
        return settings.provider.openai_api_key is not None
    if provider == "anthropic":
        return settings.provider.anthropic_api_key is not None
    if provider == "gemini":
        return settings.provider.gemini_api_key is not None
    return True
