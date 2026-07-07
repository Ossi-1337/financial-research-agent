from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

from financial_research_agent.llm import ChatRequest

DEFAULT_CHARS_PER_TOKEN = 4


@dataclass(frozen=True, slots=True)
class BudgetCheck:
    budget_name: str
    estimated_input_tokens: int
    max_input_tokens: int
    recommended_max_output_tokens: int
    over_budget: bool
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "budget_name": self.budget_name,
            "estimated_input_tokens": self.estimated_input_tokens,
            "max_input_tokens": self.max_input_tokens,
            "recommended_max_output_tokens": self.recommended_max_output_tokens,
            "over_budget": self.over_budget,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class PromptBudget:
    name: str
    max_input_tokens: int
    max_output_tokens: int
    reserved_output_tokens: int = 0
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _require_text("name", self.name))
        if self.max_input_tokens <= 0:
            raise ValueError("max_input_tokens must be positive")
        if self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        if self.reserved_output_tokens < 0:
            raise ValueError("reserved_output_tokens must be non-negative")
        if self.reserved_output_tokens >= self.max_output_tokens:
            raise ValueError("reserved_output_tokens must be lower than max_output_tokens")
        object.__setattr__(self, "notes", _text_tuple(self.notes))

    @property
    def recommended_max_output_tokens(self) -> int:
        return self.max_output_tokens - self.reserved_output_tokens

    def check(self, request: ChatRequest) -> BudgetCheck:
        return check_chat_request_budget(request, self)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "max_input_tokens": self.max_input_tokens,
            "max_output_tokens": self.max_output_tokens,
            "reserved_output_tokens": self.reserved_output_tokens,
            "recommended_max_output_tokens": self.recommended_max_output_tokens,
            "notes": list(self.notes),
        }


def default_prompt_budgets() -> Mapping[str, PromptBudget]:
    return MappingProxyType(
        {
            "chat": PromptBudget(
                name="chat",
                max_input_tokens=16_000,
                max_output_tokens=1_024,
                reserved_output_tokens=128,
                notes=("Direct chat keeps room for short assistant answers on local models.",),
            ),
            "cited_answer": PromptBudget(
                name="cited_answer",
                max_input_tokens=24_000,
                max_output_tokens=1_500,
                reserved_output_tokens=200,
                notes=("Cited answers reserve space for citation markers and limitations.",),
            ),
            "orchestrator": PromptBudget(
                name="orchestrator",
                max_input_tokens=32_000,
                max_output_tokens=2_000,
                reserved_output_tokens=256,
                notes=("Specialist handoffs should stay bounded before synthesis.",),
            ),
        }
    )


def prompt_budgets_for_limits(
    *,
    max_input_tokens: int,
    max_output_tokens: int,
) -> Mapping[str, PromptBudget]:
    return MappingProxyType(
        {
            name: replace(
                budget,
                max_input_tokens=max_input_tokens,
                max_output_tokens=max_output_tokens,
                reserved_output_tokens=min(
                    budget.reserved_output_tokens,
                    max_output_tokens - 1,
                ),
            )
            for name, budget in default_prompt_budgets().items()
        }
    )


def estimate_text_tokens(text: str, *, chars_per_token: int = DEFAULT_CHARS_PER_TOKEN) -> int:
    if chars_per_token <= 0:
        raise ValueError("chars_per_token must be positive")
    if text == "":
        return 0
    return max(1, (len(text) + chars_per_token - 1) // chars_per_token)


def estimate_chat_request_tokens(request: ChatRequest) -> int:
    return sum(estimate_text_tokens(message.content) for message in request.messages)


def check_chat_request_budget(request: ChatRequest, budget: PromptBudget) -> BudgetCheck:
    estimated = estimate_chat_request_tokens(request)
    over_budget = estimated > budget.max_input_tokens
    warnings: tuple[str, ...] = ()
    if over_budget:
        warnings = (
            (
                f"Estimated input tokens {estimated} exceed budget "
                f"{budget.max_input_tokens} for {budget.name}."
            ),
        )
    return BudgetCheck(
        budget_name=budget.name,
        estimated_input_tokens=estimated,
        max_input_tokens=budget.max_input_tokens,
        recommended_max_output_tokens=budget.recommended_max_output_tokens,
        over_budget=over_budget,
        warnings=warnings,
    )


def _require_text(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    text = value.strip()
    if text == "":
        raise ValueError(f"{name} is required")
    return text


def _text_tuple(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(_require_text(f"value[{index}]", value) for index, value in enumerate(values))
