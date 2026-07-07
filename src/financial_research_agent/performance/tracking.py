from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from enum import StrEnum
from time import perf_counter_ns

from financial_research_agent.llm import ChatProvider, ChatRequest, ChatResponse

MILLION = Decimal("1000000")


class ProviderCallKind(StrEnum):
    CHAT = "chat"
    STREAMING_CHAT = "streaming_chat"
    EMBEDDING = "embedding"


@dataclass(frozen=True, slots=True)
class ProviderRate:
    provider: str
    model_pattern: str
    input_cost_per_million_tokens_usd: Decimal
    output_cost_per_million_tokens_usd: Decimal
    source: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider", _require_text("provider", self.provider))
        object.__setattr__(
            self,
            "model_pattern",
            _require_text("model_pattern", self.model_pattern),
        )
        object.__setattr__(
            self,
            "input_cost_per_million_tokens_usd",
            _decimal_rate(
                "input_cost_per_million_tokens_usd",
                self.input_cost_per_million_tokens_usd,
            ),
        )
        object.__setattr__(
            self,
            "output_cost_per_million_tokens_usd",
            _decimal_rate(
                "output_cost_per_million_tokens_usd",
                self.output_cost_per_million_tokens_usd,
            ),
        )
        object.__setattr__(self, "source", _require_text("source", self.source))

    def matches(self, *, provider: str, model: str) -> bool:
        if self.provider != provider:
            return False
        pattern = self.model_pattern
        return pattern == "*" or model == pattern or model.startswith(pattern.rstrip("*"))


@dataclass(frozen=True, slots=True)
class ProviderCallMetrics:
    call_kind: ProviderCallKind
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: int
    estimated_cost_usd: str | None = None
    cost_source: str = "not_estimated"
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "call_kind", ProviderCallKind(self.call_kind))
        object.__setattr__(self, "provider", _require_text("provider", self.provider))
        object.__setattr__(self, "model", _require_text("model", self.model))
        for name in ("input_tokens", "output_tokens", "total_tokens", "latency_ms"):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        object.__setattr__(self, "cost_source", _require_text("cost_source", self.cost_source))
        object.__setattr__(self, "warnings", _text_tuple(self.warnings))

    def to_dict(self) -> dict[str, object]:
        return {
            "call_kind": self.call_kind.value,
            "provider": self.provider,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": self.latency_ms,
            "estimated_cost_usd": self.estimated_cost_usd,
            "cost_source": self.cost_source,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class ProviderCallResult[T]:
    value: T
    metrics: ProviderCallMetrics


def default_provider_rates() -> tuple[ProviderRate, ...]:
    zero = Decimal("0")
    return (
        ProviderRate(
            provider="offline-test",
            model_pattern="*",
            input_cost_per_million_tokens_usd=zero,
            output_cost_per_million_tokens_usd=zero,
            source="deterministic_offline_provider",
        ),
        ProviderRate(
            provider="local-openai",
            model_pattern="*",
            input_cost_per_million_tokens_usd=zero,
            output_cost_per_million_tokens_usd=zero,
            source="local_runtime_excludes_electricity_and_hardware_cost",
        ),
    )


def estimate_provider_cost_usd(
    *,
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    rates: Iterable[ProviderRate] | None = None,
) -> tuple[str | None, str, tuple[str, ...]]:
    for rate in rates or default_provider_rates():
        if not rate.matches(provider=provider, model=model):
            continue
        cost = (
            Decimal(input_tokens) / MILLION * rate.input_cost_per_million_tokens_usd
            + Decimal(output_tokens) / MILLION * rate.output_cost_per_million_tokens_usd
        )
        return _money(cost), rate.source, ()
    return (
        None,
        "not_estimated",
        (
            "No local rate card is configured for this provider/model; usage is tracked "
            "without a dollar estimate.",
        ),
    )


async def measured_chat(
    provider: ChatProvider,
    request: ChatRequest,
    *,
    call_kind: ProviderCallKind = ProviderCallKind.CHAT,
) -> ProviderCallResult[ChatResponse]:
    started_ns = perf_counter_ns()
    response = await provider.chat(request)
    completed_ns = perf_counter_ns()
    return ProviderCallResult(
        value=response,
        metrics=call_metrics_from_response(
            response,
            call_kind=call_kind,
            started_ns=started_ns,
            completed_ns=completed_ns,
        ),
    )


def call_metrics_from_response(
    response: ChatResponse,
    *,
    call_kind: ProviderCallKind,
    started_ns: int,
    completed_ns: int,
) -> ProviderCallMetrics:
    latency_ms = max(0, int((completed_ns - started_ns) / 1_000_000))
    estimated_cost, cost_source, warnings = estimate_provider_cost_usd(
        provider=response.provider,
        model=response.model,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )
    return ProviderCallMetrics(
        call_kind=call_kind,
        provider=response.provider,
        model=response.model,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        total_tokens=response.usage.total_tokens,
        latency_ms=latency_ms,
        estimated_cost_usd=estimated_cost,
        cost_source=cost_source,
        warnings=warnings,
    )


def _money(value: Decimal) -> str:
    return str(value.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP))


def _decimal_rate(name: str, value: Decimal) -> Decimal:
    result = Decimal(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _require_text(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    text = value.strip()
    if text == "":
        raise ValueError(f"{name} is required")
    return text


def _text_tuple(values: Iterable[str]) -> tuple[str, ...]:
    if isinstance(values, str):
        raise ValueError("value must be an iterable of strings, not a string")
    return tuple(_require_text(f"value[{index}]", value) for index, value in enumerate(values))
