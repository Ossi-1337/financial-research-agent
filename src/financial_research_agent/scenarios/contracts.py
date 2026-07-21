from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

from financial_research_agent.orchestration import OrchestratedResearchRun
from financial_research_agent.report_exports import ReportExportSnapshot

_SCENARIO_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_SEMANTIC_VERSION = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
_RESOURCE_NAME = re.compile(r"^[a-z0-9][a-z0-9_.-]*\.json$")


class ScenarioCheckStatus(StrEnum):
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"


class ScenarioExecutionStatus(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    FAILED = "failed"


class ScenarioErrorCode(StrEnum):
    UNKNOWN_SCENARIO = "unknown_scenario"
    INVALID_CONTEXT_SNAPSHOT = "invalid_context_snapshot"
    MISSING_MARKET_DATA_CREDENTIALS = "missing_market_data_credentials"
    INVALID_SEC_USER_AGENT = "invalid_sec_user_agent"
    SCENARIO_EXPORT_FAILED = "scenario_export_failed"
    LOCAL_QA_UNAVAILABLE = "local_qa_unavailable"


class ScenarioError(Exception):
    def __init__(self, code: ScenarioErrorCode | str, message: str) -> None:
        self.code = ScenarioErrorCode(code)
        self.message = _require_text("message", message)
        super().__init__(self.message)

    def to_dict(self) -> dict[str, str]:
        return {"error": "scenario_error", "code": self.code.value, "message": self.message}


@dataclass(frozen=True, slots=True)
class ScenarioDefinition:
    id: str
    version: str
    query: str
    expected_cik: str
    preferred_ticker: str
    preferred_exchange: str
    fiscal_years: int
    filing_form_limits: Mapping[str, int]
    market_outputsize: str
    benchmark_symbol: str
    context_resource: str

    def __post_init__(self) -> None:
        for name in ("id", "version", "query", "preferred_exchange", "context_resource"):
            object.__setattr__(self, name, _require_text(name, getattr(self, name)))
        if not _SCENARIO_ID.fullmatch(self.id):
            raise ValueError("id must be a lowercase kebab-case identifier")
        if not _SEMANTIC_VERSION.fullmatch(self.version):
            raise ValueError("version must use semantic major.minor.patch format")
        if not _RESOURCE_NAME.fullmatch(self.context_resource):
            raise ValueError("context_resource must be a safe JSON resource name")
        cik = _require_text("expected_cik", self.expected_cik)
        if not cik.isdigit() or len(cik) != 10:
            raise ValueError("expected_cik must contain exactly 10 digits")
        object.__setattr__(self, "expected_cik", cik)
        object.__setattr__(self, "preferred_ticker", _upper_text(self.preferred_ticker))
        object.__setattr__(self, "benchmark_symbol", _upper_text(self.benchmark_symbol))
        if self.fiscal_years <= 0:
            raise ValueError("fiscal_years must be positive")
        if self.market_outputsize not in {"compact", "full"}:
            raise ValueError("market_outputsize must be compact or full")
        object.__setattr__(
            self,
            "filing_form_limits",
            _positive_int_mapping("filing_form_limits", self.filing_form_limits),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "version": self.version,
            "query": self.query,
            "expected_cik": self.expected_cik,
            "preferred_ticker": self.preferred_ticker,
            "preferred_exchange": self.preferred_exchange,
            "fiscal_years": self.fiscal_years,
            "filing_form_limits": dict(self.filing_form_limits),
            "market_outputsize": self.market_outputsize,
            "benchmark_symbol": self.benchmark_symbol,
            "context_resource": self.context_resource,
        }


@dataclass(frozen=True, slots=True)
class ScenarioCheck:
    id: str
    status: ScenarioCheckStatus
    message: str
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_text("id", self.id))
        object.__setattr__(self, "status", ScenarioCheckStatus(self.status))
        object.__setattr__(self, "message", _require_text("message", self.message))
        if not isinstance(self.details, Mapping):
            raise ValueError("details must be a mapping")
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "status": self.status.value,
            "message": self.message,
            "details": dict(self.details),
        }


@dataclass(frozen=True, slots=True)
class ScenarioLocalQA:
    status: ScenarioCheckStatus
    answer: str
    provider: str
    model: str
    source_markers: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", ScenarioCheckStatus(self.status))
        for name in ("answer", "provider", "model"):
            object.__setattr__(self, name, _require_text(name, getattr(self, name)))
        object.__setattr__(
            self,
            "source_markers",
            _text_tuple("source_markers", self.source_markers),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "answer": self.answer,
            "provider": self.provider,
            "model": self.model,
            "source_markers": list(self.source_markers),
            "generation_method": "llm_source_bounded",
        }


@dataclass(frozen=True, slots=True)
class ScenarioExecutionResult:
    scenario: ScenarioDefinition
    status: ScenarioExecutionStatus
    run: OrchestratedResearchRun
    checks: tuple[ScenarioCheck, ...]
    export: ReportExportSnapshot | None = None
    local_qa: ScenarioLocalQA | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.scenario, ScenarioDefinition):
            raise ValueError("scenario must be a ScenarioDefinition")
        object.__setattr__(self, "status", ScenarioExecutionStatus(self.status))
        if not isinstance(self.run, OrchestratedResearchRun):
            raise ValueError("run must be an OrchestratedResearchRun")
        checks = tuple(self.checks)
        if any(not isinstance(check, ScenarioCheck) for check in checks):
            raise ValueError("checks must contain ScenarioCheck values")
        object.__setattr__(self, "checks", checks)
        if self.export is not None and not isinstance(self.export, ReportExportSnapshot):
            raise ValueError("export must be a ReportExportSnapshot")
        if self.local_qa is not None and not isinstance(self.local_qa, ScenarioLocalQA):
            raise ValueError("local_qa must be a ScenarioLocalQA")

    def to_dict(self) -> dict[str, object]:
        return {
            "scenario": self.scenario.to_dict(),
            "status": self.status.value,
            "run": self.run.to_dict(),
            "checks": [check.to_dict() for check in self.checks],
            "export": self.export.to_dict() if self.export is not None else None,
            "local_qa": self.local_qa.to_dict() if self.local_qa is not None else None,
        }


class ScenarioCatalog:
    def __init__(self, definitions: Iterable[ScenarioDefinition] = ()) -> None:
        items = tuple(definitions)
        by_id = {item.id: item for item in items}
        if len(by_id) != len(items):
            raise ValueError("scenario ids must be unique")
        self._by_id = MappingProxyType(by_id)

    def get(self, scenario_id: str) -> ScenarioDefinition:
        normalized = _require_text("scenario_id", scenario_id)
        try:
            return self._by_id[normalized]
        except KeyError as exc:
            raise ScenarioError(
                ScenarioErrorCode.UNKNOWN_SCENARIO,
                f"Unknown scenario: {normalized}",
            ) from exc

    def list(self) -> tuple[ScenarioDefinition, ...]:
        return tuple(self._by_id.values())


def _require_text(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    text = value.strip()
    if not text:
        raise ValueError(f"{name} is required")
    return text


def _upper_text(value: str) -> str:
    return _require_text("value", value).upper()


def _text_tuple(name: str, values: Iterable[str]) -> tuple[str, ...]:
    if isinstance(values, str):
        raise ValueError(f"{name} must be an iterable of strings")
    return tuple(_require_text(f"{name}[{index}]", item) for index, item in enumerate(values))


def _positive_int_mapping(name: str, values: Mapping[str, int]) -> Mapping[str, int]:
    if not isinstance(values, Mapping):
        raise ValueError(f"{name} must be a mapping")
    normalized: dict[str, int] = {}
    for key, value in values.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name}[{key!r}] must be a positive integer")
        normalized[_upper_text(key)] = value
    return MappingProxyType(normalized)
