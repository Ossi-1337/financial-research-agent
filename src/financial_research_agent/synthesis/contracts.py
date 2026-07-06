from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

NO_SYNTHESIS_RECOMMENDATION_NOTICE = (
    "This synthesis is a source-backed research summary only and does not provide buy, "
    "sell, hold, price-target, or personalized investment advice."
)


class SynthesisReportStatus(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    NO_DATA = "no_data"


class SynthesisSection(StrEnum):
    CURRENT_SITUATION = "current_situation"
    STRENGTHS = "strengths"
    WEAKNESSES = "weaknesses"
    OPPORTUNITIES = "opportunities"
    RISKS = "risks"
    UNKNOWNS = "unknowns"


class ScenarioDirection(StrEnum):
    UPSIDE = "upside"
    DOWNSIDE = "downside"


class EvidenceCoverage(StrEnum):
    STRONG = "strong"
    MODERATE = "moderate"
    LIMITED = "limited"
    NONE = "none"


class ConfidenceLabel(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class SynthesisPoint:
    id: str
    section: SynthesisSection
    title: str
    summary: str
    confidence: ConfidenceLabel
    evidence_ids: tuple[str, ...] = ()
    source_handoff_ids: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_text("id", self.id))
        object.__setattr__(self, "section", SynthesisSection(self.section))
        object.__setattr__(self, "title", _require_text("title", self.title))
        object.__setattr__(self, "summary", _require_text("summary", self.summary))
        object.__setattr__(self, "confidence", ConfidenceLabel(self.confidence))
        object.__setattr__(self, "evidence_ids", _text_tuple("evidence_ids", self.evidence_ids))
        object.__setattr__(
            self,
            "source_handoff_ids",
            _text_tuple("source_handoff_ids", self.source_handoff_ids),
        )
        object.__setattr__(self, "limitations", _text_tuple("limitations", self.limitations))
        if not self.evidence_ids and not self.source_handoff_ids and not self.limitations:
            raise ValueError("synthesis point must include evidence, handoff ids, or limitations")

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "section": self.section.value,
            "title": self.title,
            "summary": self.summary,
            "confidence": self.confidence.value,
            "evidence_ids": list(self.evidence_ids),
            "source_handoff_ids": list(self.source_handoff_ids),
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True, slots=True)
class SynthesisScenario:
    id: str
    direction: ScenarioDirection
    title: str
    condition: str
    potential_development: str
    confidence: ConfidenceLabel
    evidence_ids: tuple[str, ...] = ()
    source_handoff_ids: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_text("id", self.id))
        object.__setattr__(self, "direction", ScenarioDirection(self.direction))
        object.__setattr__(self, "title", _require_text("title", self.title))
        object.__setattr__(self, "condition", _conditional_text("condition", self.condition))
        object.__setattr__(
            self,
            "potential_development",
            _conditional_text("potential_development", self.potential_development),
        )
        object.__setattr__(self, "confidence", ConfidenceLabel(self.confidence))
        object.__setattr__(self, "evidence_ids", _text_tuple("evidence_ids", self.evidence_ids))
        object.__setattr__(
            self,
            "source_handoff_ids",
            _text_tuple("source_handoff_ids", self.source_handoff_ids),
        )
        object.__setattr__(self, "limitations", _text_tuple("limitations", self.limitations))
        if not self.evidence_ids and not self.source_handoff_ids and not self.limitations:
            raise ValueError("scenario must include evidence, handoff ids, or limitations")

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "direction": self.direction.value,
            "title": self.title,
            "condition": self.condition,
            "potential_development": self.potential_development,
            "confidence": self.confidence.value,
            "evidence_ids": list(self.evidence_ids),
            "source_handoff_ids": list(self.source_handoff_ids),
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True, slots=True)
class SynthesisReport:
    id: str
    query: str
    status: SynthesisReportStatus
    created_at: datetime
    current_situation: tuple[SynthesisPoint, ...]
    strengths: tuple[SynthesisPoint, ...]
    weaknesses: tuple[SynthesisPoint, ...]
    opportunities: tuple[SynthesisPoint, ...]
    risks: tuple[SynthesisPoint, ...]
    upside_scenario: SynthesisScenario
    downside_scenario: SynthesisScenario
    unknowns: tuple[SynthesisPoint, ...]
    overall_confidence: ConfidenceLabel
    evidence_coverage: EvidenceCoverage
    evidence_coverage_ratio: float
    company_name: str | None = None
    security_symbol: str | None = None
    warnings: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    no_recommendation_notice: str = NO_SYNTHESIS_RECOMMENDATION_NOTICE

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_text("id", self.id))
        object.__setattr__(self, "query", _require_text("query", self.query))
        object.__setattr__(self, "status", SynthesisReportStatus(self.status))
        object.__setattr__(self, "created_at", _aware_datetime("created_at", self.created_at))
        object.__setattr__(
            self,
            "current_situation",
            _point_tuple(
                "current_situation",
                self.current_situation,
                SynthesisSection.CURRENT_SITUATION,
            ),
        )
        object.__setattr__(
            self,
            "strengths",
            _point_tuple("strengths", self.strengths, SynthesisSection.STRENGTHS),
        )
        object.__setattr__(
            self,
            "weaknesses",
            _point_tuple("weaknesses", self.weaknesses, SynthesisSection.WEAKNESSES),
        )
        object.__setattr__(
            self,
            "opportunities",
            _point_tuple("opportunities", self.opportunities, SynthesisSection.OPPORTUNITIES),
        )
        object.__setattr__(
            self,
            "risks",
            _point_tuple("risks", self.risks, SynthesisSection.RISKS),
        )
        if not isinstance(self.upside_scenario, SynthesisScenario):
            raise ValueError("upside_scenario must be a SynthesisScenario")
        if self.upside_scenario.direction != ScenarioDirection.UPSIDE:
            raise ValueError("upside_scenario must have direction=upside")
        if not isinstance(self.downside_scenario, SynthesisScenario):
            raise ValueError("downside_scenario must be a SynthesisScenario")
        if self.downside_scenario.direction != ScenarioDirection.DOWNSIDE:
            raise ValueError("downside_scenario must have direction=downside")
        object.__setattr__(
            self,
            "unknowns",
            _point_tuple("unknowns", self.unknowns, SynthesisSection.UNKNOWNS),
        )
        object.__setattr__(self, "overall_confidence", ConfidenceLabel(self.overall_confidence))
        object.__setattr__(self, "evidence_coverage", EvidenceCoverage(self.evidence_coverage))
        if self.evidence_coverage_ratio < 0 or self.evidence_coverage_ratio > 1:
            raise ValueError("evidence_coverage_ratio must be between 0 and 1")
        object.__setattr__(self, "company_name", _optional_text(self.company_name))
        object.__setattr__(self, "security_symbol", _optional_upper_text(self.security_symbol))
        object.__setattr__(self, "warnings", _text_tuple("warnings", self.warnings))
        object.__setattr__(self, "limitations", _text_tuple("limitations", self.limitations))
        object.__setattr__(
            self,
            "no_recommendation_notice",
            _require_text("no_recommendation_notice", self.no_recommendation_notice),
        )

    @property
    def summary(self) -> str:
        subject = self.company_name or self.security_symbol or self.query
        return (
            f"Synthesis report for {subject}: status={self.status.value}, "
            f"evidence_coverage={self.evidence_coverage.value}, "
            f"confidence={self.overall_confidence.value}."
        )

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                (
                    *(evidence_id for point in self.points for evidence_id in point.evidence_ids),
                    *self.upside_scenario.evidence_ids,
                    *self.downside_scenario.evidence_ids,
                )
            )
        )

    @property
    def points(self) -> tuple[SynthesisPoint, ...]:
        return (
            *self.current_situation,
            *self.strengths,
            *self.weaknesses,
            *self.opportunities,
            *self.risks,
            *self.unknowns,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "query": self.query,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "company_name": self.company_name,
            "security_symbol": self.security_symbol,
            "summary": self.summary,
            "sections": {
                SynthesisSection.CURRENT_SITUATION.value: [
                    point.to_dict() for point in self.current_situation
                ],
                SynthesisSection.STRENGTHS.value: [point.to_dict() for point in self.strengths],
                SynthesisSection.WEAKNESSES.value: [point.to_dict() for point in self.weaknesses],
                SynthesisSection.OPPORTUNITIES.value: [
                    point.to_dict() for point in self.opportunities
                ],
                SynthesisSection.RISKS.value: [point.to_dict() for point in self.risks],
                SynthesisSection.UNKNOWNS.value: [point.to_dict() for point in self.unknowns],
            },
            "scenarios": {
                ScenarioDirection.UPSIDE.value: self.upside_scenario.to_dict(),
                ScenarioDirection.DOWNSIDE.value: self.downside_scenario.to_dict(),
            },
            "overall_confidence": self.overall_confidence.value,
            "evidence_coverage": self.evidence_coverage.value,
            "evidence_coverage_ratio": self.evidence_coverage_ratio,
            "evidence_ids": list(self.evidence_ids),
            "warnings": list(self.warnings),
            "limitations": list(self.limitations),
            "no_recommendation_notice": self.no_recommendation_notice,
        }


def _require_text(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    text = value.strip()
    if text == "":
        raise ValueError(f"{name} is required")
    return text


def _conditional_text(name: str, value: str) -> str:
    text = _require_text(name, value)
    lowered = text.casefold()
    if "if " not in lowered and "when " not in lowered:
        raise ValueError(f"{name} must use conditional language")
    return text


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    return text or None


def _optional_upper_text(value: str | None) -> str | None:
    text = _optional_text(value)
    return text.upper() if text is not None else None


def _text_tuple(name: str, values: Iterable[str]) -> tuple[str, ...]:
    if isinstance(values, str):
        raise ValueError(f"{name} must be an iterable of strings, not a string")
    return tuple(_require_text(f"{name}[{index}]", value) for index, value in enumerate(values))


def _point_tuple(
    name: str,
    values: Iterable[SynthesisPoint],
    section: SynthesisSection,
) -> tuple[SynthesisPoint, ...]:
    points = tuple(values)
    for index, point in enumerate(points):
        if not isinstance(point, SynthesisPoint):
            raise ValueError(f"{name}[{index}] must be a SynthesisPoint")
        if point.section != section:
            raise ValueError(f"{name}[{index}] must have section={section.value}")
    return points


def _aware_datetime(name: str, value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError(f"{name} must be a datetime")
    if value.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value
