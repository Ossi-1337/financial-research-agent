from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Self

REPORT_EXPORT_MANIFEST_VERSION = 1
REPORT_EXPORT_CONTENT_VERSION = 3
MAX_SOURCE_QUOTE_CHARS = 280
_SAFE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,159}$")
_MARKER_PATTERN = re.compile(r"^\[S[1-9][0-9]*\]$")


class ReportExportFormat(StrEnum):
    MARKDOWN = "markdown"
    HTML = "html"
    PDF = "pdf"

    @property
    def suffix(self) -> str:
        return {
            ReportExportFormat.MARKDOWN: ".md",
            ReportExportFormat.HTML: ".html",
            ReportExportFormat.PDF: ".pdf",
        }[self]

    @property
    def mime_type(self) -> str:
        return {
            ReportExportFormat.MARKDOWN: "text/markdown; charset=utf-8",
            ReportExportFormat.HTML: "text/html; charset=utf-8",
            ReportExportFormat.PDF: "application/pdf",
        }[self]


class ReportExportErrorCode(StrEnum):
    SYNTHESIS_REPORT_UNAVAILABLE = "synthesis_report_unavailable"
    REPORT_EXPORT_NOT_FOUND = "report_export_not_found"
    REPORT_EXPORT_ARTIFACT_NOT_FOUND = "report_export_artifact_not_found"
    REPORT_EXPORT_FAILED = "report_export_failed"


class ReportExportError(Exception):
    def __init__(self, code: ReportExportErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = ReportExportErrorCode(code)
        self.message = _require_text("message", message)


@dataclass(frozen=True, slots=True)
class ReportSourceReference:
    marker: str
    evidence_ids: tuple[str, ...]
    resolved: bool
    source_url: str | None = None
    source_name: str | None = None
    source_date: str | None = None
    retrieved_at: str | None = None
    section: str | None = None
    quote: str | None = None

    def __post_init__(self) -> None:
        if not _MARKER_PATTERN.fullmatch(self.marker):
            raise ValueError("marker must use the [S1] export marker format")
        object.__setattr__(self, "evidence_ids", _text_tuple("evidence_ids", self.evidence_ids))
        object.__setattr__(self, "source_url", _optional_text(self.source_url))
        object.__setattr__(self, "source_name", _optional_text(self.source_name))
        object.__setattr__(self, "source_date", _optional_text(self.source_date))
        object.__setattr__(self, "retrieved_at", _optional_text(self.retrieved_at))
        object.__setattr__(self, "section", _optional_text(self.section))
        object.__setattr__(
            self,
            "quote",
            _bounded_optional_text(self.quote, MAX_SOURCE_QUOTE_CHARS),
        )
        if self.resolved and self.source_url is None:
            raise ValueError("resolved source references require source_url")
        if not self.resolved and self.source_url is not None:
            raise ValueError("unresolved source references cannot include source_url")

    def to_dict(self) -> dict[str, object]:
        return {
            "marker": self.marker,
            "evidence_ids": list(self.evidence_ids),
            "resolved": self.resolved,
            "source_url": self.source_url,
            "source_name": self.source_name,
            "source_date": self.source_date,
            "retrieved_at": self.retrieved_at,
            "section": self.section,
            "quote": self.quote,
        }


@dataclass(frozen=True, slots=True)
class ReportEvidenceIndex:
    sources: tuple[ReportSourceReference, ...]
    unresolved_evidence_ids: tuple[str, ...]
    evidence_markers: Mapping[str, tuple[str, ...]]
    handoff_markers: Mapping[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "sources", _source_tuple(self.sources))
        object.__setattr__(
            self,
            "unresolved_evidence_ids",
            _text_tuple("unresolved_evidence_ids", self.unresolved_evidence_ids),
        )
        object.__setattr__(
            self,
            "evidence_markers",
            _marker_mapping("evidence_markers", self.evidence_markers),
        )
        object.__setattr__(
            self,
            "handoff_markers",
            _marker_mapping("handoff_markers", self.handoff_markers),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "sources": [source.to_dict() for source in self.sources],
            "unresolved_evidence_ids": list(self.unresolved_evidence_ids),
            "evidence_markers": {
                key: list(markers) for key, markers in self.evidence_markers.items()
            },
            "handoff_markers": {
                key: list(markers) for key, markers in self.handoff_markers.items()
            },
        }


@dataclass(frozen=True, slots=True)
class ReportExportPoint:
    id: str
    title: str
    summary: str
    confidence: str
    evidence_ids: tuple[str, ...] = ()
    source_markers: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_text("id", self.id))
        object.__setattr__(self, "title", _require_text("title", self.title))
        object.__setattr__(self, "summary", _require_text("summary", self.summary))
        object.__setattr__(self, "confidence", _require_text("confidence", self.confidence))
        object.__setattr__(self, "evidence_ids", _text_tuple("evidence_ids", self.evidence_ids))
        object.__setattr__(
            self,
            "source_markers",
            _marker_tuple("source_markers", self.source_markers),
        )
        object.__setattr__(self, "limitations", _text_tuple("limitations", self.limitations))


@dataclass(frozen=True, slots=True)
class ReportExportScenario:
    title: str
    condition: str
    potential_development: str
    confidence: str
    evidence_ids: tuple[str, ...] = ()
    source_markers: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "title", _require_text("title", self.title))
        object.__setattr__(self, "condition", _require_text("condition", self.condition))
        object.__setattr__(
            self,
            "potential_development",
            _require_text("potential_development", self.potential_development),
        )
        object.__setattr__(self, "confidence", _require_text("confidence", self.confidence))
        object.__setattr__(self, "evidence_ids", _text_tuple("evidence_ids", self.evidence_ids))
        object.__setattr__(
            self,
            "source_markers",
            _marker_tuple("source_markers", self.source_markers),
        )
        object.__setattr__(self, "limitations", _text_tuple("limitations", self.limitations))


@dataclass(frozen=True, slots=True)
class ReportExportNarrativeParagraph:
    text: str
    source_markers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", _require_text("text", self.text))
        object.__setattr__(
            self,
            "source_markers",
            _marker_tuple("source_markers", self.source_markers),
        )


@dataclass(frozen=True, slots=True)
class ReportExportNarrativeSection:
    name: str
    paragraphs: tuple[ReportExportNarrativeParagraph, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _require_text("name", self.name))
        paragraphs = tuple(self.paragraphs)
        if any(not isinstance(item, ReportExportNarrativeParagraph) for item in paragraphs):
            raise ValueError("paragraphs must contain ReportExportNarrativeParagraph values")
        object.__setattr__(self, "paragraphs", paragraphs)


@dataclass(frozen=True, slots=True)
class ReportExportChartPoint:
    priced_at: date
    indexed_value: Decimal

    def __post_init__(self) -> None:
        if not isinstance(self.priced_at, date) or isinstance(self.priced_at, datetime):
            raise ValueError("priced_at must be a date")
        try:
            value = Decimal(str(self.indexed_value))
        except (InvalidOperation, ValueError) as exc:
            raise ValueError("indexed_value must be a decimal") from exc
        if not value.is_finite() or value <= 0:
            raise ValueError("indexed_value must be finite and positive")
        object.__setattr__(self, "indexed_value", value)


@dataclass(frozen=True, slots=True)
class ReportExportChartSeries:
    symbol: str
    points: tuple[ReportExportChartPoint, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", _require_text("symbol", self.symbol).upper())
        points = tuple(self.points)
        if len(points) < 2:
            raise ValueError("chart series requires at least two points")
        if any(not isinstance(point, ReportExportChartPoint) for point in points):
            raise ValueError("points must contain ReportExportChartPoint values")
        dates = tuple(point.priced_at for point in points)
        if dates != tuple(sorted(set(dates))):
            raise ValueError("chart point dates must be unique and ascending")
        object.__setattr__(self, "points", points)


@dataclass(frozen=True, slots=True)
class ReportExportDocument:
    export_id: str
    run_id: str
    report_id: str
    generated_at: datetime
    run_created_at: datetime
    run_updated_at: datetime
    report_created_at: datetime
    query: str
    run_status: str
    report_status: str
    company_name: str | None
    company_id: str | None
    ticker: str | None
    security_id: str | None
    current_situation: tuple[ReportExportPoint, ...]
    strengths: tuple[ReportExportPoint, ...]
    weaknesses: tuple[ReportExportPoint, ...]
    opportunities: tuple[ReportExportPoint, ...]
    risks: tuple[ReportExportPoint, ...]
    unknowns: tuple[ReportExportPoint, ...]
    upside_scenario: ReportExportScenario
    downside_scenario: ReportExportScenario
    overall_confidence: str
    evidence_coverage: str
    evidence_coverage_ratio: float
    warnings: tuple[str, ...]
    limitations: tuple[str, ...]
    disclaimer: str
    sources: tuple[ReportSourceReference, ...]
    chart_series: tuple[ReportExportChartSeries, ...] = ()
    narrative_sections: tuple[ReportExportNarrativeSection, ...] = ()
    narrative_provider: str | None = None
    narrative_model: str | None = None
    narrative_synthesis_sha256: str | None = None
    generation_method: str = "deterministic"
    llm_provider: str = "not used"
    llm_model: str = "not used"

    def __post_init__(self) -> None:
        for name in ("export_id", "run_id", "report_id"):
            object.__setattr__(self, name, _safe_name(name, getattr(self, name)))
        for name in ("generated_at", "run_created_at", "run_updated_at", "report_created_at"):
            _aware_datetime(name, getattr(self, name))
        for name in (
            "query",
            "run_status",
            "report_status",
            "overall_confidence",
            "evidence_coverage",
            "disclaimer",
            "generation_method",
            "llm_provider",
            "llm_model",
        ):
            object.__setattr__(self, name, _require_text(name, getattr(self, name)))
        for name in ("company_name", "company_id", "ticker", "security_id"):
            object.__setattr__(self, name, _optional_text(getattr(self, name)))
        for name in (
            "current_situation",
            "strengths",
            "weaknesses",
            "opportunities",
            "risks",
            "unknowns",
        ):
            object.__setattr__(self, name, _point_tuple(name, getattr(self, name)))
        if not isinstance(self.upside_scenario, ReportExportScenario):
            raise ValueError("upside_scenario must be a ReportExportScenario")
        if not isinstance(self.downside_scenario, ReportExportScenario):
            raise ValueError("downside_scenario must be a ReportExportScenario")
        if not 0 <= self.evidence_coverage_ratio <= 1:
            raise ValueError("evidence_coverage_ratio must be between 0 and 1")
        object.__setattr__(self, "warnings", _text_tuple("warnings", self.warnings))
        object.__setattr__(self, "limitations", _text_tuple("limitations", self.limitations))
        object.__setattr__(self, "sources", _source_tuple(self.sources))
        object.__setattr__(self, "chart_series", _chart_series_tuple(self.chart_series))
        narrative_sections = tuple(self.narrative_sections)
        if any(not isinstance(item, ReportExportNarrativeSection) for item in narrative_sections):
            raise ValueError("narrative_sections must contain ReportExportNarrativeSection values")
        object.__setattr__(self, "narrative_sections", narrative_sections)
        for name in ("narrative_provider", "narrative_model", "narrative_synthesis_sha256"):
            object.__setattr__(self, name, _optional_text(getattr(self, name)))
        narrative_metadata = (
            self.narrative_provider,
            self.narrative_model,
            self.narrative_synthesis_sha256,
        )
        if any(narrative_metadata) and not all(narrative_metadata):
            raise ValueError("narrative metadata must be provided together")
        if self.narrative_synthesis_sha256 is not None and not re.fullmatch(
            r"[0-9a-f]{64}",
            self.narrative_synthesis_sha256,
        ):
            raise ValueError("narrative_synthesis_sha256 must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class ReportExportArtifact:
    format: ReportExportFormat
    filename: str
    mime_type: str
    byte_size: int
    sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "format", ReportExportFormat(self.format))
        object.__setattr__(self, "filename", _safe_name("filename", self.filename))
        object.__setattr__(self, "mime_type", _require_text("mime_type", self.mime_type))
        if self.byte_size <= 0:
            raise ValueError("byte_size must be positive")
        if not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError("sha256 must be a lowercase SHA-256 digest")
        if not self.filename.endswith(self.format.suffix):
            raise ValueError("filename suffix must match format")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        return cls(
            format=ReportExportFormat(str(payload["format"])),
            filename=str(payload["filename"]),
            mime_type=str(payload["mime_type"]),
            byte_size=int(payload["byte_size"]),
            sha256=str(payload["sha256"]),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "format": self.format.value,
            "filename": self.filename,
            "mime_type": self.mime_type,
            "byte_size": self.byte_size,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class ReportExportSnapshot:
    export_id: str
    run_id: str
    report_id: str
    created_at: datetime
    company_name: str | None
    company_id: str | None
    ticker: str | None
    security_id: str | None
    artifacts: tuple[ReportExportArtifact, ...]
    narrative_synthesis_sha256: str | None = None
    schema_version: int = REPORT_EXPORT_MANIFEST_VERSION
    content_version: int = REPORT_EXPORT_CONTENT_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "export_id", _safe_name("export_id", self.export_id))
        object.__setattr__(self, "run_id", _safe_name("run_id", self.run_id))
        object.__setattr__(self, "report_id", _safe_name("report_id", self.report_id))
        _aware_datetime("created_at", self.created_at)
        for name in ("company_name", "company_id", "ticker", "security_id"):
            object.__setattr__(self, name, _optional_text(getattr(self, name)))
        artifacts = tuple(self.artifacts)
        if {artifact.format for artifact in artifacts} != set(ReportExportFormat):
            raise ValueError("snapshot must contain markdown, html, and pdf artifacts")
        object.__setattr__(self, "artifacts", artifacts)
        object.__setattr__(
            self,
            "narrative_synthesis_sha256",
            _optional_text(self.narrative_synthesis_sha256),
        )
        if self.narrative_synthesis_sha256 is not None and not re.fullmatch(
            r"[0-9a-f]{64}",
            self.narrative_synthesis_sha256,
        ):
            raise ValueError("narrative_synthesis_sha256 must be a lowercase SHA-256 digest")
        if self.schema_version != REPORT_EXPORT_MANIFEST_VERSION:
            raise ValueError("unsupported report export manifest version")
        if not 1 <= self.content_version <= REPORT_EXPORT_CONTENT_VERSION:
            raise ValueError("unsupported report export content version")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        company = _mapping(payload.get("company"))
        security = _mapping(payload.get("security"))
        return cls(
            export_id=str(payload["export_id"]),
            run_id=str(payload["run_id"]),
            report_id=str(payload["report_id"]),
            created_at=datetime.fromisoformat(str(payload["created_at"])),
            company_name=_optional_mapping_text(company, "name"),
            company_id=_optional_mapping_text(company, "id"),
            ticker=_optional_mapping_text(security, "ticker"),
            security_id=_optional_mapping_text(security, "id"),
            artifacts=tuple(
                ReportExportArtifact.from_dict(item)
                for item in payload.get("artifacts", ())
                if isinstance(item, Mapping)
            ),
            narrative_synthesis_sha256=_optional_text(payload.get("narrative_synthesis_sha256")),
            schema_version=int(payload["schema_version"]),
            content_version=int(payload.get("content_version", 1)),
        )

    def artifact(self, export_format: ReportExportFormat) -> ReportExportArtifact | None:
        expected = ReportExportFormat(export_format)
        return next((item for item in self.artifacts if item.format == expected), None)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "content_version": self.content_version,
            "export_id": self.export_id,
            "run_id": self.run_id,
            "report_id": self.report_id,
            "created_at": self.created_at.isoformat(),
            "company": {"id": self.company_id, "name": self.company_name},
            "security": {"id": self.security_id, "ticker": self.ticker},
            "narrative_synthesis_sha256": self.narrative_synthesis_sha256,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
        }


def _require_text(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    text = value.strip()
    if not text:
        raise ValueError(f"{name} is required")
    return text


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _bounded_optional_text(value: object, limit: int) -> str | None:
    text = _optional_text(value)
    if text is None or len(text) <= limit:
        return text
    return f"{text[: limit - 1].rstrip()}…"


def _safe_name(name: str, value: str) -> str:
    text = _require_text(name, value)
    if not _SAFE_NAME_PATTERN.fullmatch(text) or text in {".", ".."}:
        raise ValueError(f"{name} must be a safe generated name")
    return text


def _aware_datetime(name: str, value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{name} must be a timezone-aware datetime")
    return value


def _text_tuple(name: str, values: Iterable[str]) -> tuple[str, ...]:
    if isinstance(values, str):
        raise ValueError(f"{name} must be an iterable of strings")
    return tuple(
        dict.fromkeys(_require_text(f"{name}[{index}]", item) for index, item in enumerate(values))
    )


def _marker_tuple(name: str, values: Iterable[str]) -> tuple[str, ...]:
    markers = _text_tuple(name, values)
    if any(not _MARKER_PATTERN.fullmatch(marker) for marker in markers):
        raise ValueError(f"{name} contains an invalid source marker")
    return markers


def _marker_mapping(
    name: str, values: Mapping[str, Iterable[str]]
) -> Mapping[str, tuple[str, ...]]:
    if not isinstance(values, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return MappingProxyType(
        {
            _require_text(f"{name}.key", key): _marker_tuple(f"{name}[{key!r}]", markers)
            for key, markers in values.items()
        }
    )


def _point_tuple(name: str, values: Iterable[ReportExportPoint]) -> tuple[ReportExportPoint, ...]:
    points = tuple(values)
    if any(not isinstance(point, ReportExportPoint) for point in points):
        raise ValueError(f"{name} must contain ReportExportPoint values")
    return points


def _source_tuple(values: Iterable[ReportSourceReference]) -> tuple[ReportSourceReference, ...]:
    sources = tuple(values)
    if any(not isinstance(source, ReportSourceReference) for source in sources):
        raise ValueError("sources must contain ReportSourceReference values")
    if len({source.marker for source in sources}) != len(sources):
        raise ValueError("source markers must be unique")
    return sources


def _chart_series_tuple(
    values: Iterable[ReportExportChartSeries],
) -> tuple[ReportExportChartSeries, ...]:
    series = tuple(values)
    if any(not isinstance(item, ReportExportChartSeries) for item in series):
        raise ValueError("chart_series must contain ReportExportChartSeries values")
    if len({item.symbol for item in series}) != len(series):
        raise ValueError("chart series symbols must be unique")
    if series:
        expected_dates = tuple(point.priced_at for point in series[0].points)
        if any(
            tuple(point.priced_at for point in item.points) != expected_dates for item in series[1:]
        ):
            raise ValueError("chart series must use aligned dates")
    return series


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _optional_mapping_text(values: Mapping[str, Any], key: str) -> str | None:
    return _optional_text(values.get(key))
