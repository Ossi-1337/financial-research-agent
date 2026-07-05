from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType

from financial_research_agent.reports import Citation, EvidenceSnippet

NO_RECOMMENDATION_NOTICE = (
    "This financial report analysis is evidence review only and does not provide buy, "
    "sell, hold, or other investment recommendations."
)


class FinancialReportAnalysisStatus(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    NO_DATA = "no_data"


class FinancialReportSection(StrEnum):
    REVENUE = "revenue"
    MARGINS = "margins"
    CASH_FLOW = "cash_flow"
    DEBT_LIQUIDITY = "debt_liquidity"
    GUIDANCE = "guidance"
    RISKS = "risks"
    ACCOUNTING_CAVEATS = "accounting_caveats"


class ConfidenceLabel(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class FinancialReportAnalysisCompany:
    cik: str
    company_id: str | None = None
    legal_name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "cik", _normalize_cik(self.cik))
        object.__setattr__(self, "company_id", _optional_text(self.company_id))
        object.__setattr__(self, "legal_name", _optional_text(self.legal_name))

    @property
    def padded_cik(self) -> str:
        return self.cik.zfill(10)

    def to_dict(self) -> dict[str, object]:
        return {
            "cik": self.cik,
            "padded_cik": self.padded_cik,
            "company_id": self.company_id,
            "legal_name": self.legal_name,
        }


@dataclass(frozen=True, slots=True)
class FinancialReportQuestion:
    id: str
    section: FinancialReportSection
    question: str
    evidence_ids: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_text("id", self.id))
        object.__setattr__(self, "section", FinancialReportSection(self.section))
        object.__setattr__(self, "question", _require_text("question", self.question))
        object.__setattr__(self, "evidence_ids", _text_tuple("evidence_ids", self.evidence_ids))
        object.__setattr__(self, "limitations", _text_tuple("limitations", self.limitations))

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "section": self.section.value,
            "question": self.question,
            "evidence_ids": list(self.evidence_ids),
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True, slots=True)
class FinancialReportFinding:
    id: str
    section: FinancialReportSection
    title: str
    summary: str
    confidence: ConfidenceLabel
    evidence_ids: tuple[str, ...] = ()
    citation_ids: tuple[str, ...] = ()
    prior_period_comparison: str | None = None
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_text("id", self.id))
        object.__setattr__(self, "section", FinancialReportSection(self.section))
        object.__setattr__(self, "title", _require_text("title", self.title))
        object.__setattr__(self, "summary", _require_text("summary", self.summary))
        object.__setattr__(self, "confidence", ConfidenceLabel(self.confidence))
        object.__setattr__(self, "evidence_ids", _text_tuple("evidence_ids", self.evidence_ids))
        object.__setattr__(self, "citation_ids", _text_tuple("citation_ids", self.citation_ids))
        object.__setattr__(
            self,
            "prior_period_comparison",
            _optional_text(self.prior_period_comparison),
        )
        object.__setattr__(self, "limitations", _text_tuple("limitations", self.limitations))
        if not self.evidence_ids and not self.limitations:
            raise ValueError("finding must include evidence_ids or limitations")

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "section": self.section.value,
            "title": self.title,
            "summary": self.summary,
            "confidence": self.confidence.value,
            "evidence_ids": list(self.evidence_ids),
            "citation_ids": list(self.citation_ids),
            "prior_period_comparison": self.prior_period_comparison,
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True, slots=True)
class FinancialReportAnalysisResult:
    id: str
    company: FinancialReportAnalysisCompany
    status: FinancialReportAnalysisStatus
    created_at: datetime
    questions: tuple[FinancialReportQuestion, ...]
    findings: tuple[FinancialReportFinding, ...]
    citations: tuple[Citation, ...] = ()
    evidence: tuple[EvidenceSnippet, ...] = ()
    limitations: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    source_summary: Mapping[str, str] = field(default_factory=dict)
    no_recommendation_notice: str = NO_RECOMMENDATION_NOTICE

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_text("id", self.id))
        if not isinstance(self.company, FinancialReportAnalysisCompany):
            raise ValueError("company must be a FinancialReportAnalysisCompany")
        object.__setattr__(self, "status", FinancialReportAnalysisStatus(self.status))
        object.__setattr__(self, "created_at", _aware_datetime("created_at", self.created_at))
        object.__setattr__(self, "questions", _question_tuple(self.questions))
        object.__setattr__(self, "findings", _finding_tuple(self.findings))
        object.__setattr__(self, "citations", _citation_tuple(self.citations))
        object.__setattr__(self, "evidence", _evidence_tuple(self.evidence))
        object.__setattr__(self, "limitations", _text_tuple("limitations", self.limitations))
        object.__setattr__(self, "warnings", _text_tuple("warnings", self.warnings))
        object.__setattr__(
            self,
            "source_summary",
            _text_mapping("source_summary", self.source_summary),
        )
        object.__setattr__(
            self,
            "no_recommendation_notice",
            _require_text("no_recommendation_notice", self.no_recommendation_notice),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "company": self.company.to_dict(),
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "questions": [question.to_dict() for question in self.questions],
            "findings": [finding.to_dict() for finding in self.findings],
            "citations": [citation.to_dict() for citation in self.citations],
            "evidence": [snippet.to_dict() for snippet in self.evidence],
            "limitations": list(self.limitations),
            "warnings": list(self.warnings),
            "source_summary": dict(self.source_summary),
            "no_recommendation_notice": self.no_recommendation_notice,
        }


def _normalize_cik(value: str) -> str:
    text = _require_text("cik", value)
    if text.upper().startswith("CIK"):
        text = text[3:]
    digits = text.lstrip("0") or "0"
    if not digits.isdigit() or int(digits) <= 0:
        raise ValueError("cik must contain positive digits")
    if len(digits) > 10:
        raise ValueError("cik must be at most 10 digits")
    return digits


def _require_text(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    text = value.strip()
    if text == "":
        raise ValueError(f"{name} is required")
    return text


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    return text or None


def _text_tuple(name: str, values: Iterable[str]) -> tuple[str, ...]:
    if isinstance(values, str):
        raise ValueError(f"{name} must be an iterable of strings, not a string")
    return tuple(_require_text(f"{name}[{index}]", value) for index, value in enumerate(values))


def _text_mapping(name: str, values: Mapping[str, str]) -> Mapping[str, str]:
    if not isinstance(values, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return MappingProxyType(
        {
            _require_text(f"{name}.key", key): _require_text(f"{name}[{key!r}]", value)
            for key, value in values.items()
        }
    )


def _question_tuple(
    values: Iterable[FinancialReportQuestion],
) -> tuple[FinancialReportQuestion, ...]:
    questions = tuple(values)
    for index, question in enumerate(questions):
        if not isinstance(question, FinancialReportQuestion):
            raise ValueError(f"questions[{index}] must be a FinancialReportQuestion")
    return questions


def _finding_tuple(
    values: Iterable[FinancialReportFinding],
) -> tuple[FinancialReportFinding, ...]:
    findings = tuple(values)
    for index, finding in enumerate(findings):
        if not isinstance(finding, FinancialReportFinding):
            raise ValueError(f"findings[{index}] must be a FinancialReportFinding")
    return findings


def _citation_tuple(values: Iterable[Citation]) -> tuple[Citation, ...]:
    citations = tuple(values)
    for index, citation in enumerate(citations):
        if not isinstance(citation, Citation):
            raise ValueError(f"citations[{index}] must be a Citation")
    return citations


def _evidence_tuple(values: Iterable[EvidenceSnippet]) -> tuple[EvidenceSnippet, ...]:
    snippets = tuple(values)
    for index, snippet in enumerate(snippets):
        if not isinstance(snippet, EvidenceSnippet):
            raise ValueError(f"evidence[{index}] must be an EvidenceSnippet")
    return snippets


def _aware_datetime(name: str, value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError(f"{name} must be a datetime")
    if value.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value
