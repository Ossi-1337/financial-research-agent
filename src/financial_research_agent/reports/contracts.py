from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Self

MAX_QUOTE_CHARS = 500
MAX_SNIPPET_CHARS = 1_200


class CitedResearchRunStatus(StrEnum):
    ANSWERED = "answered"
    LIMITED = "limited"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class Citation:
    id: str
    evidence_id: str
    source_url: str
    retrieved_at: datetime
    source_id: str | None = None
    document_id: str | None = None
    chunk_id: str | None = None
    section: str | None = None
    quote_start: int | None = None
    quote_end: int | None = None
    quote: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_text("id", self.id))
        object.__setattr__(self, "evidence_id", _require_text("evidence_id", self.evidence_id))
        object.__setattr__(self, "source_url", _require_text("source_url", self.source_url))
        object.__setattr__(self, "retrieved_at", _aware_datetime("retrieved_at", self.retrieved_at))
        object.__setattr__(self, "source_id", _optional_text(self.source_id))
        object.__setattr__(self, "document_id", _optional_text(self.document_id))
        object.__setattr__(self, "chunk_id", _optional_text(self.chunk_id))
        object.__setattr__(self, "section", _optional_text(self.section))
        object.__setattr__(
            self,
            "quote",
            _optional_bounded_text("quote", self.quote, MAX_QUOTE_CHARS),
        )
        _validate_quote_bounds(self.quote_start, self.quote_end)
        object.__setattr__(self, "metadata", _text_mapping("metadata", self.metadata))

    @property
    def marker(self) -> str:
        return f"[{self.id}]"

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        return cls(
            id=_payload_text(payload, "id"),
            evidence_id=_payload_text(payload, "evidence_id"),
            source_url=_payload_text(payload, "source_url"),
            retrieved_at=_datetime_from_payload(payload, "retrieved_at"),
            source_id=_payload_optional_text(payload, "source_id"),
            document_id=_payload_optional_text(payload, "document_id"),
            chunk_id=_payload_optional_text(payload, "chunk_id"),
            section=_payload_optional_text(payload, "section"),
            quote_start=_payload_optional_int(payload, "quote_start"),
            quote_end=_payload_optional_int(payload, "quote_end"),
            quote=_payload_optional_text(payload, "quote"),
            metadata=_payload_text_mapping(payload, "metadata"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "marker": self.marker,
            "evidence_id": self.evidence_id,
            "source_url": self.source_url,
            "retrieved_at": self.retrieved_at.isoformat(),
            "source_id": self.source_id,
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "section": self.section,
            "quote_start": self.quote_start,
            "quote_end": self.quote_end,
            "quote": self.quote,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class EvidenceSnippet:
    id: str
    citation_id: str
    text: str
    source_url: str
    retrieved_at: datetime
    score: float
    source_id: str | None = None
    document_id: str | None = None
    chunk_id: str | None = None
    section: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_text("id", self.id))
        object.__setattr__(self, "citation_id", _require_text("citation_id", self.citation_id))
        object.__setattr__(
            self,
            "text",
            _bounded_text("text", self.text, MAX_SNIPPET_CHARS),
        )
        object.__setattr__(self, "source_url", _require_text("source_url", self.source_url))
        object.__setattr__(self, "retrieved_at", _aware_datetime("retrieved_at", self.retrieved_at))
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "source_id", _optional_text(self.source_id))
        object.__setattr__(self, "document_id", _optional_text(self.document_id))
        object.__setattr__(self, "chunk_id", _optional_text(self.chunk_id))
        object.__setattr__(self, "section", _optional_text(self.section))
        object.__setattr__(self, "metadata", _text_mapping("metadata", self.metadata))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        return cls(
            id=_payload_text(payload, "id"),
            citation_id=_payload_text(payload, "citation_id"),
            text=_payload_text(payload, "text"),
            source_url=_payload_text(payload, "source_url"),
            retrieved_at=_datetime_from_payload(payload, "retrieved_at"),
            score=float(payload["score"]),
            source_id=_payload_optional_text(payload, "source_id"),
            document_id=_payload_optional_text(payload, "document_id"),
            chunk_id=_payload_optional_text(payload, "chunk_id"),
            section=_payload_optional_text(payload, "section"),
            metadata=_payload_text_mapping(payload, "metadata"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "citation_id": self.citation_id,
            "text": self.text,
            "source_url": self.source_url,
            "retrieved_at": self.retrieved_at.isoformat(),
            "score": self.score,
            "source_id": self.source_id,
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "section": self.section,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class CitedResearchRun:
    id: str
    query: str
    answer: str
    status: CitedResearchRunStatus
    created_at: datetime
    citations: tuple[Citation, ...] = ()
    evidence: tuple[EvidenceSnippet, ...] = ()
    provider: str | None = None
    model: str | None = None
    limitations: tuple[str, ...] = ()
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_text("id", self.id))
        object.__setattr__(self, "query", _require_text("query", self.query))
        object.__setattr__(self, "answer", _require_text("answer", self.answer))
        object.__setattr__(self, "status", CitedResearchRunStatus(self.status))
        object.__setattr__(self, "created_at", _aware_datetime("created_at", self.created_at))
        object.__setattr__(self, "citations", _citation_tuple(self.citations))
        object.__setattr__(self, "evidence", _evidence_tuple(self.evidence))
        object.__setattr__(self, "provider", _optional_text(self.provider))
        object.__setattr__(self, "model", _optional_text(self.model))
        object.__setattr__(self, "limitations", _text_tuple("limitations", self.limitations))
        object.__setattr__(self, "metadata", _text_mapping("metadata", self.metadata))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        return cls(
            id=_payload_text(payload, "id"),
            query=_payload_text(payload, "query"),
            answer=_payload_text(payload, "answer"),
            status=CitedResearchRunStatus(_payload_text(payload, "status")),
            created_at=_datetime_from_payload(payload, "created_at"),
            citations=tuple(
                Citation.from_dict(item) for item in _payload_mapping_list(payload, "citations")
            ),
            evidence=tuple(
                EvidenceSnippet.from_dict(item)
                for item in _payload_mapping_list(payload, "evidence")
            ),
            provider=_payload_optional_text(payload, "provider"),
            model=_payload_optional_text(payload, "model"),
            limitations=tuple(str(item) for item in _payload_sequence(payload, "limitations")),
            metadata=_payload_text_mapping(payload, "metadata"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "query": self.query,
            "answer": self.answer,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "citations": [citation.to_dict() for citation in self.citations],
            "evidence": [snippet.to_dict() for snippet in self.evidence],
            "provider": self.provider,
            "model": self.model,
            "limitations": list(self.limitations),
            "metadata": dict(self.metadata),
        }


def _validate_quote_bounds(start: int | None, end: int | None) -> None:
    if start is None and end is None:
        return
    if start is None or end is None:
        raise ValueError("quote_start and quote_end must be provided together")
    if start < 0 or end < 0:
        raise ValueError("quote bounds must be non-negative")
    if end <= start:
        raise ValueError("quote_end must be greater than quote_start")


def _citation_tuple(values: Iterable[Citation]) -> tuple[Citation, ...]:
    citations = tuple(values)
    for index, citation in enumerate(citations):
        if not isinstance(citation, Citation):
            raise ValueError(f"citations[{index}] must be Citation")
    return citations


def _evidence_tuple(values: Iterable[EvidenceSnippet]) -> tuple[EvidenceSnippet, ...]:
    snippets = tuple(values)
    for index, snippet in enumerate(snippets):
        if not isinstance(snippet, EvidenceSnippet):
            raise ValueError(f"evidence[{index}] must be EvidenceSnippet")
    return snippets


def _require_text(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    text = value.strip()
    if text == "":
        raise ValueError(f"{name} is required")
    return text


def _bounded_text(name: str, value: str, max_chars: int) -> str:
    text = _require_text(name, value)
    if len(text) > max_chars:
        raise ValueError(f"{name} must be at most {max_chars} characters")
    return text


def _optional_bounded_text(name: str, value: str | None, max_chars: int) -> str | None:
    text = _optional_text(value)
    if text is not None and len(text) > max_chars:
        raise ValueError(f"{name} must be at most {max_chars} characters")
    return text


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    return text or None


def _text_mapping(name: str, values: Mapping[str, str]) -> Mapping[str, str]:
    if not isinstance(values, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return MappingProxyType(
        {
            _require_text(f"{name}.key", key): _require_text(f"{name}[{key!r}]", value)
            for key, value in values.items()
        }
    )


def _text_tuple(name: str, values: Iterable[str]) -> tuple[str, ...]:
    if isinstance(values, str):
        raise ValueError(f"{name} must be an iterable of strings, not a string")
    return tuple(_require_text(f"{name}[{index}]", value) for index, value in enumerate(values))


def _aware_datetime(name: str, value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError(f"{name} must be a datetime")
    if value.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value


def _datetime_from_payload(payload: Mapping[str, Any], name: str) -> datetime:
    try:
        value = datetime.fromisoformat(_payload_text(payload, name))
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO datetime") from exc
    return _aware_datetime(name, value)


def _payload_text(payload: Mapping[str, Any], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return _require_text(name, value)


def _payload_optional_text(payload: Mapping[str, Any], name: str) -> str | None:
    value = payload.get(name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return _optional_text(value)


def _payload_optional_int(payload: Mapping[str, Any], name: str) -> int | None:
    value = payload.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _payload_sequence(payload: Mapping[str, Any], name: str) -> tuple[Any, ...]:
    value = payload.get(name, ())
    if isinstance(value, str) or not isinstance(value, Iterable):
        raise ValueError(f"{name} must be a sequence")
    return tuple(value)


def _payload_mapping_list(payload: Mapping[str, Any], name: str) -> tuple[Mapping[str, Any], ...]:
    values = _payload_sequence(payload, name)
    for index, value in enumerate(values):
        if not isinstance(value, Mapping):
            raise ValueError(f"{name}[{index}] must be an object")
    return values


def _payload_text_mapping(payload: Mapping[str, Any], name: str) -> Mapping[str, str]:
    value = payload.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return {str(key): str(item) for key, item in value.items()}
