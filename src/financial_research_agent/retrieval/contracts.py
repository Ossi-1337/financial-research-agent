from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any


class RetrievalProviderName(StrEnum):
    LOCAL_VECTOR = "local-vector"


class RetrievalSourceKind(StrEnum):
    FILING_CHUNK = "filing_chunk"
    EVIDENCE = "evidence"


class RetrievalScoreKind(StrEnum):
    COSINE_SIMILARITY = "cosine_similarity"


class RetrievalErrorCode(StrEnum):
    INVALID_REQUEST = "invalid_request"
    INDEX_EMPTY = "index_empty"
    INDEX_NOT_FOUND = "index_not_found"
    EMBEDDING_FAILED = "embedding_failed"
    VECTOR_DIMENSION_MISMATCH = "vector_dimension_mismatch"
    MALFORMED_INDEX = "malformed_index"


@dataclass(frozen=True, slots=True)
class ChatRetrievalMetadata:
    query: str | None = None
    specialist_roles: tuple[str, ...] = ()
    methods: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    duration_ms: int | None = None
    no_result_reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "query", _optional_text(self.query))
        object.__setattr__(
            self,
            "specialist_roles",
            _bounded_text_tuple("specialist_roles", self.specialist_roles, maximum=4),
        )
        object.__setattr__(
            self,
            "methods",
            _bounded_text_tuple("methods", self.methods, maximum=8),
        )
        object.__setattr__(
            self,
            "evidence_ids",
            _bounded_text_tuple("evidence_ids", self.evidence_ids, maximum=100),
        )
        if self.duration_ms is not None and self.duration_ms < 0:
            raise ValueError("duration_ms must be non-negative")
        object.__setattr__(self, "no_result_reason", _optional_text(self.no_result_reason))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ChatRetrievalMetadata:
        return cls(
            query=_optional_payload_text(payload, "query"),
            specialist_roles=tuple(
                str(item) for item in _sequence_payload_default(payload, "specialist_roles")
            ),
            methods=tuple(str(item) for item in _sequence_payload_default(payload, "methods")),
            evidence_ids=tuple(
                str(item) for item in _sequence_payload_default(payload, "evidence_ids")
            ),
            duration_ms=(
                int(payload["duration_ms"]) if payload.get("duration_ms") is not None else None
            ),
            no_result_reason=_optional_payload_text(payload, "no_result_reason"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "query": self.query,
            "specialist_roles": list(self.specialist_roles),
            "methods": list(self.methods),
            "evidence_ids": list(self.evidence_ids),
            "duration_ms": self.duration_ms,
            "no_result_reason": self.no_result_reason,
        }


@dataclass(frozen=True, slots=True)
class RetrievalChunk:
    id: str
    text: str
    source_kind: RetrievalSourceKind
    source_id: str
    source_url: str
    document_id: str | None = None
    title: str | None = None
    section_heading: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_text("id", self.id))
        object.__setattr__(self, "text", _require_text("text", self.text))
        object.__setattr__(self, "source_kind", RetrievalSourceKind(self.source_kind))
        object.__setattr__(self, "source_id", _require_text("source_id", self.source_id))
        object.__setattr__(self, "source_url", _require_text("source_url", self.source_url))
        object.__setattr__(self, "document_id", _optional_text(self.document_id))
        object.__setattr__(self, "title", _optional_text(self.title))
        object.__setattr__(self, "section_heading", _optional_text(self.section_heading))
        object.__setattr__(self, "metadata", _text_mapping("metadata", self.metadata))

    def snippet(self, *, max_chars: int = 500) -> str:
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        normalized = " ".join(self.text.split())
        if len(normalized) <= max_chars:
            return normalized
        return normalized[: max_chars - 1].rstrip() + "..."

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "text": self.text,
            "source_kind": self.source_kind.value,
            "source_id": self.source_id,
            "source_url": self.source_url,
            "document_id": self.document_id,
            "title": self.title,
            "section_heading": self.section_heading,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class IndexedChunk:
    chunk: RetrievalChunk
    embedding: tuple[float, ...]
    embedding_provider: str
    embedding_model: str
    indexed_at: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.chunk, RetrievalChunk):
            raise ValueError("chunk must be a RetrievalChunk")
        object.__setattr__(self, "embedding", _float_tuple("embedding", self.embedding))
        object.__setattr__(
            self,
            "embedding_provider",
            _require_text("embedding_provider", self.embedding_provider),
        )
        object.__setattr__(
            self,
            "embedding_model",
            _require_text("embedding_model", self.embedding_model),
        )
        object.__setattr__(self, "indexed_at", _aware_datetime("indexed_at", self.indexed_at))

    @property
    def id(self) -> str:
        return self.chunk.id

    def to_dict(self) -> dict[str, object]:
        return {
            "chunk": self.chunk.to_dict(),
            "embedding": list(self.embedding),
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "indexed_at": self.indexed_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class RetrievalQuery:
    query: str
    top_k: int = 5
    min_score: float = 0.0
    filters: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "query", _require_text("query", self.query))
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.min_score < -1.0 or self.min_score > 1.0:
            raise ValueError("min_score must be between -1 and 1")
        object.__setattr__(self, "filters", _text_mapping("filters", self.filters))

    def to_dict(self) -> dict[str, object]:
        return {
            "query": self.query,
            "top_k": self.top_k,
            "min_score": self.min_score,
            "filters": dict(self.filters),
        }


@dataclass(frozen=True, slots=True)
class RetrievalMatch:
    chunk: RetrievalChunk
    score: float
    rank: int
    score_kind: RetrievalScoreKind = RetrievalScoreKind.COSINE_SIMILARITY

    def __post_init__(self) -> None:
        if not isinstance(self.chunk, RetrievalChunk):
            raise ValueError("chunk must be a RetrievalChunk")
        if self.rank <= 0:
            raise ValueError("rank must be positive")
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "score_kind", RetrievalScoreKind(self.score_kind))

    def to_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "score": self.score,
            "score_kind": self.score_kind.value,
            "snippet": self.chunk.snippet(),
            "chunk": self.chunk.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    query: RetrievalQuery
    matches: tuple[RetrievalMatch, ...]
    provider: str
    index_id: str
    generated_at: datetime
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.query, RetrievalQuery):
            raise ValueError("query must be a RetrievalQuery")
        object.__setattr__(self, "matches", _match_tuple(self.matches))
        object.__setattr__(self, "provider", _require_text("provider", self.provider))
        object.__setattr__(self, "index_id", _require_text("index_id", self.index_id))
        object.__setattr__(self, "generated_at", _aware_datetime("generated_at", self.generated_at))
        object.__setattr__(self, "warnings", _text_tuple("warnings", self.warnings))

    def to_dict(self) -> dict[str, object]:
        return {
            "query": self.query.to_dict(),
            "provider": self.provider,
            "index_id": self.index_id,
            "generated_at": self.generated_at.isoformat(),
            "matches": [match.to_dict() for match in self.matches],
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class RetrievalIndexMetadata:
    provider: str
    index_id: str
    record_count: int
    storage_path: str | None = None
    embedding_provider: str | None = None
    embedding_model: str | None = None
    updated_at: datetime | None = None
    vector_dimensions: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider", _require_text("provider", self.provider))
        object.__setattr__(self, "index_id", _require_text("index_id", self.index_id))
        if self.record_count < 0:
            raise ValueError("record_count must be non-negative")
        object.__setattr__(self, "storage_path", _optional_text(self.storage_path))
        object.__setattr__(self, "embedding_provider", _optional_text(self.embedding_provider))
        object.__setattr__(self, "embedding_model", _optional_text(self.embedding_model))
        if self.updated_at is not None:
            object.__setattr__(self, "updated_at", _aware_datetime("updated_at", self.updated_at))
        if self.vector_dimensions is not None and self.vector_dimensions <= 0:
            raise ValueError("vector_dimensions must be positive")

    def to_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "index_id": self.index_id,
            "record_count": self.record_count,
            "storage_path": self.storage_path,
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "updated_at": self.updated_at.isoformat() if self.updated_at is not None else None,
            "vector_dimensions": self.vector_dimensions,
        }


@dataclass(frozen=True, slots=True)
class RetrievalIndexBuildResult:
    provider: str
    index_id: str
    indexed_count: int
    skipped_count: int
    embedding_provider: str | None
    embedding_model: str | None
    indexed_at: datetime
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider", _require_text("provider", self.provider))
        object.__setattr__(self, "index_id", _require_text("index_id", self.index_id))
        if self.indexed_count < 0:
            raise ValueError("indexed_count must be non-negative")
        if self.skipped_count < 0:
            raise ValueError("skipped_count must be non-negative")
        object.__setattr__(self, "embedding_provider", _optional_text(self.embedding_provider))
        object.__setattr__(self, "embedding_model", _optional_text(self.embedding_model))
        object.__setattr__(self, "indexed_at", _aware_datetime("indexed_at", self.indexed_at))
        object.__setattr__(self, "warnings", _text_tuple("warnings", self.warnings))

    def to_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "index_id": self.index_id,
            "indexed_count": self.indexed_count,
            "skipped_count": self.skipped_count,
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "indexed_at": self.indexed_at.isoformat(),
            "warnings": list(self.warnings),
        }


class RetrievalError(Exception):
    def __init__(
        self,
        *,
        code: RetrievalErrorCode | str,
        message: str,
        provider: str,
        retryable: bool = False,
    ) -> None:
        self.code = RetrievalErrorCode(code)
        self.message = _require_text("message", message)
        self.provider = _require_text("provider", provider)
        self.retryable = retryable
        super().__init__(self.message)

    def to_dict(self) -> dict[str, object]:
        return {
            "error": "retrieval_error",
            "code": self.code.value,
            "message": self.message,
            "provider": self.provider,
            "retryable": self.retryable,
        }


def indexed_chunk_from_payload(payload: Mapping[str, Any]) -> IndexedChunk:
    chunk_payload = payload.get("chunk")
    if not isinstance(chunk_payload, Mapping):
        raise ValueError("indexed chunk payload must include chunk")
    return IndexedChunk(
        chunk=retrieval_chunk_from_payload(chunk_payload),
        embedding=tuple(float(item) for item in _sequence_payload(payload, "embedding")),
        embedding_provider=str(payload["embedding_provider"]),
        embedding_model=str(payload["embedding_model"]),
        indexed_at=datetime.fromisoformat(str(payload["indexed_at"])),
    )


def retrieval_chunk_from_payload(payload: Mapping[str, Any]) -> RetrievalChunk:
    return RetrievalChunk(
        id=str(payload["id"]),
        text=str(payload["text"]),
        source_kind=RetrievalSourceKind(str(payload["source_kind"])),
        source_id=str(payload["source_id"]),
        source_url=str(payload["source_url"]),
        document_id=_optional_payload_text(payload, "document_id"),
        title=_optional_payload_text(payload, "title"),
        section_heading=_optional_payload_text(payload, "section_heading"),
        metadata={str(key): str(value) for key, value in _mapping_payload(payload, "metadata")},
    )


def _sequence_payload(payload: Mapping[str, Any], key: str) -> Sequence[Any]:
    value = payload.get(key)
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{key} must be a sequence")
    return value


def _sequence_payload_default(payload: Mapping[str, Any], key: str) -> Sequence[Any]:
    value = payload.get(key, ())
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{key} must be a sequence")
    return value


def _mapping_payload(payload: Mapping[str, Any], key: str) -> Iterable[tuple[Any, Any]]:
    value = payload.get(key, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be an object")
    return value.items()


def _optional_payload_text(payload: Mapping[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    return str(value)


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


def _bounded_text_tuple(
    name: str,
    values: Iterable[str],
    *,
    maximum: int,
) -> tuple[str, ...]:
    result = tuple(dict.fromkeys(_text_tuple(name, values)))
    if len(result) > maximum:
        raise ValueError(f"{name} must contain at most {maximum} values")
    return result


def _float_tuple(name: str, values: Iterable[float]) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result:
        raise ValueError(f"{name} must contain at least one value")
    return result


def _match_tuple(values: Iterable[RetrievalMatch]) -> tuple[RetrievalMatch, ...]:
    matches = tuple(values)
    for index, match in enumerate(matches):
        if not isinstance(match, RetrievalMatch):
            raise ValueError(f"matches[{index}] must be RetrievalMatch")
    return matches


def _aware_datetime(name: str, value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError(f"{name} must be a datetime")
    if value.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value
