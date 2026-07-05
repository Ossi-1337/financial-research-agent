from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Self

from financial_research_agent.retrieval.contracts import (
    IndexedChunk,
    RetrievalError,
    RetrievalErrorCode,
    RetrievalIndexMetadata,
    RetrievalMatch,
    RetrievalProviderName,
    RetrievalQuery,
    RetrievalResult,
    indexed_chunk_from_payload,
)
from financial_research_agent.settings import Settings

LOCAL_VECTOR_INDEX_VERSION = 1


class LocalVectorIndex:
    def __init__(
        self,
        *,
        storage_path: Path | None = None,
        provider: str = RetrievalProviderName.LOCAL_VECTOR.value,
        index_id: str = RetrievalProviderName.LOCAL_VECTOR.value,
    ) -> None:
        self.storage_path = storage_path
        self.provider = _require_text("provider", provider)
        self.index_id = _require_text("index_id", index_id)
        self._records: dict[str, IndexedChunk] = {}
        self._updated_at: datetime | None = None
        self._load()

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        return cls(storage_path=settings.local_paths.data_dir / "retrieval" / "vector_index.json")

    def count(self) -> int:
        return len(self._records)

    def metadata(self) -> RetrievalIndexMetadata:
        latest_record = max(
            self._records.values(),
            key=lambda record: record.indexed_at,
            default=None,
        )
        return RetrievalIndexMetadata(
            provider=self.provider,
            index_id=self.index_id,
            record_count=len(self._records),
            storage_path=str(self.storage_path) if self.storage_path is not None else None,
            embedding_provider=latest_record.embedding_provider if latest_record else None,
            embedding_model=latest_record.embedding_model if latest_record else None,
            updated_at=self._updated_at or (latest_record.indexed_at if latest_record else None),
            vector_dimensions=self.vector_dimensions,
        )

    @property
    def vector_dimensions(self) -> int | None:
        first = next(iter(self._records.values()), None)
        if first is None:
            return None
        return len(first.embedding)

    def upsert(self, records: tuple[IndexedChunk, ...]) -> None:
        self._validate_dimensions(records)
        for record in records:
            self._records[record.id] = record
        self._updated_at = max(
            (record.indexed_at for record in self._records.values()),
            default=None,
        )
        self._save()

    def delete_by_metadata(self, key: str, value: str) -> int:
        normalized_key = _require_text("key", key)
        normalized_value = _require_text("value", value)
        deleted_ids = [
            record_id
            for record_id, record in self._records.items()
            if record.chunk.metadata.get(normalized_key) == normalized_value
        ]
        for record_id in deleted_ids:
            del self._records[record_id]
        if deleted_ids:
            self._updated_at = datetime.now(UTC) if self._records else None
            self._save()
        return len(deleted_ids)

    def clear(self) -> int:
        deleted = len(self._records)
        self._records.clear()
        self._updated_at = None
        if self.storage_path is not None and self.storage_path.exists():
            self.storage_path.unlink()
        return deleted

    def search(
        self,
        query: RetrievalQuery,
        *,
        query_embedding: tuple[float, ...],
        now: datetime | None = None,
    ) -> RetrievalResult:
        if not isinstance(query, RetrievalQuery):
            raise ValueError("query must be a RetrievalQuery")
        embedding = _float_tuple("query_embedding", query_embedding)
        if not self._records:
            raise RetrievalError(
                code=RetrievalErrorCode.INDEX_EMPTY,
                message="Retrieval index is empty; rebuild it from stored source chunks first.",
                provider=self.provider,
            )
        self._validate_query_dimensions(embedding)
        scored = []
        for record in self._records.values():
            if not _matches_filters(record, query):
                continue
            score = _cosine_similarity(embedding, record.embedding)
            if score >= query.min_score:
                scored.append((score, record))
        scored.sort(key=lambda item: (-item[0], item[1].id))
        matches = tuple(
            RetrievalMatch(chunk=record.chunk, score=score, rank=index)
            for index, (score, record) in enumerate(scored[: query.top_k], start=1)
        )
        return RetrievalResult(
            query=query,
            matches=matches,
            provider=self.provider,
            index_id=self.index_id,
            generated_at=now or datetime.now(UTC),
        )

    def _validate_dimensions(self, records: tuple[IndexedChunk, ...]) -> None:
        expected = self.vector_dimensions
        for record in records:
            if not isinstance(record, IndexedChunk):
                raise ValueError("records must contain IndexedChunk values")
            if expected is None:
                expected = len(record.embedding)
            elif len(record.embedding) != expected:
                raise RetrievalError(
                    code=RetrievalErrorCode.VECTOR_DIMENSION_MISMATCH,
                    message=(
                        f"Embedding dimension {len(record.embedding)} does not match "
                        f"index dimension {expected}."
                    ),
                    provider=self.provider,
                )

    def _validate_query_dimensions(self, embedding: tuple[float, ...]) -> None:
        expected = self.vector_dimensions
        if expected is not None and len(embedding) != expected:
            raise RetrievalError(
                code=RetrievalErrorCode.VECTOR_DIMENSION_MISMATCH,
                message=(
                    f"Query embedding dimension {len(embedding)} does not match "
                    f"index dimension {expected}."
                ),
                provider=self.provider,
            )

    def _load(self) -> None:
        if self.storage_path is None or not self.storage_path.exists():
            return
        try:
            payload = json.loads(self.storage_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("index root must be an object")
            if payload.get("version") != LOCAL_VECTOR_INDEX_VERSION:
                raise ValueError("unsupported retrieval index version")
            self.provider = str(payload.get("provider") or self.provider)
            self.index_id = str(payload.get("index_id") or self.index_id)
            updated_at = payload.get("updated_at")
            self._updated_at = _optional_aware_datetime_from_payload(updated_at)
            records_payload = payload.get("records", [])
            if not isinstance(records_payload, list):
                raise ValueError("records must be a list")
            records = tuple(indexed_chunk_from_payload(item) for item in records_payload)
            self._validate_dimensions(records)
            self._records = {record.id: record for record in records}
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RetrievalError(
                code=RetrievalErrorCode.MALFORMED_INDEX,
                message=f"Could not load retrieval index: {self.storage_path}",
                provider=self.provider,
            ) from exc

    def _save(self) -> None:
        if self.storage_path is None:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": LOCAL_VECTOR_INDEX_VERSION,
            "provider": self.provider,
            "index_id": self.index_id,
            "updated_at": self._updated_at.isoformat() if self._updated_at else None,
            "records": [record.to_dict() for record in self._records.values()],
        }
        temp_path = self.storage_path.with_suffix(f"{self.storage_path.suffix}.tmp")
        temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temp_path.replace(self.storage_path)


def _matches_filters(record: IndexedChunk, query: RetrievalQuery) -> bool:
    return all(record.chunk.metadata.get(key) == value for key, value in query.filters.items())


def _cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    dot_product = sum(
        left_value * right_value for left_value, right_value in zip(left, right, strict=True)
    )
    return dot_product / (left_norm * right_norm)


def _float_tuple(name: str, values: tuple[float, ...]) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result:
        raise ValueError(f"{name} must contain at least one value")
    return result


def _require_text(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    text = value.strip()
    if text == "":
        raise ValueError(f"{name} is required")
    return text


def _optional_aware_datetime_from_payload(value: object) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("updated_at must be a string")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("updated_at must be timezone-aware")
    return parsed
