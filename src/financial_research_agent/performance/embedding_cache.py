from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Self

from financial_research_agent.llm import (
    EmbeddingProvider,
    EmbeddingRequest,
    EmbeddingResponse,
    ModelMetadata,
    TokenUsage,
)
from financial_research_agent.settings import Settings

EMBEDDING_CACHE_VERSION = 1


@dataclass(frozen=True, slots=True)
class EmbeddingCacheEntry:
    provider: str
    model: str
    text_sha256: str
    embedding: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "model": self.model,
            "text_sha256": self.text_sha256,
            "embedding": list(self.embedding),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> Self:
        embedding = payload.get("embedding")
        if not isinstance(embedding, list):
            raise ValueError("embedding must be a list")
        return cls(
            provider=_payload_text(payload, "provider"),
            model=_payload_text(payload, "model"),
            text_sha256=_payload_text(payload, "text_sha256"),
            embedding=tuple(float(value) for value in embedding),
        )


class LocalEmbeddingCache:
    def __init__(self, *, storage_path: Path | None = None) -> None:
        self.storage_path = storage_path
        self._lock = Lock()
        self._entries: dict[str, EmbeddingCacheEntry] = {}
        self._load()

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        return cls(storage_path=settings.local_paths.cache_dir / "embedding_cache.json")

    def get(self, *, provider: str, model: str, text: str) -> tuple[float, ...] | None:
        key = _cache_key(provider=provider, model=model, text=text)
        with self._lock:
            entry = self._entries.get(key)
            return entry.embedding if entry is not None else None

    def put(self, *, provider: str, model: str, text: str, embedding: tuple[float, ...]) -> None:
        self.put_many(provider=provider, model=model, items={text: embedding})

    def put_many(
        self,
        *,
        provider: str,
        model: str,
        items: Mapping[str, tuple[float, ...]],
    ) -> None:
        with self._lock:
            for text, embedding in items.items():
                key = _cache_key(provider=provider, model=model, text=text)
                self._entries[key] = EmbeddingCacheEntry(
                    provider=provider,
                    model=model,
                    text_sha256=_text_hash(text),
                    embedding=tuple(float(value) for value in embedding),
                )
            self._save()

    def count(self) -> int:
        with self._lock:
            return len(self._entries)

    def clear(self) -> int:
        with self._lock:
            count = len(self._entries)
            self._entries = {}
            self._save()
            return count

    def to_dict(self) -> dict[str, object]:
        return {
            "version": EMBEDDING_CACHE_VERSION,
            "storage_path": str(self.storage_path) if self.storage_path is not None else None,
            "record_count": self.count(),
            "stores_raw_text": False,
        }

    def _load(self) -> None:
        if self.storage_path is None or not self.storage_path.exists():
            return
        try:
            payload = json.loads(self.storage_path.read_text(encoding="utf-8"))
            if payload.get("version") != EMBEDDING_CACHE_VERSION:
                raise ValueError("unsupported embedding cache version")
            entries = payload.get("entries", {})
            if not isinstance(entries, dict):
                raise ValueError("embedding cache entries must be an object")
            self._entries = {
                key: EmbeddingCacheEntry.from_dict(_payload_mapping(value))
                for key, value in entries.items()
            }
        except (OSError, ValueError, json.JSONDecodeError, TypeError) as exc:
            raise ValueError(f"Could not load embedding cache: {self.storage_path}") from exc

    def _save(self) -> None:
        if self.storage_path is None:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": EMBEDDING_CACHE_VERSION,
            "entries": {key: entry.to_dict() for key, entry in sorted(self._entries.items())},
        }
        temp_path = self.storage_path.with_suffix(f"{self.storage_path.suffix}.tmp")
        temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temp_path.replace(self.storage_path)


class CachingEmbeddingProvider:
    def __init__(self, provider: EmbeddingProvider, cache: LocalEmbeddingCache) -> None:
        self.provider = provider
        self.cache = cache

    @property
    def metadata(self) -> ModelMetadata:
        return self.provider.metadata

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        resolved_model = request.model or self.provider.metadata.model
        resolved_provider = self.provider.metadata.provider
        embeddings: list[tuple[float, ...] | None] = []
        missing_positions_by_text: dict[str, list[int]] = {}
        hits = 0
        duplicates = 0

        for index, text in enumerate(request.input_texts):
            cached = self.cache.get(provider=resolved_provider, model=resolved_model, text=text)
            if cached is None:
                embeddings.append(None)
                positions = missing_positions_by_text.setdefault(text, [])
                if positions:
                    duplicates += 1
                positions.append(index)
            else:
                embeddings.append(cached)
                hits += 1

        usage = TokenUsage()
        provider_name = resolved_provider
        model_name = resolved_model
        missing_texts = tuple(missing_positions_by_text)
        if missing_texts:
            response = await self.provider.embed(
                EmbeddingRequest(
                    input_texts=missing_texts,
                    model=request.model,
                    metadata=request.metadata,
                )
            )
            if len(response.embeddings) != len(missing_texts):
                raise ValueError("embedding provider returned a different number of vectors")
            usage = response.usage
            provider_name = response.provider
            model_name = response.model
            cache_items: dict[str, tuple[float, ...]] = {}
            for text, embedding in zip(missing_texts, response.embeddings, strict=True):
                for position in missing_positions_by_text[text]:
                    embeddings[position] = embedding
                cache_items[text] = embedding
            self.cache.put_many(provider=provider_name, model=model_name, items=cache_items)

        return EmbeddingResponse(
            embeddings=tuple(_require_embedding(item) for item in embeddings),
            provider=provider_name,
            model=model_name,
            usage=usage,
            metadata={
                "cache_hits": str(hits),
                "cache_misses": str(len(missing_texts)),
                "request_duplicates": str(duplicates),
            },
        )


def _cache_key(*, provider: str, model: str, text: str) -> str:
    payload = "\n".join((_require_text(provider), _require_text(model), _text_hash(text)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _require_embedding(value: tuple[float, ...] | None) -> tuple[float, ...]:
    if value is None:
        raise ValueError("embedding is missing")
    return value


def _require_text(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("value must be a string")
    text = value.strip()
    if text == "":
        raise ValueError("value is required")
    return text


def _payload_text(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return _require_text(value)


def _payload_mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("embedding cache entry must be an object")
    return value
