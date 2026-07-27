from __future__ import annotations

import json
from dataclasses import dataclass, fields, replace
from pathlib import Path
from threading import Lock
from typing import Self

from financial_research_agent.settings import (
    BackgroundSettings,
    ChatSettings,
    DataSourceSettings,
    ProviderSettings,
    RetrievalSettings,
    Settings,
)

RUNTIME_SETTINGS_STORE_VERSION = 1

_SECRET_FIELD_NAMES = {
    "openai_api_key",
    "anthropic_api_key",
    "gemini_api_key",
    "litellm_api_key",
    "alpha_vantage_api_key",
    "interop_api_key",
    "api_key",
}


@dataclass(frozen=True, slots=True)
class RuntimeSettingsOverrides:
    llm_provider: str | None = None
    llm_model: str | None = None
    llm_base_url: str | None = None
    llm_local_runtime: str | None = None
    llm_timeout_seconds: float | None = None
    embedding_provider: str | None = None
    embedding_model: str | None = None
    chat_provider: str | None = None
    chat_model: str | None = None
    streaming_provider: str | None = None
    streaming_model: str | None = None
    tool_calling_provider: str | None = None
    tool_calling_model: str | None = None
    structured_output_provider: str | None = None
    structured_output_model: str | None = None
    chat_history_recent_turns: int | None = None
    chat_history_summary_max_chars: int | None = None
    company_lookup_provider: str | None = None
    company_lookup_cache_ttl_days: int | None = None
    market_data_provider: str | None = None
    market_data_cache_ttl_days: int | None = None
    financial_statement_provider: str | None = None
    financial_statement_cache_ttl_days: int | None = None
    filing_provider: str | None = None
    filing_cache_ttl_days: int | None = None
    filing_max_document_bytes: int | None = None
    retrieval_provider: str | None = None
    retrieval_top_k: int | None = None
    retrieval_min_score: float | None = None
    background_max_concurrent_research_runs: int | None = None

    @classmethod
    def from_mapping(cls, values: dict[str, object]) -> Self:
        forbidden = set(values) & _SECRET_FIELD_NAMES
        if forbidden:
            raise ValueError(f"Secret settings are environment-only: {sorted(forbidden)[0]}")
        unknown = set(values) - _override_field_names()
        if unknown:
            raise ValueError(f"Unsupported runtime setting: {sorted(unknown)[0]}")
        return cls(**{key: value for key, value in values.items() if value is not None})

    def apply_to(self, settings: Settings) -> Settings:
        provider = _replace_provider_settings(settings.provider, self)
        chat = _replace_chat_settings(settings.chat, self)
        data_sources = _replace_data_source_settings(settings.data_sources, self)
        retrieval = _replace_retrieval_settings(settings.retrieval, self)
        background = _replace_background_settings(settings.background, self)
        return replace(
            settings,
            provider=provider,
            chat=chat,
            data_sources=data_sources,
            retrieval=retrieval,
            background=background,
        )

    def merged_with(self, updates: RuntimeSettingsOverrides) -> RuntimeSettingsOverrides:
        current = self.to_dict()
        current.update(updates.to_dict())
        return RuntimeSettingsOverrides.from_mapping(current)

    def to_dict(self) -> dict[str, object]:
        return {
            field.name: value
            for field in fields(self)
            if (value := getattr(self, field.name)) is not None
        }


class RuntimeSettingsStore:
    def __init__(self, *, storage_path: Path | None = None) -> None:
        self.storage_path = storage_path
        self._lock = Lock()
        self._overrides = RuntimeSettingsOverrides()
        self._load()

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        return cls(storage_path=settings.local_paths.data_dir / "settings_overrides.json")

    def get(self) -> RuntimeSettingsOverrides:
        with self._lock:
            return self._overrides

    def settings(self, base_settings: Settings) -> Settings:
        return self.get().apply_to(base_settings)

    def update(
        self,
        updates: RuntimeSettingsOverrides,
        *,
        base_settings: Settings,
    ) -> RuntimeSettingsOverrides:
        with self._lock:
            merged = self._overrides.merged_with(updates)
            merged.apply_to(base_settings)
            self._overrides = merged
            self._save()
            return self._overrides

    def replace(
        self,
        overrides: RuntimeSettingsOverrides,
        *,
        base_settings: Settings,
    ) -> RuntimeSettingsOverrides:
        overrides.apply_to(base_settings)
        with self._lock:
            self._overrides = overrides
            self._save()
            return self._overrides

    def clear(self) -> RuntimeSettingsOverrides:
        with self._lock:
            self._overrides = RuntimeSettingsOverrides()
            self._save()
            return self._overrides

    def _load(self) -> None:
        if self.storage_path is None or not self.storage_path.exists():
            return
        try:
            payload = json.loads(self.storage_path.read_text(encoding="utf-8"))
            if payload.get("version") != RUNTIME_SETTINGS_STORE_VERSION:
                raise ValueError("unsupported runtime settings version")
            overrides = payload.get("overrides", {})
            if not isinstance(overrides, dict):
                raise ValueError("runtime settings overrides must be an object")
            self._overrides = RuntimeSettingsOverrides.from_mapping(overrides)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            message = f"Could not load runtime settings store: {self.storage_path}"
            raise ValueError(message) from exc

    def _save(self) -> None:
        if self.storage_path is None:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": RUNTIME_SETTINGS_STORE_VERSION,
            "overrides": self._overrides.to_dict(),
        }
        temp_path = self.storage_path.with_suffix(f"{self.storage_path.suffix}.tmp")
        temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temp_path.replace(self.storage_path)


def _override_field_names() -> set[str]:
    return {field.name for field in fields(RuntimeSettingsOverrides)}


def _replace_provider_settings(
    settings: ProviderSettings,
    overrides: RuntimeSettingsOverrides,
) -> ProviderSettings:
    return ProviderSettings(
        llm_provider=overrides.llm_provider or settings.llm_provider,
        llm_model=overrides.llm_model or settings.llm_model,
        llm_base_url=(
            overrides.llm_base_url if overrides.llm_base_url is not None else settings.llm_base_url
        ),
        llm_local_runtime=overrides.llm_local_runtime or settings.llm_local_runtime,
        llm_timeout_seconds=overrides.llm_timeout_seconds or settings.llm_timeout_seconds,
        embedding_provider=overrides.embedding_provider or settings.embedding_provider,
        embedding_model=(
            overrides.embedding_model
            if overrides.embedding_model is not None
            else settings.embedding_model
        ),
        openai_api_key=settings.openai_api_key,
        openai_base_url=settings.openai_base_url,
        openai_organization=settings.openai_organization,
        openai_project=settings.openai_project,
        anthropic_api_key=settings.anthropic_api_key,
        anthropic_base_url=settings.anthropic_base_url,
        anthropic_api_version=settings.anthropic_api_version,
        gemini_api_key=settings.gemini_api_key,
        gemini_base_url=settings.gemini_base_url,
        litellm_api_key=settings.litellm_api_key,
        litellm_base_url=settings.litellm_base_url,
        chat_provider=(
            overrides.chat_provider
            if overrides.chat_provider is not None
            else settings.chat_provider
        ),
        chat_model=(
            overrides.chat_model if overrides.chat_model is not None else settings.chat_model
        ),
        tool_calling_provider=(
            overrides.tool_calling_provider
            if overrides.tool_calling_provider is not None
            else settings.tool_calling_provider
        ),
        tool_calling_model=(
            overrides.tool_calling_model
            if overrides.tool_calling_model is not None
            else settings.tool_calling_model
        ),
        structured_output_provider=(
            overrides.structured_output_provider
            if overrides.structured_output_provider is not None
            else settings.structured_output_provider
        ),
        structured_output_model=(
            overrides.structured_output_model
            if overrides.structured_output_model is not None
            else settings.structured_output_model
        ),
        streaming_provider=(
            overrides.streaming_provider
            if overrides.streaming_provider is not None
            else settings.streaming_provider
        ),
        streaming_model=(
            overrides.streaming_model
            if overrides.streaming_model is not None
            else settings.streaming_model
        ),
    )


def _replace_chat_settings(
    settings: ChatSettings,
    overrides: RuntimeSettingsOverrides,
) -> ChatSettings:
    return ChatSettings(
        history_recent_turns=overrides.chat_history_recent_turns or settings.history_recent_turns,
        history_summary_max_chars=overrides.chat_history_summary_max_chars
        or settings.history_summary_max_chars,
    )


def _replace_data_source_settings(
    settings: DataSourceSettings,
    overrides: RuntimeSettingsOverrides,
) -> DataSourceSettings:
    return DataSourceSettings(
        company_lookup_provider=overrides.company_lookup_provider
        or settings.company_lookup_provider,
        company_lookup_cache_ttl_days=overrides.company_lookup_cache_ttl_days
        or settings.company_lookup_cache_ttl_days,
        sec_user_agent=settings.sec_user_agent,
        market_data_provider=overrides.market_data_provider or settings.market_data_provider,
        market_data_cache_ttl_days=overrides.market_data_cache_ttl_days
        or settings.market_data_cache_ttl_days,
        alpha_vantage_api_key=settings.alpha_vantage_api_key,
        financial_statement_provider=overrides.financial_statement_provider
        or settings.financial_statement_provider,
        financial_statement_cache_ttl_days=overrides.financial_statement_cache_ttl_days
        or settings.financial_statement_cache_ttl_days,
        filing_provider=overrides.filing_provider or settings.filing_provider,
        filing_cache_ttl_days=overrides.filing_cache_ttl_days or settings.filing_cache_ttl_days,
        filing_max_document_bytes=overrides.filing_max_document_bytes
        or settings.filing_max_document_bytes,
    )


def _replace_retrieval_settings(
    settings: RetrievalSettings,
    overrides: RuntimeSettingsOverrides,
) -> RetrievalSettings:
    return RetrievalSettings(
        provider=overrides.retrieval_provider or settings.provider,
        top_k=overrides.retrieval_top_k or settings.top_k,
        min_score=(
            overrides.retrieval_min_score
            if overrides.retrieval_min_score is not None
            else settings.min_score
        ),
    )


def _replace_background_settings(
    settings: BackgroundSettings,
    overrides: RuntimeSettingsOverrides,
) -> BackgroundSettings:
    return BackgroundSettings(
        max_concurrent_research_runs=overrides.background_max_concurrent_research_runs
        or settings.max_concurrent_research_runs
    )
