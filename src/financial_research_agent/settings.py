from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Self

DEFAULT_ENVIRONMENT = "local"
DEFAULT_LLM_PROVIDER = "offline-test"
DEFAULT_LLM_MODEL = "offline-test"
DEFAULT_LLM_LOCAL_RUNTIME = "llama.cpp"
DEFAULT_LLM_TIMEOUT_SECONDS = 30.0
DEFAULT_EMBEDDING_PROVIDER = "disabled"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1"
DEFAULT_ANTHROPIC_API_VERSION = "2023-06-01"
DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_LITELLM_BASE_URL = "http://127.0.0.1:4000/v1"
DEFAULT_CHAT_HISTORY_RECENT_TURNS = 6
DEFAULT_CHAT_HISTORY_SUMMARY_MAX_CHARS = 1200
DEFAULT_COMPANY_LOOKUP_PROVIDER = "sec"
DEFAULT_COMPANY_LOOKUP_CACHE_TTL_DAYS = 30
DEFAULT_SEC_USER_AGENT = (
    "financial-research-agent/0.1 local-research contact@financial-research-agent.local"
)
DEFAULT_MARKET_DATA_PROVIDER = "alpha-vantage"
DEFAULT_MARKET_DATA_CACHE_TTL_DAYS = 1
DEFAULT_FINANCIAL_STATEMENT_PROVIDER = "sec-companyfacts"
DEFAULT_FINANCIAL_STATEMENT_CACHE_TTL_DAYS = 30
DEFAULT_FILING_PROVIDER = "sec-edgar"
DEFAULT_FILING_CACHE_TTL_DAYS = 30
DEFAULT_FILING_MAX_DOCUMENT_BYTES = 8_000_000
DEFAULT_STORAGE_PROVIDER = "sqlite"
DEFAULT_RETRIEVAL_PROVIDER = "local-vector"
DEFAULT_RETRIEVAL_TOP_K = 5
DEFAULT_RETRIEVAL_MIN_SCORE = 0.0
DEFAULT_INTEROP_ENABLED = False
DEFAULT_INTEROP_LOCAL_ONLY = True
DEFAULT_BACKGROUND_MAX_CONCURRENT_RESEARCH_RUNS = 1
DEFAULT_PROMPT_BUDGET_INPUT_TOKENS = 16_000
DEFAULT_PROMPT_BUDGET_OUTPUT_TOKENS = 1_024
DEFAULT_EMBEDDING_CACHE_ENABLED = True
DEFAULT_ALLOW_REMOTE_BIND = False
EXPLICIT_MODEL_PROVIDERS = frozenset({"anthropic", "gemini", "litellm", "openai"})


class ProviderTask(StrEnum):
    CHAT = "chat"
    TOOL_CALLING = "tool_calling"
    STRUCTURED_OUTPUT = "structured_output"
    STREAMING = "streaming"
    EMBEDDINGS = "embeddings"


@dataclass(frozen=True, slots=True)
class LocalPaths:
    app_home: Path
    data_dir: Path
    cache_dir: Path
    logs_dir: Path

    @classmethod
    def from_app_home(cls, app_home: str | Path) -> Self:
        root = _normalize_path(app_home)
        return cls(
            app_home=root,
            data_dir=root / "data",
            cache_dir=root / "cache",
            logs_dir=root / "logs",
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "app_home": str(self.app_home),
            "data_dir": str(self.data_dir),
            "cache_dir": str(self.cache_dir),
            "logs_dir": str(self.logs_dir),
        }


@dataclass(frozen=True, slots=True)
class TaskProviderSelection:
    provider: str
    model: str | None
    base_url: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
        }


@dataclass(frozen=True, slots=True)
class ProviderSettings:
    llm_provider: str = DEFAULT_LLM_PROVIDER
    llm_model: str = DEFAULT_LLM_MODEL
    llm_base_url: str | None = None
    llm_local_runtime: str = DEFAULT_LLM_LOCAL_RUNTIME
    llm_timeout_seconds: float = DEFAULT_LLM_TIMEOUT_SECONDS
    embedding_provider: str = DEFAULT_EMBEDDING_PROVIDER
    embedding_model: str | None = None
    openai_api_key: str | None = None
    openai_base_url: str = DEFAULT_OPENAI_BASE_URL
    openai_organization: str | None = None
    openai_project: str | None = None
    anthropic_api_key: str | None = None
    anthropic_base_url: str = DEFAULT_ANTHROPIC_BASE_URL
    anthropic_api_version: str = DEFAULT_ANTHROPIC_API_VERSION
    gemini_api_key: str | None = None
    gemini_base_url: str = DEFAULT_GEMINI_BASE_URL
    litellm_api_key: str | None = None
    litellm_base_url: str = DEFAULT_LITELLM_BASE_URL
    chat_provider: str | None = None
    chat_model: str | None = None
    tool_calling_provider: str | None = None
    tool_calling_model: str | None = None
    structured_output_provider: str | None = None
    structured_output_model: str | None = None
    streaming_provider: str | None = None
    streaming_model: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "llm_local_runtime", _require_text(self.llm_local_runtime))
        object.__setattr__(self, "openai_api_key", _optional_text(self.openai_api_key))
        object.__setattr__(self, "openai_base_url", _require_text(self.openai_base_url))
        object.__setattr__(self, "openai_organization", _optional_text(self.openai_organization))
        object.__setattr__(self, "openai_project", _optional_text(self.openai_project))
        object.__setattr__(self, "anthropic_api_key", _optional_text(self.anthropic_api_key))
        object.__setattr__(self, "anthropic_base_url", _require_text(self.anthropic_base_url))
        object.__setattr__(
            self,
            "anthropic_api_version",
            _require_text(self.anthropic_api_version),
        )
        object.__setattr__(self, "gemini_api_key", _optional_text(self.gemini_api_key))
        object.__setattr__(self, "gemini_base_url", _require_text(self.gemini_base_url))
        object.__setattr__(self, "litellm_api_key", _optional_text(self.litellm_api_key))
        object.__setattr__(self, "litellm_base_url", _require_text(self.litellm_base_url))
        if self.llm_timeout_seconds <= 0:
            raise ValueError("llm_timeout_seconds must be positive")
        self._validate_explicit_models()

    def to_dict(self) -> dict[str, object]:
        return {
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "llm_base_url": self.llm_base_url,
            "llm_local_runtime": self.llm_local_runtime,
            "llm_timeout_seconds": self.llm_timeout_seconds,
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "openai_api_key_configured": self.openai_api_key is not None,
            "openai_base_url": self.openai_base_url,
            "openai_organization_configured": self.openai_organization is not None,
            "openai_project_configured": self.openai_project is not None,
            "anthropic_api_key_configured": self.anthropic_api_key is not None,
            "anthropic_base_url": self.anthropic_base_url,
            "anthropic_api_version": self.anthropic_api_version,
            "gemini_api_key_configured": self.gemini_api_key is not None,
            "gemini_base_url": self.gemini_base_url,
            "litellm_api_key_configured": self.litellm_api_key is not None,
            "litellm_base_url": self.litellm_base_url,
            "chat_provider": self.chat_provider,
            "chat_model": self.chat_model,
            "tool_calling_provider": self.tool_calling_provider,
            "tool_calling_model": self.tool_calling_model,
            "structured_output_provider": self.structured_output_provider,
            "structured_output_model": self.structured_output_model,
            "streaming_provider": self.streaming_provider,
            "streaming_model": self.streaming_model,
            "task_defaults": {
                task.value: self.selection_for_task(task).to_dict() for task in ProviderTask
            },
        }

    def selection_for_task(self, task: ProviderTask | str) -> TaskProviderSelection:
        provider_task = ProviderTask(task)
        if provider_task == ProviderTask.CHAT:
            return self._llm_selection(self.chat_provider, self.chat_model)
        if provider_task == ProviderTask.TOOL_CALLING:
            return self._llm_selection(self.tool_calling_provider, self.tool_calling_model)
        if provider_task == ProviderTask.STRUCTURED_OUTPUT:
            return self._llm_selection(
                self.structured_output_provider,
                self.structured_output_model,
            )
        if provider_task == ProviderTask.STREAMING:
            return self._llm_selection(self.streaming_provider, self.streaming_model)
        return TaskProviderSelection(
            provider=self.embedding_provider,
            model=self.embedding_model,
            base_url=None,
        )

    def _llm_selection(
        self,
        provider_override: str | None,
        model_override: str | None,
    ) -> TaskProviderSelection:
        return TaskProviderSelection(
            provider=provider_override or self.llm_provider,
            model=model_override or self.llm_model,
            base_url=self.llm_base_url,
        )

    def _validate_explicit_models(self) -> None:
        selections = (
            ("FRA_LLM_MODEL", self.llm_provider, self.llm_model),
            (
                "FRA_CHAT_MODEL",
                self.chat_provider or self.llm_provider,
                self.chat_model or self.llm_model,
            ),
            (
                "FRA_TOOL_CALLING_MODEL",
                self.tool_calling_provider or self.llm_provider,
                self.tool_calling_model or self.llm_model,
            ),
            (
                "FRA_STRUCTURED_OUTPUT_MODEL",
                self.structured_output_provider or self.llm_provider,
                self.structured_output_model or self.llm_model,
            ),
            (
                "FRA_STREAMING_MODEL",
                self.streaming_provider or self.llm_provider,
                self.streaming_model or self.llm_model,
            ),
            ("FRA_EMBEDDING_MODEL", self.embedding_provider, self.embedding_model),
        )
        for setting_name, provider, model in selections:
            if provider in EXPLICIT_MODEL_PROVIDERS and (
                model is None or model == DEFAULT_LLM_MODEL
            ):
                raise ValueError(
                    f"{setting_name} must be explicitly configured for provider {provider!r}"
                )


@dataclass(frozen=True, slots=True)
class ChatSettings:
    history_recent_turns: int = DEFAULT_CHAT_HISTORY_RECENT_TURNS
    history_summary_max_chars: int = DEFAULT_CHAT_HISTORY_SUMMARY_MAX_CHARS

    def __post_init__(self) -> None:
        if self.history_recent_turns <= 0:
            raise ValueError("history_recent_turns must be positive")
        if self.history_summary_max_chars <= 0:
            raise ValueError("history_summary_max_chars must be positive")

    def to_dict(self) -> dict[str, object]:
        return {
            "history_recent_turns": self.history_recent_turns,
            "history_summary_max_chars": self.history_summary_max_chars,
        }


@dataclass(frozen=True, slots=True)
class DataSourceSettings:
    company_lookup_provider: str = DEFAULT_COMPANY_LOOKUP_PROVIDER
    company_lookup_cache_ttl_days: int = DEFAULT_COMPANY_LOOKUP_CACHE_TTL_DAYS
    sec_user_agent: str = DEFAULT_SEC_USER_AGENT
    market_data_provider: str = DEFAULT_MARKET_DATA_PROVIDER
    market_data_cache_ttl_days: int = DEFAULT_MARKET_DATA_CACHE_TTL_DAYS
    alpha_vantage_api_key: str | None = None
    financial_statement_provider: str = DEFAULT_FINANCIAL_STATEMENT_PROVIDER
    financial_statement_cache_ttl_days: int = DEFAULT_FINANCIAL_STATEMENT_CACHE_TTL_DAYS
    filing_provider: str = DEFAULT_FILING_PROVIDER
    filing_cache_ttl_days: int = DEFAULT_FILING_CACHE_TTL_DAYS
    filing_max_document_bytes: int = DEFAULT_FILING_MAX_DOCUMENT_BYTES

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "company_lookup_provider",
            _require_text(self.company_lookup_provider),
        )
        object.__setattr__(self, "sec_user_agent", _require_text(self.sec_user_agent))
        object.__setattr__(self, "market_data_provider", _require_text(self.market_data_provider))
        object.__setattr__(
            self,
            "financial_statement_provider",
            _require_text(self.financial_statement_provider),
        )
        object.__setattr__(self, "filing_provider", _require_text(self.filing_provider))
        object.__setattr__(
            self, "alpha_vantage_api_key", _optional_text(self.alpha_vantage_api_key)
        )
        if self.company_lookup_cache_ttl_days <= 0:
            raise ValueError("company_lookup_cache_ttl_days must be positive")
        if self.market_data_cache_ttl_days <= 0:
            raise ValueError("market_data_cache_ttl_days must be positive")
        if self.financial_statement_cache_ttl_days <= 0:
            raise ValueError("financial_statement_cache_ttl_days must be positive")
        if self.filing_cache_ttl_days <= 0:
            raise ValueError("filing_cache_ttl_days must be positive")
        if self.filing_max_document_bytes <= 0:
            raise ValueError("filing_max_document_bytes must be positive")

    def to_dict(self) -> dict[str, object]:
        return {
            "company_lookup_provider": self.company_lookup_provider,
            "company_lookup_cache_ttl_days": self.company_lookup_cache_ttl_days,
            "sec_user_agent": self.sec_user_agent,
            "market_data_provider": self.market_data_provider,
            "market_data_cache_ttl_days": self.market_data_cache_ttl_days,
            "alpha_vantage_api_key_configured": self.alpha_vantage_api_key is not None,
            "financial_statement_provider": self.financial_statement_provider,
            "financial_statement_cache_ttl_days": self.financial_statement_cache_ttl_days,
            "filing_provider": self.filing_provider,
            "filing_cache_ttl_days": self.filing_cache_ttl_days,
            "filing_max_document_bytes": self.filing_max_document_bytes,
        }


@dataclass(frozen=True, slots=True)
class StorageSettings:
    provider: str = DEFAULT_STORAGE_PROVIDER

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider", _require_text(self.provider))
        if self.provider not in {"sqlite", "local-json"}:
            raise ValueError("storage provider must be sqlite or local-json")

    def to_dict(self) -> dict[str, object]:
        return {"provider": self.provider}


@dataclass(frozen=True, slots=True)
class RetrievalSettings:
    provider: str = DEFAULT_RETRIEVAL_PROVIDER
    top_k: int = DEFAULT_RETRIEVAL_TOP_K
    min_score: float = DEFAULT_RETRIEVAL_MIN_SCORE

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider", _require_text(self.provider))
        if self.top_k <= 0:
            raise ValueError("retrieval top_k must be positive")
        if self.min_score < -1.0 or self.min_score > 1.0:
            raise ValueError("retrieval min_score must be between -1 and 1")

    def to_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "top_k": self.top_k,
            "min_score": self.min_score,
        }


@dataclass(frozen=True, slots=True)
class InteroperabilitySettings:
    enabled: bool = DEFAULT_INTEROP_ENABLED
    local_only: bool = DEFAULT_INTEROP_LOCAL_ONLY
    api_key: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "api_key", _optional_text(self.api_key))
        if self.enabled and not self.local_only and self.api_key is None:
            raise ValueError(
                "FRA_INTEROP_API_KEY is required when FRA_INTEROP_ENABLED=true and "
                "FRA_INTEROP_LOCAL_ONLY=false"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "local_only": self.local_only,
            "api_key_configured": self.api_key is not None,
            "protocols": ["a2a_discovery", "mcp_read_only"],
        }


@dataclass(frozen=True, slots=True)
class BackgroundSettings:
    max_concurrent_research_runs: int = DEFAULT_BACKGROUND_MAX_CONCURRENT_RESEARCH_RUNS

    def __post_init__(self) -> None:
        if self.max_concurrent_research_runs <= 0:
            raise ValueError("max_concurrent_research_runs must be positive")

    def to_dict(self) -> dict[str, object]:
        return {"max_concurrent_research_runs": self.max_concurrent_research_runs}


@dataclass(frozen=True, slots=True)
class PerformanceSettings:
    prompt_budget_input_tokens: int = DEFAULT_PROMPT_BUDGET_INPUT_TOKENS
    prompt_budget_output_tokens: int = DEFAULT_PROMPT_BUDGET_OUTPUT_TOKENS
    embedding_cache_enabled: bool = DEFAULT_EMBEDDING_CACHE_ENABLED

    def __post_init__(self) -> None:
        if self.prompt_budget_input_tokens <= 0:
            raise ValueError("prompt_budget_input_tokens must be positive")
        if self.prompt_budget_output_tokens <= 0:
            raise ValueError("prompt_budget_output_tokens must be positive")

    def to_dict(self) -> dict[str, object]:
        return {
            "prompt_budget_input_tokens": self.prompt_budget_input_tokens,
            "prompt_budget_output_tokens": self.prompt_budget_output_tokens,
            "embedding_cache_enabled": self.embedding_cache_enabled,
        }


@dataclass(frozen=True, slots=True)
class SecuritySettings:
    allow_remote_bind: bool = DEFAULT_ALLOW_REMOTE_BIND

    def to_dict(self) -> dict[str, object]:
        return {
            "allow_remote_bind": self.allow_remote_bind,
            "default_bind_host": "127.0.0.1",
            "tool_policy": "deny_by_default_name_and_permission_allowlist",
            "external_content": "untrusted_data",
            "secret_storage": "environment_only",
        }


@dataclass(frozen=True, slots=True)
class Settings:
    environment: str
    local_paths: LocalPaths
    provider: ProviderSettings
    chat: ChatSettings
    data_sources: DataSourceSettings
    storage: StorageSettings
    retrieval: RetrievalSettings
    interoperability: InteroperabilitySettings
    background: BackgroundSettings
    performance: PerformanceSettings
    security: SecuritySettings

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Self:
        env = os.environ if environ is None else environ
        app_home = _env_value(env, "FRA_HOME", str(Path.home() / ".financial-research-agent"))

        return cls(
            environment=_env_value(env, "FRA_ENV", DEFAULT_ENVIRONMENT),
            local_paths=LocalPaths.from_app_home(app_home),
            provider=ProviderSettings(
                llm_provider=_env_value(env, "FRA_LLM_PROVIDER", DEFAULT_LLM_PROVIDER),
                llm_model=_env_value(env, "FRA_LLM_MODEL", DEFAULT_LLM_MODEL),
                llm_base_url=_env_optional(env, "FRA_LLM_BASE_URL"),
                llm_local_runtime=_env_value(
                    env,
                    "FRA_LLM_LOCAL_RUNTIME",
                    DEFAULT_LLM_LOCAL_RUNTIME,
                ),
                llm_timeout_seconds=_env_float_value(
                    env,
                    "FRA_LLM_TIMEOUT_SECONDS",
                    DEFAULT_LLM_TIMEOUT_SECONDS,
                ),
                embedding_provider=_env_value(
                    env,
                    "FRA_EMBEDDING_PROVIDER",
                    DEFAULT_EMBEDDING_PROVIDER,
                ),
                embedding_model=_env_optional(env, "FRA_EMBEDDING_MODEL"),
                openai_api_key=_env_optional_any(env, "FRA_OPENAI_API_KEY", "OPENAI_API_KEY"),
                openai_base_url=_env_value(env, "FRA_OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL),
                openai_organization=_env_optional_any(
                    env,
                    "FRA_OPENAI_ORGANIZATION",
                    "OPENAI_ORG_ID",
                ),
                openai_project=_env_optional_any(env, "FRA_OPENAI_PROJECT", "OPENAI_PROJECT_ID"),
                anthropic_api_key=_env_optional_any(
                    env,
                    "FRA_ANTHROPIC_API_KEY",
                    "ANTHROPIC_API_KEY",
                ),
                anthropic_base_url=_env_value(
                    env,
                    "FRA_ANTHROPIC_BASE_URL",
                    DEFAULT_ANTHROPIC_BASE_URL,
                ),
                anthropic_api_version=_env_value(
                    env,
                    "FRA_ANTHROPIC_API_VERSION",
                    DEFAULT_ANTHROPIC_API_VERSION,
                ),
                gemini_api_key=_env_optional_any(
                    env,
                    "FRA_GEMINI_API_KEY",
                    "GEMINI_API_KEY",
                ),
                gemini_base_url=_env_value(
                    env,
                    "FRA_GEMINI_BASE_URL",
                    DEFAULT_GEMINI_BASE_URL,
                ),
                litellm_api_key=_env_optional(env, "FRA_LITELLM_API_KEY"),
                litellm_base_url=_env_value(
                    env,
                    "FRA_LITELLM_BASE_URL",
                    DEFAULT_LITELLM_BASE_URL,
                ),
                chat_provider=_env_optional(env, "FRA_CHAT_PROVIDER"),
                chat_model=_env_optional(env, "FRA_CHAT_MODEL"),
                tool_calling_provider=_env_optional(env, "FRA_TOOL_CALLING_PROVIDER"),
                tool_calling_model=_env_optional(env, "FRA_TOOL_CALLING_MODEL"),
                structured_output_provider=_env_optional(env, "FRA_STRUCTURED_OUTPUT_PROVIDER"),
                structured_output_model=_env_optional(env, "FRA_STRUCTURED_OUTPUT_MODEL"),
                streaming_provider=_env_optional(env, "FRA_STREAMING_PROVIDER"),
                streaming_model=_env_optional(env, "FRA_STREAMING_MODEL"),
            ),
            chat=ChatSettings(
                history_recent_turns=_env_int_value(
                    env,
                    "FRA_CHAT_HISTORY_RECENT_TURNS",
                    DEFAULT_CHAT_HISTORY_RECENT_TURNS,
                ),
                history_summary_max_chars=_env_int_value(
                    env,
                    "FRA_CHAT_HISTORY_SUMMARY_MAX_CHARS",
                    DEFAULT_CHAT_HISTORY_SUMMARY_MAX_CHARS,
                ),
            ),
            data_sources=DataSourceSettings(
                company_lookup_provider=_env_value(
                    env,
                    "FRA_COMPANY_LOOKUP_PROVIDER",
                    DEFAULT_COMPANY_LOOKUP_PROVIDER,
                ),
                company_lookup_cache_ttl_days=_env_int_value(
                    env,
                    "FRA_COMPANY_LOOKUP_CACHE_TTL_DAYS",
                    DEFAULT_COMPANY_LOOKUP_CACHE_TTL_DAYS,
                ),
                sec_user_agent=_env_value(env, "FRA_SEC_USER_AGENT", DEFAULT_SEC_USER_AGENT),
                market_data_provider=_env_value(
                    env,
                    "FRA_MARKET_DATA_PROVIDER",
                    DEFAULT_MARKET_DATA_PROVIDER,
                ),
                market_data_cache_ttl_days=_env_int_value(
                    env,
                    "FRA_MARKET_DATA_CACHE_TTL_DAYS",
                    DEFAULT_MARKET_DATA_CACHE_TTL_DAYS,
                ),
                alpha_vantage_api_key=_env_optional_any(
                    env,
                    "FRA_ALPHA_VANTAGE_API_KEY",
                    "ALPHA_VANTAGE_API_KEY",
                ),
                financial_statement_provider=_env_value(
                    env,
                    "FRA_FINANCIAL_STATEMENT_PROVIDER",
                    DEFAULT_FINANCIAL_STATEMENT_PROVIDER,
                ),
                financial_statement_cache_ttl_days=_env_int_value(
                    env,
                    "FRA_FINANCIAL_STATEMENT_CACHE_TTL_DAYS",
                    DEFAULT_FINANCIAL_STATEMENT_CACHE_TTL_DAYS,
                ),
                filing_provider=_env_value(
                    env,
                    "FRA_FILING_PROVIDER",
                    DEFAULT_FILING_PROVIDER,
                ),
                filing_cache_ttl_days=_env_int_value(
                    env,
                    "FRA_FILING_CACHE_TTL_DAYS",
                    DEFAULT_FILING_CACHE_TTL_DAYS,
                ),
                filing_max_document_bytes=_env_int_value(
                    env,
                    "FRA_FILING_MAX_DOCUMENT_BYTES",
                    DEFAULT_FILING_MAX_DOCUMENT_BYTES,
                ),
            ),
            storage=StorageSettings(
                provider=_env_value(env, "FRA_STORAGE_PROVIDER", DEFAULT_STORAGE_PROVIDER),
            ),
            retrieval=RetrievalSettings(
                provider=_env_value(env, "FRA_RETRIEVAL_PROVIDER", DEFAULT_RETRIEVAL_PROVIDER),
                top_k=_env_int_value(env, "FRA_RETRIEVAL_TOP_K", DEFAULT_RETRIEVAL_TOP_K),
                min_score=_env_float_between_value(
                    env,
                    "FRA_RETRIEVAL_MIN_SCORE",
                    DEFAULT_RETRIEVAL_MIN_SCORE,
                    minimum=-1.0,
                    maximum=1.0,
                ),
            ),
            interoperability=InteroperabilitySettings(
                enabled=_env_bool_value(
                    env,
                    "FRA_INTEROP_ENABLED",
                    DEFAULT_INTEROP_ENABLED,
                ),
                local_only=_env_bool_value(
                    env,
                    "FRA_INTEROP_LOCAL_ONLY",
                    DEFAULT_INTEROP_LOCAL_ONLY,
                ),
                api_key=_env_optional(env, "FRA_INTEROP_API_KEY"),
            ),
            background=BackgroundSettings(
                max_concurrent_research_runs=_env_int_value(
                    env,
                    "FRA_BACKGROUND_MAX_CONCURRENT_RESEARCH_RUNS",
                    DEFAULT_BACKGROUND_MAX_CONCURRENT_RESEARCH_RUNS,
                ),
            ),
            performance=PerformanceSettings(
                prompt_budget_input_tokens=_env_int_value(
                    env,
                    "FRA_PROMPT_BUDGET_INPUT_TOKENS",
                    DEFAULT_PROMPT_BUDGET_INPUT_TOKENS,
                ),
                prompt_budget_output_tokens=_env_int_value(
                    env,
                    "FRA_PROMPT_BUDGET_OUTPUT_TOKENS",
                    DEFAULT_PROMPT_BUDGET_OUTPUT_TOKENS,
                ),
                embedding_cache_enabled=_env_bool_value(
                    env,
                    "FRA_EMBEDDING_CACHE_ENABLED",
                    DEFAULT_EMBEDDING_CACHE_ENABLED,
                ),
            ),
            security=SecuritySettings(
                allow_remote_bind=_env_bool_value(
                    env,
                    "FRA_ALLOW_REMOTE_BIND",
                    DEFAULT_ALLOW_REMOTE_BIND,
                ),
            ),
        )


def _env_value(environ: Mapping[str, str], name: str, default: str) -> str:
    value = environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _env_optional(environ: Mapping[str, str], name: str) -> str | None:
    value = environ.get(name)
    if value is None or value.strip() == "":
        return None
    return value.strip()


def _env_optional_any(environ: Mapping[str, str], *names: str) -> str | None:
    for name in names:
        value = _env_optional(environ, name)
        if value is not None:
            return value
    return None


def _env_float_value(environ: Mapping[str, str], name: str, default: float) -> float:
    value = environ.get(name)
    if value is None or value.strip() == "":
        return default
    try:
        result = float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _env_int_value(environ: Mapping[str, str], name: str, default: int) -> int:
    value = environ.get(name)
    if value is None or value.strip() == "":
        return default
    try:
        result = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _env_float_between_value(
    environ: Mapping[str, str],
    name: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    value = environ.get(name)
    if value is None or value.strip() == "":
        return default
    try:
        result = float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if result < minimum or result > maximum:
        raise ValueError(f"{name} must be between {minimum:g} and {maximum:g}")
    return result


def _env_bool_value(environ: Mapping[str, str], name: str, default: bool) -> bool:
    value = environ.get(name)
    if value is None or value.strip() == "":
        return default
    normalized = value.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


def _require_text(value: str) -> str:
    text = value.strip()
    if text == "":
        raise ValueError("value is required")
    return text


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    return text or None


def _normalize_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path
