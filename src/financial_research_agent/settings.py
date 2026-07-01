from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Self

DEFAULT_ENVIRONMENT = "local"
DEFAULT_LLM_PROVIDER = "offline-test"
DEFAULT_LLM_MODEL = "offline-test"
DEFAULT_EMBEDDING_PROVIDER = "disabled"


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
class ProviderSettings:
    llm_provider: str = DEFAULT_LLM_PROVIDER
    llm_model: str = DEFAULT_LLM_MODEL
    llm_base_url: str | None = None
    embedding_provider: str = DEFAULT_EMBEDDING_PROVIDER
    embedding_model: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "llm_base_url": self.llm_base_url,
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
        }


@dataclass(frozen=True, slots=True)
class Settings:
    environment: str
    local_paths: LocalPaths
    provider: ProviderSettings

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
                embedding_provider=_env_value(
                    env,
                    "FRA_EMBEDDING_PROVIDER",
                    DEFAULT_EMBEDDING_PROVIDER,
                ),
                embedding_model=_env_optional(env, "FRA_EMBEDDING_MODEL"),
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


def _normalize_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path
