from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path


class StorageArea(StrEnum):
    DATA = "data"
    CACHE = "cache"
    LOGS = "logs"


class StorageFormat(StrEnum):
    JSON = "json"
    FILE_TREE = "file_tree"


class StorageDataset(StrEnum):
    CHAT_SESSIONS = "chat_sessions"
    COMPANY_LOOKUP_CACHE = "company_lookup_cache"
    MARKET_DATA_PRICE_BARS = "market_data_price_bars"
    FINANCIAL_STATEMENTS = "financial_statements"
    FILINGS_INDEX = "filings_index"
    FILING_RAW_DOCUMENTS = "filing_raw_documents"
    FILING_EXTRACTED_TEXT = "filing_extracted_text"
    EMBEDDING_CACHE = "embedding_cache"
    RETRIEVAL_VECTOR_INDEX = "retrieval_vector_index"
    REPORT_RUNS = "report_runs"
    ORCHESTRATOR_RUNS = "orchestrator_runs"
    STORAGE_MIGRATIONS = "storage_migrations"


@dataclass(frozen=True, slots=True)
class StorageDatasetSpec:
    dataset: StorageDataset
    label: str
    area: StorageArea
    relative_path: str
    storage_format: StorageFormat
    description: str
    schema_version: int | None = None
    ttl_days: int | None = None
    contains_secrets: bool = False
    clear_on_cache_clear: bool = False
    reset_on_data_reset: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset", StorageDataset(self.dataset))
        object.__setattr__(self, "label", _require_text("label", self.label))
        object.__setattr__(self, "area", StorageArea(self.area))
        object.__setattr__(
            self,
            "storage_format",
            StorageFormat(self.storage_format),
        )
        object.__setattr__(
            self,
            "relative_path",
            _relative_path_text(self.relative_path),
        )
        object.__setattr__(self, "description", _require_text("description", self.description))
        if self.schema_version is not None and self.schema_version <= 0:
            raise ValueError("schema_version must be positive")
        if self.ttl_days is not None and self.ttl_days <= 0:
            raise ValueError("ttl_days must be positive")

    def path(self, *, app_home: Path) -> Path:
        root = app_home / self.area.value
        return root / self.relative_path

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.dataset.value,
            "label": self.label,
            "area": self.area.value,
            "relative_path": self.relative_path,
            "format": self.storage_format.value,
            "description": self.description,
            "schema_version": self.schema_version,
            "ttl_days": self.ttl_days,
            "contains_secrets": self.contains_secrets,
            "clear_on_cache_clear": self.clear_on_cache_clear,
            "reset_on_data_reset": self.reset_on_data_reset,
        }


@dataclass(frozen=True, slots=True)
class StorageEntry:
    spec: StorageDatasetSpec
    path: Path
    exists: bool
    size_bytes: int
    file_count: int
    modified_at: datetime | None = None
    schema_version: int | None = None
    record_count: int | None = None
    freshness_basis_at: datetime | None = None
    expires_at: datetime | None = None
    stale: bool | None = None
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.spec, StorageDatasetSpec):
            raise ValueError("spec must be StorageDatasetSpec")
        if self.size_bytes < 0:
            raise ValueError("size_bytes must be non-negative")
        if self.file_count < 0:
            raise ValueError("file_count must be non-negative")
        if self.schema_version is not None and self.schema_version <= 0:
            raise ValueError("schema_version must be positive")
        if self.record_count is not None and self.record_count < 0:
            raise ValueError("record_count must be non-negative")
        object.__setattr__(self, "warnings", _text_tuple("warnings", self.warnings))

    def to_dict(self) -> dict[str, object]:
        return {
            "spec": self.spec.to_dict(),
            "path": str(self.path),
            "exists": self.exists,
            "size_bytes": self.size_bytes,
            "file_count": self.file_count,
            "modified_at": self.modified_at.isoformat() if self.modified_at else None,
            "schema_version": self.schema_version,
            "record_count": self.record_count,
            "freshness_basis_at": (
                self.freshness_basis_at.isoformat() if self.freshness_basis_at else None
            ),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "stale": self.stale,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class StorageMigrationRecord:
    id: str
    description: str
    applied_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_text("id", self.id))
        object.__setattr__(self, "description", _require_text("description", self.description))
        object.__setattr__(self, "applied_at", _aware_datetime("applied_at", self.applied_at))

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "description": self.description,
            "applied_at": self.applied_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class StorageManifest:
    generated_at: datetime
    app_home: Path
    provider: str
    datasets: tuple[StorageEntry, ...]
    migrations: tuple[StorageMigrationRecord, ...]
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "generated_at", _aware_datetime("generated_at", self.generated_at))
        object.__setattr__(self, "provider", _require_text("provider", self.provider))
        object.__setattr__(self, "datasets", _entry_tuple(self.datasets))
        object.__setattr__(self, "migrations", _migration_tuple(self.migrations))
        object.__setattr__(self, "warnings", _text_tuple("warnings", self.warnings))

    def to_dict(self) -> dict[str, object]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "app_home": str(self.app_home),
            "provider": self.provider,
            "datasets": [entry.to_dict() for entry in self.datasets],
            "migrations": [migration.to_dict() for migration in self.migrations],
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class StorageMigrationResult:
    manifest_path: Path
    applied_migrations: tuple[StorageMigrationRecord, ...]
    skipped_migrations: tuple[str, ...]
    created_directories: tuple[Path, ...]
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "applied_migrations", _migration_tuple(self.applied_migrations))
        object.__setattr__(
            self,
            "skipped_migrations",
            _text_tuple("skipped_migrations", self.skipped_migrations),
        )
        object.__setattr__(self, "created_directories", tuple(self.created_directories))
        object.__setattr__(self, "warnings", _text_tuple("warnings", self.warnings))

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest_path": str(self.manifest_path),
            "applied_migrations": [migration.to_dict() for migration in self.applied_migrations],
            "skipped_migrations": list(self.skipped_migrations),
            "created_directories": [str(path) for path in self.created_directories],
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class StorageOperationResult:
    operation: str
    deleted_paths: tuple[Path, ...]
    deleted_bytes: int
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation", _require_text("operation", self.operation))
        object.__setattr__(self, "deleted_paths", tuple(self.deleted_paths))
        if self.deleted_bytes < 0:
            raise ValueError("deleted_bytes must be non-negative")
        object.__setattr__(self, "warnings", _text_tuple("warnings", self.warnings))

    @property
    def deleted_count(self) -> int:
        return len(self.deleted_paths)

    def to_dict(self) -> dict[str, object]:
        return {
            "operation": self.operation,
            "deleted_count": self.deleted_count,
            "deleted_paths": [str(path) for path in self.deleted_paths],
            "deleted_bytes": self.deleted_bytes,
            "warnings": list(self.warnings),
        }


def _entry_tuple(values: Iterable[StorageEntry]) -> tuple[StorageEntry, ...]:
    entries = tuple(values)
    for index, entry in enumerate(entries):
        if not isinstance(entry, StorageEntry):
            raise ValueError(f"datasets[{index}] must be StorageEntry")
    return entries


def _migration_tuple(
    values: Iterable[StorageMigrationRecord],
) -> tuple[StorageMigrationRecord, ...]:
    migrations = tuple(values)
    for index, migration in enumerate(migrations):
        if not isinstance(migration, StorageMigrationRecord):
            raise ValueError(f"migrations[{index}] must be StorageMigrationRecord")
    return migrations


def _text_tuple(name: str, values: Iterable[str]) -> tuple[str, ...]:
    if isinstance(values, str):
        raise ValueError(f"{name} must be an iterable of strings, not a string")
    return tuple(_require_text(f"{name}[{index}]", value) for index, value in enumerate(values))


def _relative_path_text(value: str) -> str:
    text = _require_text("relative_path", value).replace("\\", "/")
    path = Path(text)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("relative_path must stay within its storage area")
    return text


def _require_text(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    text = value.strip()
    if text == "":
        raise ValueError(f"{name} is required")
    return text


def _aware_datetime(name: str, value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError(f"{name} must be a datetime")
    if value.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value
