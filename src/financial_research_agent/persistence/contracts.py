from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol


class PersistenceErrorCode(StrEnum):
    STORAGE_MIGRATION_REQUIRED = "storage_migration_required"
    DATABASE_BUSY = "database_busy"
    INTEGRITY_FAILED = "integrity_failed"
    INCOMPATIBLE_SCHEMA = "incompatible_schema"
    LEGACY_IMPORT_FAILED = "legacy_import_failed"
    BACKUP_FAILED = "backup_failed"
    RESTORE_FAILED = "restore_failed"
    INVALID_BACKUP_ID = "invalid_backup_id"


class PersistenceError(RuntimeError):
    def __init__(self, code: PersistenceErrorCode, message: str) -> None:
        self.code = PersistenceErrorCode(code)
        super().__init__(message)

    def to_dict(self) -> dict[str, str]:
        return {"error": self.code.value, "message": str(self)}


@dataclass(frozen=True, slots=True)
class IntegrityReport:
    checked_at: datetime
    database_path: Path
    schema_version: int
    database_size_bytes: int
    quick_check: tuple[str, ...]
    foreign_key_violations: tuple[str, ...]
    counts: Mapping[str, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "counts", MappingProxyType(dict(self.counts)))

    @property
    def healthy(self) -> bool:
        return self.quick_check == ("ok",) and not self.foreign_key_violations

    def to_dict(self) -> dict[str, object]:
        return {
            "checked_at": self.checked_at.isoformat(),
            "database_path": str(self.database_path),
            "schema_version": self.schema_version,
            "database_size_bytes": self.database_size_bytes,
            "healthy": self.healthy,
            "quick_check": list(self.quick_check),
            "foreign_key_violations": list(self.foreign_key_violations),
            "counts": dict(self.counts),
        }


@dataclass(frozen=True, slots=True)
class BackupRecord:
    id: str
    created_at: datetime
    database_size_bytes: int
    sha256: str
    schema_version: int

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "database_size_bytes": self.database_size_bytes,
            "sha256": self.sha256,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True, slots=True)
class MigrationImportResult:
    database_path: Path
    backup: BackupRecord | None
    imported_counts: Mapping[str, int]
    archived_paths: tuple[Path, ...]
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "imported_counts", MappingProxyType(dict(self.imported_counts)))

    def to_dict(self) -> dict[str, object]:
        return {
            "database_path": str(self.database_path),
            "backup": self.backup.to_dict() if self.backup is not None else None,
            "imported_counts": dict(self.imported_counts),
            "archived_paths": [str(path) for path in self.archived_paths],
            "warnings": list(self.warnings),
        }


class Repository(Protocol):
    def count(self) -> int: ...

    def clear(self) -> int: ...


class ChatSessionRepository(Repository, Protocol):
    def create(self) -> Any: ...

    def list(self, *, limit: int = 50) -> tuple[Any, ...]: ...

    def get(self, session_id: str) -> Any | None: ...

    def delete(self, session_id: str) -> bool: ...


class AggregateRepository(Repository, Protocol):
    def save(self, value: Any) -> Any: ...

    def get(self, value_id: str) -> Any | None: ...

    def list(self) -> tuple[Any, ...]: ...
