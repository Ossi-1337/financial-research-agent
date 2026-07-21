from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from types import MappingProxyType

from financial_research_agent.persistence.contracts import PersistenceError
from financial_research_agent.persistence.database import SQLiteDatabase
from financial_research_agent.persistence.factory import existing_legacy_paths
from financial_research_agent.persistence.legacy import LegacyJsonImporter
from financial_research_agent.persistence.operations import SQLiteOperations
from financial_research_agent.settings import Settings
from financial_research_agent.storage import LocalStorageManager


@dataclass(frozen=True, slots=True)
class SQLiteStorageStatus:
    generated_at: datetime
    provider: str
    schema_version: int
    database_size_bytes: int
    counts: Mapping[str, int]
    migration_warnings: tuple[str, ...]
    filesystem: object

    def __post_init__(self) -> None:
        object.__setattr__(self, "counts", MappingProxyType(dict(self.counts)))

    def to_dict(self) -> dict[str, object]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "provider": self.provider,
            "schema_version": self.schema_version,
            "database_size_bytes": self.database_size_bytes,
            "counts": dict(self.counts),
            "migration_warnings": list(self.migration_warnings),
            "filesystem": self.filesystem.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class SQLiteResetResult:
    database_counts: Mapping[str, int]
    filesystem_result: object

    def __post_init__(self) -> None:
        object.__setattr__(self, "database_counts", MappingProxyType(dict(self.database_counts)))

    def to_dict(self) -> dict[str, object]:
        return {
            "operation": "reset_local_data",
            "database_counts": dict(self.database_counts),
            "filesystem": self.filesystem_result.to_dict(),
        }


class SQLiteStorageManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.database = SQLiteDatabase.from_data_dir(settings.local_paths.data_dir)
        self.filesystem = LocalStorageManager.from_settings(settings)
        self.dataset_specs = self.filesystem.dataset_specs
        self.operations = SQLiteOperations(self.database, app_home=settings.local_paths.app_home)

    def inspect(self) -> SQLiteStorageStatus:
        warnings = []
        if not self.database.path.exists():
            warnings.append("SQLite database has not been initialized.")
            if existing_legacy_paths(self.settings.local_paths.data_dir):
                warnings.append("storage_migration_required")
            counts = {}
            schema_version = 0
            size = 0
        else:
            try:
                report = self.database.integrity()
                counts = report.counts
                schema_version = report.schema_version
                size = report.database_size_bytes
                if not report.healthy:
                    warnings.append("SQLite integrity check failed.")
            except PersistenceError as exc:
                counts = {}
                schema_version = self.database.schema_version()
                size = self.database.path.stat().st_size
                warnings.append(exc.code.value)
        return SQLiteStorageStatus(
            generated_at=datetime.now(UTC),
            provider="sqlite",
            schema_version=schema_version,
            database_size_bytes=size,
            counts=counts,
            migration_warnings=tuple(warnings),
            filesystem=self.filesystem.inspect(),
        )

    def migrate(self):
        return LegacyJsonImporter(self.settings).migrate()

    def clear_cache(self):
        return self.filesystem.clear_cache()

    def reset_local_data(self, *, include_cache: bool = True) -> SQLiteResetResult:
        self.database.initialize()
        counts = self.operations.reset_data()
        files = self.filesystem.reset_local_data(include_cache=include_cache)
        return SQLiteResetResult(counts, files)


def create_storage_manager(settings: Settings):
    if settings.storage.provider == "sqlite":
        return SQLiteStorageManager(settings)
    return LocalStorageManager.from_settings(settings)
