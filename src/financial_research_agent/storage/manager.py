from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Self

from financial_research_agent.settings import Settings
from financial_research_agent.storage.contracts import (
    StorageArea,
    StorageDataset,
    StorageDatasetSpec,
    StorageEntry,
    StorageFormat,
    StorageManifest,
    StorageMigrationRecord,
    StorageMigrationResult,
    StorageOperationResult,
)

STORAGE_MIGRATION_STORE_VERSION = 1
LOCAL_STORAGE_LAYOUT_MIGRATION_ID = "0001_local_json_storage_layout"
LOCAL_STORAGE_LAYOUT_MIGRATION_DESCRIPTION = (
    "Create FRA_HOME data, cache, and logs directories for local JSON storage."
)


class LocalStorageManager:
    def __init__(
        self,
        *,
        settings: Settings,
        dataset_specs: tuple[StorageDatasetSpec, ...] | None = None,
    ) -> None:
        self.settings = settings
        self.dataset_specs = dataset_specs or default_storage_dataset_specs(settings)
        _reject_duplicate_specs(self.dataset_specs)

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        return cls(settings=settings)

    @property
    def app_home(self) -> Path:
        return self.settings.local_paths.app_home

    @property
    def migrations_path(self) -> Path:
        return self.settings.local_paths.data_dir / "storage_migrations.json"

    def migrate(self, *, now: datetime | None = None) -> StorageMigrationResult:
        applied_at = now or datetime.now(UTC)
        created_directories = self._ensure_directories()
        state = self._read_migration_state()
        existing_ids = {
            str(item.get("id")) for item in state.get("migrations", ()) if isinstance(item, dict)
        }
        if LOCAL_STORAGE_LAYOUT_MIGRATION_ID in existing_ids:
            return StorageMigrationResult(
                manifest_path=self.migrations_path,
                applied_migrations=(),
                skipped_migrations=(LOCAL_STORAGE_LAYOUT_MIGRATION_ID,),
                created_directories=tuple(created_directories),
            )

        migration = StorageMigrationRecord(
            id=LOCAL_STORAGE_LAYOUT_MIGRATION_ID,
            description=LOCAL_STORAGE_LAYOUT_MIGRATION_DESCRIPTION,
            applied_at=applied_at,
        )
        migrations = [
            *(item for item in state.get("migrations", ()) if isinstance(item, dict)),
            migration.to_dict(),
        ]
        self._write_migration_state(
            {"version": STORAGE_MIGRATION_STORE_VERSION, "migrations": migrations}
        )
        return StorageMigrationResult(
            manifest_path=self.migrations_path,
            applied_migrations=(migration,),
            skipped_migrations=(),
            created_directories=tuple(created_directories),
        )

    def inspect(self, *, now: datetime | None = None) -> StorageManifest:
        checked_at = now or datetime.now(UTC)
        migrations, migration_warnings = self._migration_records()
        entries = tuple(self._entry_for_spec(spec, now=checked_at) for spec in self.dataset_specs)
        return StorageManifest(
            generated_at=checked_at,
            app_home=self.app_home,
            provider=self.settings.storage.provider,
            datasets=entries,
            migrations=migrations,
            warnings=migration_warnings,
        )

    def clear_cache(self) -> StorageOperationResult:
        targets = tuple(
            spec.path(app_home=self.app_home)
            for spec in self.dataset_specs
            if spec.clear_on_cache_clear
        )
        return self._delete_targets("clear_cache", targets)

    def reset_local_data(self, *, include_cache: bool = True) -> StorageOperationResult:
        targets = []
        for spec in self.dataset_specs:
            if spec.reset_on_data_reset or (include_cache and spec.clear_on_cache_clear):
                targets.append(spec.path(app_home=self.app_home))
        return self._delete_targets("reset_local_data", tuple(dict.fromkeys(targets)))

    def _ensure_directories(self) -> tuple[Path, ...]:
        created = []
        for path in (
            self.settings.local_paths.app_home,
            self.settings.local_paths.data_dir,
            self.settings.local_paths.cache_dir,
            self.settings.local_paths.logs_dir,
            self.settings.local_paths.data_dir / "filings",
        ):
            if not path.exists():
                created.append(path)
            path.mkdir(parents=True, exist_ok=True)
        return tuple(created)

    def _entry_for_spec(self, spec: StorageDatasetSpec, *, now: datetime) -> StorageEntry:
        path = spec.path(app_home=self.app_home)
        warnings: list[str] = []
        exists = path.exists()
        size_bytes, file_count, modified_at = _path_stats(path)
        schema_version = None
        record_count = None
        freshness_basis_at = None
        if exists and spec.storage_format == StorageFormat.JSON:
            payload, warning = _read_json_payload(path)
            if warning is not None:
                warnings.append(warning)
            if isinstance(payload, dict):
                schema_version = _payload_int(payload.get("version"))
                record_count = _payload_record_count(payload)
                freshness_basis_at = _payload_freshness_basis(payload)
        elif exists and spec.storage_format == StorageFormat.FILE_TREE:
            record_count = file_count
            freshness_basis_at = modified_at

        expires_at = None
        stale = None
        if exists and spec.ttl_days is not None:
            basis = freshness_basis_at or modified_at
            if basis is not None:
                expires_at = basis + timedelta(days=spec.ttl_days)
                stale = expires_at <= now
            else:
                warnings.append("Could not determine freshness timestamp for TTL check.")

        return StorageEntry(
            spec=spec,
            path=path,
            exists=exists,
            size_bytes=size_bytes,
            file_count=file_count,
            modified_at=modified_at,
            schema_version=schema_version,
            record_count=record_count,
            freshness_basis_at=freshness_basis_at,
            expires_at=expires_at,
            stale=stale,
            warnings=tuple(warnings),
        )

    def _delete_targets(
        self,
        operation: str,
        targets: tuple[Path, ...],
    ) -> StorageOperationResult:
        deleted_paths = []
        deleted_bytes = 0
        warnings = []
        for target in targets:
            self._assert_within_app_home(target)
            if not target.exists():
                continue
            size_bytes, _file_count, _modified_at = _path_stats(target)
            try:
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            except OSError as exc:
                warnings.append(f"Could not delete {target}: {exc}")
                continue
            deleted_paths.append(target)
            deleted_bytes += size_bytes
        return StorageOperationResult(
            operation=operation,
            deleted_paths=tuple(deleted_paths),
            deleted_bytes=deleted_bytes,
            warnings=tuple(warnings),
        )

    def _assert_within_app_home(self, target: Path) -> None:
        root = self.app_home.resolve()
        resolved = target.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"storage target must stay within FRA_HOME: {target}") from exc

    def _read_migration_state(self) -> dict[str, Any]:
        if not self.migrations_path.exists():
            return {"version": STORAGE_MIGRATION_STORE_VERSION, "migrations": []}
        try:
            payload = json.loads(self.migrations_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            message = f"Could not load storage migration state: {self.migrations_path}"
            raise ValueError(message) from exc
        if not isinstance(payload, dict):
            raise ValueError("storage migration state must be a JSON object")
        if payload.get("version") != STORAGE_MIGRATION_STORE_VERSION:
            raise ValueError("unsupported storage migration state version")
        migrations = payload.get("migrations", [])
        if not isinstance(migrations, list):
            raise ValueError("storage migration state migrations must be a list")
        return payload

    def _write_migration_state(self, payload: dict[str, Any]) -> None:
        self.migrations_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.migrations_path.with_suffix(f"{self.migrations_path.suffix}.tmp")
        temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temp_path.replace(self.migrations_path)

    def _migration_records(self) -> tuple[tuple[StorageMigrationRecord, ...], tuple[str, ...]]:
        if not self.migrations_path.exists():
            return (), ()
        try:
            state = self._read_migration_state()
            records = tuple(
                StorageMigrationRecord(
                    id=str(item["id"]),
                    description=str(item["description"]),
                    applied_at=_datetime_from_payload(item["applied_at"]),
                )
                for item in state.get("migrations", ())
                if isinstance(item, dict)
            )
        except (KeyError, TypeError, ValueError) as exc:
            return (), (f"Could not read storage migration state: {exc}",)
        return records, ()


def default_storage_dataset_specs(settings: Settings) -> tuple[StorageDatasetSpec, ...]:
    data_sources = settings.data_sources
    return (
        StorageDatasetSpec(
            dataset=StorageDataset.CHAT_SESSIONS,
            label="Chat sessions",
            area=StorageArea.DATA,
            relative_path="chat_sessions.json",
            storage_format=StorageFormat.JSON,
            schema_version=1,
            description="Persistent local chat session history and deterministic summaries.",
        ),
        StorageDatasetSpec(
            dataset=StorageDataset.COMPANY_LOOKUP_CACHE,
            label="SEC company lookup cache",
            area=StorageArea.CACHE,
            relative_path="sec_company_tickers.json",
            storage_format=StorageFormat.JSON,
            schema_version=1,
            ttl_days=data_sources.company_lookup_cache_ttl_days,
            description="Cached SEC company_tickers.json records for @company autocomplete.",
            clear_on_cache_clear=True,
        ),
        StorageDatasetSpec(
            dataset=StorageDataset.MARKET_DATA_PRICE_BARS,
            label="Market data price bars",
            area=StorageArea.DATA,
            relative_path="market_data_price_bars.json",
            storage_format=StorageFormat.JSON,
            schema_version=1,
            ttl_days=data_sources.market_data_cache_ttl_days,
            description="Alpha Vantage daily price history persisted as local research data.",
        ),
        StorageDatasetSpec(
            dataset=StorageDataset.FINANCIAL_STATEMENTS,
            label="Financial statements",
            area=StorageArea.DATA,
            relative_path="financial_statements.json",
            storage_format=StorageFormat.JSON,
            schema_version=1,
            ttl_days=data_sources.financial_statement_cache_ttl_days,
            description="Normalized SEC companyfacts statement results.",
        ),
        StorageDatasetSpec(
            dataset=StorageDataset.FILINGS_INDEX,
            label="Filing index",
            area=StorageArea.DATA,
            relative_path="filings/filings_index.json",
            storage_format=StorageFormat.JSON,
            schema_version=1,
            ttl_days=data_sources.filing_cache_ttl_days,
            description="SEC filing metadata and chunk index.",
        ),
        StorageDatasetSpec(
            dataset=StorageDataset.FILING_RAW_DOCUMENTS,
            label="Raw filing documents",
            area=StorageArea.DATA,
            relative_path="filings/raw",
            storage_format=StorageFormat.FILE_TREE,
            ttl_days=data_sources.filing_cache_ttl_days,
            description="Locally cached SEC primary HTML/TXT documents.",
        ),
        StorageDatasetSpec(
            dataset=StorageDataset.FILING_EXTRACTED_TEXT,
            label="Extracted filing text",
            area=StorageArea.DATA,
            relative_path="filings/text",
            storage_format=StorageFormat.FILE_TREE,
            ttl_days=data_sources.filing_cache_ttl_days,
            description="Extracted text files produced from SEC primary documents.",
        ),
        StorageDatasetSpec(
            dataset=StorageDataset.STORAGE_MIGRATIONS,
            label="Storage migrations",
            area=StorageArea.DATA,
            relative_path="storage_migrations.json",
            storage_format=StorageFormat.JSON,
            schema_version=STORAGE_MIGRATION_STORE_VERSION,
            description="Local migration bookkeeping for the FRA_HOME storage layout.",
            reset_on_data_reset=False,
        ),
    )


def _reject_duplicate_specs(specs: tuple[StorageDatasetSpec, ...]) -> None:
    seen = set()
    for spec in specs:
        if spec.dataset in seen:
            raise ValueError(f"duplicate storage dataset spec: {spec.dataset.value}")
        seen.add(spec.dataset)


def _path_stats(path: Path) -> tuple[int, int, datetime | None]:
    if not path.exists():
        return 0, 0, None
    if path.is_file():
        stat = path.stat()
        return stat.st_size, 1, _datetime_from_timestamp(stat.st_mtime)

    size_bytes = 0
    file_count = 0
    latest_modified: datetime | None = None
    for child in path.rglob("*"):
        if not child.is_file():
            continue
        try:
            stat = child.stat()
        except OSError:
            continue
        size_bytes += stat.st_size
        file_count += 1
        modified = _datetime_from_timestamp(stat.st_mtime)
        if latest_modified is None or modified > latest_modified:
            latest_modified = modified
    return size_bytes, file_count, latest_modified


def _read_json_payload(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"Could not read JSON metadata: {exc}"
    if not isinstance(payload, dict):
        return None, "JSON store root is not an object."
    return payload, None


def _payload_record_count(payload: dict[str, Any]) -> int | None:
    for key in ("sessions", "series", "results", "records"):
        value = payload.get(key)
        if isinstance(value, list):
            return len(value)
    return None


def _payload_freshness_basis(payload: dict[str, Any]) -> datetime | None:
    top_level = _optional_datetime_from_payload(payload.get("retrieved_at"))
    if top_level is not None:
        return top_level
    dates = []
    for key in ("series", "results"):
        items = payload.get(key)
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            source = item.get("source")
            if not isinstance(source, dict):
                continue
            retrieved_at = _optional_datetime_from_payload(source.get("retrieved_at"))
            if retrieved_at is not None:
                dates.append(retrieved_at)
    return max(dates) if dates else None


def _payload_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _optional_datetime_from_payload(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return _datetime_from_payload(value)
    except ValueError:
        return None


def _datetime_from_payload(value: object) -> datetime:
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def _datetime_from_timestamp(value: float) -> datetime:
    return datetime.fromtimestamp(value, tz=UTC)
