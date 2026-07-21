from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from financial_research_agent.persistence.contracts import (
    BackupRecord,
    PersistenceError,
    PersistenceErrorCode,
)
from financial_research_agent.persistence.database import CURRENT_SCHEMA_VERSION, SQLiteDatabase

BACKUP_ID_PATTERN = re.compile(r"^backup_[0-9]{8}T[0-9]{6}Z_[0-9a-f]{8}$")
BACKUP_DATABASE_NAME = "financial_research_agent.sqlite3"
BACKUP_MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True, slots=True)
class CleanupResult:
    dataset: str
    cutoff: datetime
    dry_run: bool
    matched_records: int
    deleted_records: int

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "cutoff": self.cutoff.isoformat(),
            "dry_run": self.dry_run,
            "matched_records": self.matched_records,
            "deleted_records": self.deleted_records,
        }


class SQLiteOperations:
    def __init__(self, database: SQLiteDatabase, *, app_home: Path) -> None:
        self.database = database
        self.app_home = app_home
        self.backups_dir = app_home / "backups"

    def backup(self, *, prefix: str = "backup") -> BackupRecord:
        self.database.initialize()
        now = datetime.now(UTC)
        backup_id = f"{prefix}_{now:%Y%m%dT%H%M%SZ}_{uuid4().hex[:8]}"
        temp_dir = self.backups_dir / f".{backup_id}.tmp"
        final_dir = self.backups_dir / backup_id
        if final_dir.exists() or temp_dir.exists():
            raise PersistenceError(PersistenceErrorCode.BACKUP_FAILED, "Backup ID collision.")
        temp_dir.mkdir(parents=True)
        backup_path = temp_dir / BACKUP_DATABASE_NAME
        try:
            source = self.database.connect()
            destination = sqlite3.connect(backup_path)
            try:
                source.backup(destination)
            finally:
                destination.close()
                source.close()
            record = _backup_record(backup_id, now, backup_path)
            _write_json(temp_dir / BACKUP_MANIFEST_NAME, record.to_dict())
            final_dir.parent.mkdir(parents=True, exist_ok=True)
            temp_dir.replace(final_dir)
            return record
        except Exception as exc:
            shutil.rmtree(temp_dir, ignore_errors=True)
            if isinstance(exc, PersistenceError):
                raise
            raise PersistenceError(
                PersistenceErrorCode.BACKUP_FAILED, "Could not create SQLite backup."
            ) from exc

    def validate_backup(self, backup_id: str) -> BackupRecord:
        if not BACKUP_ID_PATTERN.fullmatch(backup_id):
            raise PersistenceError(PersistenceErrorCode.INVALID_BACKUP_ID, "Backup ID is invalid.")
        directory = self.backups_dir / backup_id
        database_path = directory / BACKUP_DATABASE_NAME
        manifest_path = directory / BACKUP_MANIFEST_NAME
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            if payload.get("id") != backup_id:
                raise ValueError("backup manifest ID mismatch")
            expected_hash = str(payload["sha256"])
            if _sha256(database_path) != expected_hash:
                raise ValueError("backup checksum mismatch")
            candidate = SQLiteDatabase(database_path)
            if candidate.schema_version() != CURRENT_SCHEMA_VERSION:
                raise ValueError("backup schema version is incompatible")
            report = candidate.integrity(full=True)
            if not report.healthy or report.schema_version != CURRENT_SCHEMA_VERSION:
                raise ValueError("backup schema or integrity validation failed")
            return BackupRecord(
                id=str(payload["id"]),
                created_at=datetime.fromisoformat(str(payload["created_at"])),
                database_size_bytes=int(payload["database_size_bytes"]),
                sha256=expected_hash,
                schema_version=int(payload["schema_version"]),
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise PersistenceError(
                PersistenceErrorCode.RESTORE_FAILED, "Backup validation failed."
            ) from exc

    def restore(self, backup_id: str) -> dict[str, object]:
        record = self.validate_backup(backup_id)
        self._require_exclusive_access()
        pre_restore = self.backup()
        source = self.backups_dir / backup_id / BACKUP_DATABASE_NAME
        temp = self.database.path.with_suffix(f".restore-{uuid4().hex}.tmp")
        try:
            shutil.copy2(source, temp)
            candidate = SQLiteDatabase(temp)
            if not candidate.integrity(full=True).healthy:
                raise ValueError("restored database integrity check failed")
            candidate.prepare_for_atomic_move()
            os.replace(temp, self.database.path)
            _remove_sidecars(self.database.path)
        except Exception as exc:
            temp.unlink(missing_ok=True)
            _remove_sidecars(temp)
            raise PersistenceError(
                PersistenceErrorCode.RESTORE_FAILED, "Could not restore SQLite backup."
            ) from exc
        return {"restored_backup": record.to_dict(), "pre_restore_backup": pre_restore.to_dict()}

    def cleanup(
        self,
        *,
        dataset: str,
        older_than_days: int,
        confirmed: bool,
        include_source_documents: bool = False,
    ) -> CleanupResult:
        if older_than_days < 0:
            raise ValueError("older_than_days must be non-negative")
        if dataset == "report-exports":
            return self._cleanup_report_exports(older_than_days, confirmed)
        table, date_column = _cleanup_target(dataset)
        if dataset == "filing-documents" and not include_source_documents:
            raise ValueError("filing-documents requires --include-source-documents")
        cutoff = datetime.now(UTC) - timedelta(days=older_than_days)
        source_paths: tuple[Path, ...] = ()
        with self.database.read() as connection:
            matched = int(
                connection.execute(
                    f'SELECT COUNT(*) FROM "{table}" WHERE "{date_column}" < ?',
                    (cutoff.isoformat(),),
                ).fetchone()[0]
            )
            if dataset == "filing-documents" and confirmed:
                source_paths = tuple(
                    Path(value)
                    for row in connection.execute(
                        """
                        SELECT f.local_raw_path, f.local_text_path
                        FROM filings f
                        JOIN filing_results r ON r.id = f.result_id
                        WHERE r.retrieved_at < ?
                        """,
                        (cutoff.isoformat(),),
                    )
                    for value in row
                )
        deleted = 0
        if confirmed:
            safe_source_paths = tuple(
                self._validated_source_document_path(path) for path in source_paths
            )
            with self.database.transaction() as connection:
                cursor = connection.execute(
                    f'DELETE FROM "{table}" WHERE "{date_column}" < ?',
                    (cutoff.isoformat(),),
                )
                deleted = cursor.rowcount
            for path in safe_source_paths:
                path.unlink(missing_ok=True)
        return CleanupResult(dataset, cutoff, not confirmed, matched, deleted)

    def reset_data(self) -> dict[str, int]:
        tables = (
            "chat_sessions",
            "market_series",
            "statement_results",
            "filing_results",
            "cited_runs",
            "orchestrator_runs",
            "background_jobs",
            "runtime_settings",
            "securities",
            "companies",
        )
        counts: dict[str, int] = {}
        with self.database.transaction() as connection:
            for table in tables:
                counts[table] = int(
                    connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
                )
                connection.execute(f'DELETE FROM "{table}"')
        return counts

    def _require_exclusive_access(self) -> None:
        with self.database.transaction(exclusive=True):
            pass

    def _cleanup_report_exports(self, older_than_days: int, confirmed: bool) -> CleanupResult:
        cutoff = datetime.now(UTC) - timedelta(days=older_than_days)
        root = self.app_home / "data" / "exports"
        matches = []
        if root.exists():
            for directory in root.iterdir():
                manifest = directory / "manifest.json"
                if not directory.is_dir() or not manifest.exists():
                    continue
                try:
                    payload = json.loads(manifest.read_text(encoding="utf-8"))
                    created_at = datetime.fromisoformat(str(payload["created_at"]))
                except OSError, KeyError, ValueError, json.JSONDecodeError:
                    continue
                if created_at < cutoff:
                    matches.append(directory)
        if confirmed:
            for directory in matches:
                shutil.rmtree(directory)
        return CleanupResult(
            "report-exports", cutoff, not confirmed, len(matches), len(matches) if confirmed else 0
        )

    def _validated_source_document_path(self, path: Path) -> Path:
        root = (self.app_home / "data" / "filings").resolve()
        resolved = path.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise PersistenceError(
                PersistenceErrorCode.INTEGRITY_FAILED,
                "Stored filing path is outside FRA_HOME and was not deleted.",
            ) from exc
        return resolved


def backup_legacy_files(
    *, app_home: Path, data_dir: Path, paths: tuple[Path, ...]
) -> tuple[BackupRecord, tuple[Path, ...]]:
    now = datetime.now(UTC)
    backup_id = f"legacy_{now:%Y%m%dT%H%M%SZ}_{uuid4().hex[:8]}"
    temp_dir = app_home / "backups" / f".{backup_id}.tmp"
    final_dir = app_home / "backups" / backup_id
    copied = []
    files = []
    temp_dir.mkdir(parents=True)
    try:
        for source in paths:
            relative = source.relative_to(data_dir)
            destination = temp_dir / "legacy-data" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            copied.append(destination)
            files.append(
                {
                    "path": relative.as_posix(),
                    "size_bytes": destination.stat().st_size,
                    "sha256": _sha256(destination),
                }
            )
        manifest = {
            "id": backup_id,
            "created_at": now.isoformat(),
            "schema_version": 0,
            "files": files,
        }
        _write_json(temp_dir / BACKUP_MANIFEST_NAME, manifest)
        temp_dir.replace(final_dir)
    except Exception as exc:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise PersistenceError(
            PersistenceErrorCode.BACKUP_FAILED, "Could not back up legacy JSON stores."
        ) from exc
    total_size = sum(int(item["size_bytes"]) for item in files)
    combined_hash = hashlib.sha256(
        "".join(str(item["sha256"]) for item in files).encode("ascii")
    ).hexdigest()
    record = BackupRecord(backup_id, now, total_size, combined_hash, 0)
    archived = tuple(final_dir / path.relative_to(temp_dir) for path in copied)
    return record, archived


def _backup_record(backup_id: str, created_at: datetime, path: Path) -> BackupRecord:
    database = SQLiteDatabase(path)
    return BackupRecord(
        id=backup_id,
        created_at=created_at,
        database_size_bytes=path.stat().st_size,
        sha256=_sha256(path),
        schema_version=database.schema_version(),
    )


def _cleanup_target(dataset: str) -> tuple[str, str]:
    targets = {
        "chat-sessions": ("chat_sessions", "updated_at"),
        "research-runs": ("orchestrator_runs", "updated_at"),
        "cited-runs": ("cited_runs", "created_at"),
        "background-jobs": ("background_jobs", "updated_at"),
        "filing-documents": ("filing_results", "retrieved_at"),
    }
    try:
        return targets[dataset]
    except KeyError as exc:
        raise ValueError(f"Unsupported cleanup dataset: {dataset}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _remove_sidecars(database_path: Path) -> None:
    for suffix in ("-wal", "-shm"):
        Path(f"{database_path}{suffix}").unlink(missing_ok=True)
