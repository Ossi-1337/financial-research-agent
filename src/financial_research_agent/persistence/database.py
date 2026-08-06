from __future__ import annotations

import hashlib
import importlib.resources
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from threading import local
from time import monotonic, sleep
from uuid import uuid4

from financial_research_agent.persistence.contracts import (
    IntegrityReport,
    PersistenceError,
    PersistenceErrorCode,
)

DATABASE_FILENAME = "financial_research_agent.sqlite3"
CURRENT_SCHEMA_VERSION = 7
BUSY_TIMEOUT_MS = 5_000
COUNTED_TABLES = (
    "companies",
    "securities",
    "chat_sessions",
    "chat_messages",
    "market_series",
    "price_bars",
    "statement_results",
    "financial_statements",
    "filing_results",
    "filings",
    "filing_chunks",
    "cited_runs",
    "citations",
    "evidence_snippets",
    "orchestrator_runs",
    "agent_handoffs",
    "background_jobs",
    "a2a_tasks",
    "a2a_task_events",
    "a2a_delegations",
    "narrative_presentations",
    "web_source_evidence",
    "runtime_settings",
    "import_records",
)
REQUIRED_TABLES = (
    "schema_migrations",
    "app_metadata",
    "companies",
    "company_identifiers",
    "securities",
    "security_identifiers",
    "chat_sessions",
    "chat_messages",
    "market_series",
    "price_bars",
    "statement_results",
    "financial_statements",
    "filing_results",
    "filings",
    "filing_chunks",
    "cited_runs",
    "citations",
    "evidence_snippets",
    "orchestrator_runs",
    "agent_handoffs",
    "background_jobs",
    "a2a_tasks",
    "a2a_task_events",
    "a2a_delegations",
    "narrative_presentations",
    "web_source_evidence",
    "runtime_settings",
    "import_records",
)
REQUIRED_INDEXES = (
    "companies_cik_unique",
    "chat_sessions_updated_at_idx",
    "orchestrator_runs_updated_at_idx",
    "background_jobs_updated_at_idx",
    "a2a_tasks_updated_at_idx",
    "a2a_task_events_task_sequence_idx",
    "idx_a2a_tasks_owner_state_updated",
    "idx_a2a_delegations_run_role",
    "filing_chunks_page_idx",
    "narrative_presentations_cache_key_idx",
    "narrative_presentations_run_created_idx",
    "web_source_evidence_expires_at_idx",
)


class SQLiteDatabase:
    def __init__(self, path: Path, *, busy_timeout_ms: int = BUSY_TIMEOUT_MS) -> None:
        if busy_timeout_ms <= 0:
            raise ValueError("busy_timeout_ms must be positive")
        self.path = path
        self.busy_timeout_ms = busy_timeout_ms
        self._local = local()

    @property
    def maintenance_lock_path(self) -> Path:
        return self.path.with_name(f"{self.path.name}.maintenance.lock")

    @classmethod
    def from_data_dir(cls, data_dir: Path) -> SQLiteDatabase:
        return cls(data_dir / DATABASE_FILENAME)

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        SQLiteMigrationRunner(self).apply()

    def connect(self) -> sqlite3.Connection:
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(
                self.path,
                timeout=self.busy_timeout_ms / 1000,
                isolation_level=None,
                check_same_thread=False,
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute(f"PRAGMA busy_timeout = {self.busy_timeout_ms}")
            self._enable_wal(connection)
            return connection
        except sqlite3.OperationalError as exc:
            if connection is not None:
                connection.close()
            if _is_database_busy(exc):
                raise PersistenceError(
                    PersistenceErrorCode.DATABASE_BUSY,
                    "SQLite database is busy; stop active app processes and retry.",
                ) from exc
            raise

    def _enable_wal(self, connection: sqlite3.Connection) -> None:
        deadline = monotonic() + (self.busy_timeout_ms / 1000)
        while True:
            try:
                mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
                if mode != "wal":
                    connection.execute("PRAGMA journal_mode = WAL")
                return
            except sqlite3.OperationalError as exc:
                if not _is_database_busy(exc) or monotonic() >= deadline:
                    raise
                sleep(0.025)

    @contextmanager
    def read(self) -> Iterator[sqlite3.Connection]:
        active = getattr(self._local, "connection", None)
        if active is not None:
            yield active
            return
        connection = self.connect()
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def transaction(self, *, exclusive: bool = False) -> Iterator[sqlite3.Connection]:
        active = getattr(self._local, "connection", None)
        if active is not None:
            yield active
            return
        self._require_write_access()
        connection = self.connect()
        try:
            connection.execute("BEGIN EXCLUSIVE" if exclusive else "BEGIN IMMEDIATE")
            self._require_write_access()
            self._local.connection = connection
            yield connection
            connection.commit()
        except sqlite3.OperationalError as exc:
            connection.rollback()
            if _is_database_busy(exc):
                raise PersistenceError(
                    PersistenceErrorCode.DATABASE_BUSY,
                    "SQLite database is busy; stop active app processes and retry.",
                ) from exc
            raise
        except Exception:
            connection.rollback()
            raise
        finally:
            self._local.connection = None
            connection.close()

    @contextmanager
    def maintenance(self) -> Iterator[None]:
        if getattr(self._local, "maintenance_token", None) is not None:
            raise RuntimeError("maintenance lock cannot be nested")
        self.maintenance_lock_path.parent.mkdir(parents=True, exist_ok=True)
        token = uuid4().hex
        try:
            descriptor = os.open(
                self.maintenance_lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
        except FileExistsError as exc:
            raise _database_busy() from exc
        lock_initialized = False
        try:
            with os.fdopen(descriptor, "w", encoding="ascii") as stream:
                stream.write(token)
            lock_initialized = True
            self._local.maintenance_token = token
            yield
        finally:
            self._local.maintenance_token = None
            try:
                if not lock_initialized or (
                    self.maintenance_lock_path.read_text(encoding="ascii") == token
                ):
                    self.maintenance_lock_path.unlink(missing_ok=True)
            except FileNotFoundError:
                pass

    def _require_write_access(self) -> None:
        try:
            token = self.maintenance_lock_path.read_text(encoding="ascii")
        except FileNotFoundError:
            return
        if token != getattr(self._local, "maintenance_token", None):
            raise _database_busy()

    def schema_version(self) -> int:
        if not self.path.exists():
            return 0
        with self.read() as connection:
            table = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_migrations'"
            ).fetchone()
            if table is None:
                return 0
            row = connection.execute(
                "SELECT MAX(version) AS version FROM schema_migrations"
            ).fetchone()
            return int(row["version"] or 0)

    def integrity(self, *, full: bool = False) -> IntegrityReport:
        if not self.path.exists() or self.schema_version() == 0:
            raise PersistenceError(
                PersistenceErrorCode.INCOMPATIBLE_SCHEMA,
                "SQLite database is missing or has no migration history.",
            )
        SQLiteMigrationRunner(self).validate()
        pragma = "integrity_check" if full else "quick_check"
        with self.read() as connection:
            quick = tuple(str(row[0]) for row in connection.execute(f"PRAGMA {pragma}"))
            foreign_keys = tuple(
                ":".join(str(value) for value in row)
                for row in connection.execute("PRAGMA foreign_key_check")
            )
            counts = {
                table: int(connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
                for table in COUNTED_TABLES
            }
        return IntegrityReport(
            checked_at=datetime.now(UTC),
            database_path=self.path,
            schema_version=self.schema_version(),
            database_size_bytes=self.path.stat().st_size if self.path.exists() else 0,
            quick_check=quick,
            foreign_key_violations=foreign_keys,
            counts=counts,
        )

    def prepare_for_atomic_move(self) -> None:
        with self.read() as connection:
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            connection.execute("PRAGMA journal_mode = DELETE")


class SQLiteMigrationRunner:
    def __init__(self, database: SQLiteDatabase) -> None:
        self.database = database

    def apply(self) -> tuple[int, ...]:
        self.database.path.parent.mkdir(parents=True, exist_ok=True)
        self.database._require_write_access()
        connection = self.database.connect()
        try:
            connection.execute("BEGIN EXCLUSIVE")
            self.database._require_write_access()
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    applied_at TEXT NOT NULL
                )
                """
            )
            existing = {
                int(row["version"]): str(row["checksum"])
                for row in connection.execute(
                    "SELECT version, checksum FROM schema_migrations ORDER BY version"
                )
            }
            if existing and max(existing) > CURRENT_SCHEMA_VERSION:
                raise PersistenceError(
                    PersistenceErrorCode.INCOMPATIBLE_SCHEMA,
                    "Database schema is newer than this application.",
                )
            applied = []
            for version, name, sql in _migration_resources():
                checksum = hashlib.sha256(sql.encode("utf-8")).hexdigest()
                if version in existing:
                    if existing[version] != checksum:
                        raise PersistenceError(
                            PersistenceErrorCode.INCOMPATIBLE_SCHEMA,
                            f"Migration checksum mismatch for version {version}.",
                        )
                    continue
                for statement in _sql_statements(sql):
                    connection.execute(statement)
                connection.execute(
                    "INSERT INTO schema_migrations(version, name, checksum, applied_at) "
                    "VALUES (?, ?, ?, ?)",
                    (version, name, checksum, datetime.now(UTC).isoformat()),
                )
                applied.append(version)
            _validate_schema_objects(connection)
            connection.commit()
            return tuple(applied)
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def validate(self) -> None:
        expected = {
            version: hashlib.sha256(sql.encode("utf-8")).hexdigest()
            for version, _name, sql in _migration_resources()
        }
        try:
            with self.database.read() as connection:
                actual = {
                    int(row["version"]): str(row["checksum"])
                    for row in connection.execute(
                        "SELECT version, checksum FROM schema_migrations ORDER BY version"
                    )
                }
                _validate_schema_objects(connection)
        except sqlite3.OperationalError as exc:
            raise PersistenceError(
                PersistenceErrorCode.INCOMPATIBLE_SCHEMA,
                "Database schema is missing required migration objects.",
            ) from exc
        if set(actual) != set(expected):
            raise PersistenceError(
                PersistenceErrorCode.INCOMPATIBLE_SCHEMA,
                "Database migration history is incomplete or newer than this application.",
            )
        for version, checksum in expected.items():
            if actual[version] != checksum:
                raise PersistenceError(
                    PersistenceErrorCode.INCOMPATIBLE_SCHEMA,
                    f"Migration checksum mismatch for version {version}.",
                )


def _migration_resources() -> tuple[tuple[int, str, str], ...]:
    root = importlib.resources.files("financial_research_agent.persistence.migrations")
    resources = []
    for resource in root.iterdir():
        if not resource.name.endswith(".sql"):
            continue
        prefix, _separator, _name = resource.name.partition("_")
        resources.append((int(prefix), resource.name, resource.read_text(encoding="utf-8")))
    return tuple(sorted(resources))


def _sql_statements(sql: str) -> tuple[str, ...]:
    statements = []
    buffer = ""
    for line in sql.splitlines(keepends=True):
        buffer += line
        if sqlite3.complete_statement(buffer):
            statements.append(buffer.strip())
            buffer = ""
    if buffer.strip():
        raise ValueError("migration SQL ends with an incomplete statement")
    return tuple(statements)


def _validate_schema_objects(connection: sqlite3.Connection) -> None:
    rows = connection.execute(
        "SELECT type, name FROM sqlite_master WHERE type IN ('table', 'index')"
    )
    objects = {(str(row["type"]), str(row["name"])) for row in rows}
    missing_tables = tuple(name for name in REQUIRED_TABLES if ("table", name) not in objects)
    missing_indexes = tuple(name for name in REQUIRED_INDEXES if ("index", name) not in objects)
    if missing_tables or missing_indexes:
        missing = ", ".join((*missing_tables, *missing_indexes))
        raise PersistenceError(
            PersistenceErrorCode.INCOMPATIBLE_SCHEMA,
            f"Database schema is missing required objects: {missing}.",
        )


def _database_busy() -> PersistenceError:
    return PersistenceError(
        PersistenceErrorCode.DATABASE_BUSY,
        "SQLite maintenance is active; stop active app processes or clear a stale lock and retry.",
    )


def _is_database_busy(error: sqlite3.OperationalError) -> bool:
    message = str(error).lower()
    return "locked" in message or "busy" in message
