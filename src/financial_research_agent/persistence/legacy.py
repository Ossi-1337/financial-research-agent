from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from financial_research_agent.filings import FilingStore
from financial_research_agent.market_data import MarketDataStore
from financial_research_agent.orchestration import OrchestratorRunStore
from financial_research_agent.persistence.contracts import (
    MigrationImportResult,
    PersistenceError,
    PersistenceErrorCode,
)
from financial_research_agent.persistence.database import SQLiteDatabase
from financial_research_agent.persistence.factory import existing_legacy_paths
from financial_research_agent.persistence.operations import backup_legacy_files
from financial_research_agent.reports import CitedResearchRunStore
from financial_research_agent.runtime_settings import RuntimeSettingsStore
from financial_research_agent.settings import Settings
from financial_research_agent.statements import FinancialStatementStore
from financial_research_agent.web.sessions import ChatSessionStore


class LegacyJsonImporter:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.database = SQLiteDatabase.from_data_dir(settings.local_paths.data_dir)

    def migrate(self) -> MigrationImportResult:
        for path in (
            self.settings.local_paths.data_dir,
            self.settings.local_paths.cache_dir,
            self.settings.local_paths.logs_dir,
            self.settings.local_paths.data_dir / "filings",
            self.settings.local_paths.data_dir / "retrieval",
        ):
            path.mkdir(parents=True, exist_ok=True)
        if self.database.path.exists():
            report = self.database.integrity(full=True)
            if not report.healthy:
                raise PersistenceError(
                    PersistenceErrorCode.INTEGRITY_FAILED, "Existing SQLite database is unhealthy."
                )
            return MigrationImportResult(self.database.path, None, report.counts, ())

        legacy_paths = existing_legacy_paths(self.settings.local_paths.data_dir)
        if not legacy_paths:
            self.database.initialize()
            return MigrationImportResult(self.database.path, None, {}, ())

        try:
            sources = self._validated_sources()
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise PersistenceError(
                PersistenceErrorCode.LEGACY_IMPORT_FAILED,
                "Legacy JSON validation failed; active legacy data was not changed.",
            ) from exc
        backup, archived = backup_legacy_files(
            app_home=self.settings.local_paths.app_home,
            data_dir=self.settings.local_paths.data_dir,
            paths=legacy_paths,
        )
        temp_path = self.database.path.with_suffix(f".import-{uuid4().hex}.tmp")
        temp_database = SQLiteDatabase(temp_path)
        try:
            temp_database.initialize()
            bundle = _bundle_for_database(self.settings, temp_database)
            counts = self._import(
                sources,
                bundle,
                temp_database,
                backup_id=backup.id,
                legacy_paths=legacy_paths,
            )
            report = temp_database.integrity(full=True)
            if not report.healthy:
                raise ValueError("imported database failed integrity checks")
            if any(report.counts.get(key, 0) != value for key, value in counts.items()):
                raise ValueError("imported record count mismatch")
            temp_database.prepare_for_atomic_move()
            if self.database.path.exists():
                raise ValueError("SQLite database appeared during legacy import")
            os.replace(temp_path, self.database.path)
            warnings = []
            for source in legacy_paths:
                try:
                    source.unlink()
                except OSError:
                    warnings.append(f"Could not remove archived legacy path: {source.name}")
            return MigrationImportResult(
                self.database.path, backup, counts, archived, tuple(warnings)
            )
        except Exception as exc:
            temp_path.unlink(missing_ok=True)
            Path(f"{temp_path}-wal").unlink(missing_ok=True)
            Path(f"{temp_path}-shm").unlink(missing_ok=True)
            if isinstance(exc, PersistenceError):
                raise
            raise PersistenceError(
                PersistenceErrorCode.LEGACY_IMPORT_FAILED,
                "Legacy JSON import failed; active legacy data was not changed.",
            ) from exc

    def _validated_sources(self) -> dict[str, object]:
        return {
            "sessions": ChatSessionStore.from_settings(self.settings),
            "market": MarketDataStore.from_settings(self.settings),
            "statements": FinancialStatementStore.from_settings(self.settings),
            "filings": FilingStore.from_settings(self.settings),
            "cited": CitedResearchRunStore.from_settings(self.settings),
            "orchestrator": OrchestratorRunStore.from_settings(self.settings),
            "settings": RuntimeSettingsStore.from_settings(self.settings),
            "background": _legacy_background_jobs(self.settings.local_paths.data_dir),
        }

    def _import(
        self,
        sources,
        bundle,
        database: SQLiteDatabase,
        *,
        backup_id: str,
        legacy_paths: tuple[Path, ...],
    ) -> dict[str, int]:
        counts = _source_counts(sources)
        counts["import_records"] = 1
        with database.transaction(exclusive=True):
            for session in sources["sessions"].list(limit=1_000_000):
                bundle.sessions.save(session)
            for result in sources["market"].list():
                bundle.market_data.save_history(result)
            for result in sources["statements"].list():
                bundle.financial_statements.save_result(result)
            for result in sources["filings"].list():
                bundle.filings.save_result(result)
            for run in sources["cited"].list():
                bundle.cited_runs.save(run)
            for run in sources["orchestrator"].list():
                bundle.orchestrator_runs.save(run)
            overrides = sources["settings"].get()
            if overrides.to_dict():
                bundle.runtime_settings.replace(overrides, base_settings=self.settings)
            for job in sources["background"]:
                bundle.background_jobs.save(job)
            with database.transaction() as connection:
                connection.execute(
                    """
                    INSERT INTO import_records(
                        id, imported_at, backup_id, source_manifest_json, counts_json
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        f"import_{uuid4().hex}",
                        datetime.now(UTC).isoformat(),
                        backup_id,
                        json.dumps(
                            [
                                path.relative_to(self.settings.local_paths.data_dir).as_posix()
                                for path in legacy_paths
                            ],
                            sort_keys=True,
                        ),
                        json.dumps(counts, sort_keys=True),
                    ),
                )
        return counts


def _bundle_for_database(settings: Settings, database: SQLiteDatabase):
    from datetime import timedelta

    from financial_research_agent.persistence.factory import PersistenceBundle
    from financial_research_agent.persistence.repositories import (
        SQLiteBackgroundJobStore,
        SQLiteChatSessionStore,
        SQLiteCitedResearchRunStore,
        SQLiteFilingStore,
        SQLiteFinancialStatementStore,
        SQLiteMarketDataStore,
        SQLiteOrchestratorRunStore,
        SQLiteRuntimeSettingsStore,
    )

    return PersistenceBundle(
        database=database,
        sessions=SQLiteChatSessionStore(
            database,
            recent_turns=settings.chat.history_recent_turns,
            summary_max_chars=settings.chat.history_summary_max_chars,
        ),
        market_data=SQLiteMarketDataStore(
            database, stale_after=timedelta(days=settings.data_sources.market_data_cache_ttl_days)
        ),
        financial_statements=SQLiteFinancialStatementStore(
            database,
            stale_after=timedelta(days=settings.data_sources.financial_statement_cache_ttl_days),
        ),
        filings=SQLiteFilingStore(
            database, stale_after=timedelta(days=settings.data_sources.filing_cache_ttl_days)
        ),
        cited_runs=SQLiteCitedResearchRunStore(database),
        orchestrator_runs=SQLiteOrchestratorRunStore(database),
        runtime_settings=SQLiteRuntimeSettingsStore(database),
        background_jobs=SQLiteBackgroundJobStore(database),
    )


def _legacy_background_jobs(data_dir: Path):
    from financial_research_agent.background import BackgroundResearchJob

    path = data_dir / "background_jobs.json"
    if not path.exists():
        return ()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != 1 or not isinstance(payload.get("jobs"), list):
        raise ValueError("unsupported background job JSON store")
    return tuple(BackgroundResearchJob.from_dict(item) for item in payload["jobs"])


def _source_counts(sources) -> dict[str, int]:
    sessions = sources["sessions"].list(limit=1_000_000)
    market = sources["market"].list()
    statements = sources["statements"].list()
    filings = sources["filings"].list()
    cited = sources["cited"].list()
    orchestrator = sources["orchestrator"].list()
    return {
        "chat_sessions": len(sessions),
        "chat_messages": sum(len(item.messages) for item in sessions),
        "market_series": len(market),
        "price_bars": sum(len(item.bars) for item in market),
        "statement_results": len(statements),
        "financial_statements": sum(len(item.statements) for item in statements),
        "filing_results": len(filings),
        "filings": sum(len(item.filings) for item in filings),
        "filing_chunks": sum(len(item.chunks) for item in filings),
        "cited_runs": len(cited),
        "citations": sum(len(item.citations) for item in cited),
        "evidence_snippets": sum(len(item.evidence) for item in cited),
        "orchestrator_runs": len(orchestrator),
        "agent_handoffs": sum(len(item.handoffs) for item in orchestrator),
        "background_jobs": len(sources["background"]),
        "runtime_settings": 1 if sources["settings"].get().to_dict() else 0,
    }
