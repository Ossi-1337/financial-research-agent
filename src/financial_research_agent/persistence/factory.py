from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

from financial_research_agent.filings import FilingStore
from financial_research_agent.market_data import MarketDataStore
from financial_research_agent.orchestration import OrchestratorRunStore
from financial_research_agent.persistence.contracts import PersistenceError, PersistenceErrorCode
from financial_research_agent.persistence.database import SQLiteDatabase
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
from financial_research_agent.reports import CitedResearchRunStore
from financial_research_agent.runtime_settings import RuntimeSettingsStore
from financial_research_agent.settings import Settings
from financial_research_agent.statements import FinancialStatementStore
from financial_research_agent.web.sessions import ChatSessionStore

LEGACY_STRUCTURED_PATHS = (
    "chat_sessions.json",
    "market_data_price_bars.json",
    "financial_statements.json",
    "filings/filings_index.json",
    "report_runs.json",
    "orchestrator_runs.json",
    "settings_overrides.json",
    "background_jobs.json",
    "storage_migrations.json",
)


@dataclass(frozen=True, slots=True)
class PersistenceBundle:
    database: SQLiteDatabase | None
    sessions: object
    market_data: object
    financial_statements: object
    filings: object
    cited_runs: object
    orchestrator_runs: object
    runtime_settings: object
    background_jobs: object | None


def create_persistence(settings: Settings, *, allow_legacy: bool = False) -> PersistenceBundle:
    _ensure_local_directories(settings)
    if settings.storage.provider == "local-json":
        return PersistenceBundle(
            database=None,
            sessions=ChatSessionStore.from_settings(settings),
            market_data=MarketDataStore.from_settings(settings),
            financial_statements=FinancialStatementStore.from_settings(settings),
            filings=FilingStore.from_settings(settings),
            cited_runs=CitedResearchRunStore.from_settings(settings),
            orchestrator_runs=OrchestratorRunStore.from_settings(settings),
            runtime_settings=RuntimeSettingsStore.from_settings(settings),
            background_jobs=None,
        )
    if settings.storage.provider != "sqlite":
        raise ValueError(f"Unsupported storage provider: {settings.storage.provider}")

    database = SQLiteDatabase.from_data_dir(settings.local_paths.data_dir)
    if not database.path.exists() and not allow_legacy:
        legacy = existing_legacy_paths(settings.local_paths.data_dir)
        if legacy:
            raise PersistenceError(
                PersistenceErrorCode.STORAGE_MIGRATION_REQUIRED,
                "Legacy JSON storage exists. Run storage-migrate before starting the app.",
            )
    database.initialize()
    return PersistenceBundle(
        database=database,
        sessions=SQLiteChatSessionStore(
            database,
            recent_turns=settings.chat.history_recent_turns,
            summary_max_chars=settings.chat.history_summary_max_chars,
        ),
        market_data=SQLiteMarketDataStore(
            database,
            stale_after=timedelta(days=settings.data_sources.market_data_cache_ttl_days),
        ),
        financial_statements=SQLiteFinancialStatementStore(
            database,
            stale_after=timedelta(days=settings.data_sources.financial_statement_cache_ttl_days),
        ),
        filings=SQLiteFilingStore(
            database,
            stale_after=timedelta(days=settings.data_sources.filing_cache_ttl_days),
        ),
        cited_runs=SQLiteCitedResearchRunStore(database),
        orchestrator_runs=SQLiteOrchestratorRunStore(database),
        runtime_settings=SQLiteRuntimeSettingsStore(database),
        background_jobs=SQLiteBackgroundJobStore(database),
    )


def existing_legacy_paths(data_dir: Path) -> tuple[Path, ...]:
    return tuple(
        path for relative in LEGACY_STRUCTURED_PATHS if (path := data_dir / relative).exists()
    )


def _ensure_local_directories(settings: Settings) -> None:
    for path in (
        settings.local_paths.app_home,
        settings.local_paths.data_dir,
        settings.local_paths.cache_dir,
        settings.local_paths.logs_dir,
        settings.local_paths.data_dir / "filings",
        settings.local_paths.data_dir / "retrieval",
    ):
        path.mkdir(parents=True, exist_ok=True)
