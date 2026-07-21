"""SQLite persistence, migration, backup, and repository services."""

from financial_research_agent.persistence.contracts import (
    AggregateRepository,
    BackupRecord,
    ChatSessionRepository,
    IntegrityReport,
    MigrationImportResult,
    PersistenceError,
    PersistenceErrorCode,
    Repository,
)
from financial_research_agent.persistence.database import (
    CURRENT_SCHEMA_VERSION,
    DATABASE_FILENAME,
    SQLiteDatabase,
    SQLiteMigrationRunner,
)
from financial_research_agent.persistence.factory import (
    LEGACY_STRUCTURED_PATHS,
    PersistenceBundle,
    create_persistence,
    existing_legacy_paths,
)
from financial_research_agent.persistence.legacy import LegacyJsonImporter
from financial_research_agent.persistence.operations import CleanupResult, SQLiteOperations
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
from financial_research_agent.persistence.storage_manager import (
    SQLiteStorageManager,
    SQLiteStorageStatus,
    create_storage_manager,
)

__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "DATABASE_FILENAME",
    "LEGACY_STRUCTURED_PATHS",
    "AggregateRepository",
    "BackupRecord",
    "ChatSessionRepository",
    "CleanupResult",
    "IntegrityReport",
    "LegacyJsonImporter",
    "MigrationImportResult",
    "PersistenceBundle",
    "PersistenceError",
    "PersistenceErrorCode",
    "Repository",
    "SQLiteBackgroundJobStore",
    "SQLiteChatSessionStore",
    "SQLiteCitedResearchRunStore",
    "SQLiteDatabase",
    "SQLiteFilingStore",
    "SQLiteFinancialStatementStore",
    "SQLiteMarketDataStore",
    "SQLiteMigrationRunner",
    "SQLiteOperations",
    "SQLiteOrchestratorRunStore",
    "SQLiteRuntimeSettingsStore",
    "SQLiteStorageManager",
    "SQLiteStorageStatus",
    "create_persistence",
    "create_storage_manager",
    "existing_legacy_paths",
]
