from __future__ import annotations

import json
import sqlite3
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from threading import Event

import pytest

from financial_research_agent.background import BackgroundResearchJob, BackgroundResearchStatus
from financial_research_agent.entities import (
    EntityIdentifier,
    EntityIdentifierType,
    ResolvedCompany,
    ResolvedSecurity,
)
from financial_research_agent.filings import (
    FilingChunk,
    FilingCompany,
    FilingDocument,
    FilingDocumentFormat,
    FilingIngestionResult,
    FilingSource,
)
from financial_research_agent.market_data import (
    HistoricalPriceBar,
    HistoricalPriceResult,
    MarketDataSource,
    MarketSecurity,
    calculate_price_metrics,
)
from financial_research_agent.orchestration import (
    OrchestratedResearchRun,
    OrchestratorExecutionPolicy,
    OrchestratorRunStatus,
    default_orchestrator_plan,
)
from financial_research_agent.persistence import (
    CURRENT_SCHEMA_VERSION,
    LegacyJsonImporter,
    PersistenceError,
    PersistenceErrorCode,
    SQLiteDatabase,
    SQLiteOperations,
    create_persistence,
)
from financial_research_agent.reports import CitedResearchRun, CitedResearchRunStatus
from financial_research_agent.runtime_settings import RuntimeSettingsOverrides
from financial_research_agent.settings import Settings
from financial_research_agent.web.sessions import ChatSessionStore


def test_clean_sqlite_setup_applies_pragmas_and_schema(tmp_path: Path) -> None:
    bundle = create_persistence(_settings(tmp_path))

    assert bundle.database is not None
    assert bundle.database.schema_version() == CURRENT_SCHEMA_VERSION
    with bundle.database.read() as connection:
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert connection.execute("PRAGMA busy_timeout").fetchone()[0] == 5_000

    report = bundle.database.integrity(full=True)
    assert report.healthy is True
    assert report.counts["chat_sessions"] == 0


def test_concurrent_database_initialization_tolerates_wal_race(tmp_path: Path) -> None:
    path = tmp_path / "data" / "financial_research_agent.sqlite3"

    with ThreadPoolExecutor(max_workers=5) as executor:
        tuple(executor.map(lambda _index: SQLiteDatabase(path).initialize(), range(5)))

    database = SQLiteDatabase(path)
    assert database.schema_version() == CURRENT_SCHEMA_VERSION
    assert database.integrity().healthy is True


def test_chat_repository_round_trip_ordering_and_concurrent_access(tmp_path: Path) -> None:
    store = create_persistence(_settings(tmp_path)).sessions

    with ThreadPoolExecutor(max_workers=4) as executor:
        sessions = tuple(executor.map(lambda _index: store.create(), range(12)))

    updated = store.append_exchange(
        session_id=sessions[0].id,
        user_content="Question",
        assistant_content="Answer",
        provider="offline-test",
        model="offline-test",
    )

    assert store.count() == 12
    assert store.list()[0].id == updated.id
    assert [message.content for message in store.get(updated.id).messages] == [
        "Question",
        "Answer",
    ]
    assert store.delete(updated.id) is True
    assert store.get(updated.id) is None


def test_chat_repository_serializes_concurrent_appends(tmp_path: Path) -> None:
    store = create_persistence(_settings(tmp_path)).sessions
    session = store.create()

    def append(index: int) -> None:
        store.append_exchange(
            session_id=session.id,
            user_content=f"Question {index}",
            assistant_content=f"Answer {index}",
            provider="offline-test",
            model="offline-test",
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        tuple(executor.map(append, range(20)))

    loaded = store.get(session.id)
    assert loaded is not None
    assert len(loaded.messages) == 40
    assert {message.content for message in loaded.messages} == {
        *(f"Question {index}" for index in range(20)),
        *(f"Answer {index}" for index in range(20)),
    }


def test_foreign_keys_and_transaction_rollback_are_enforced(tmp_path: Path) -> None:
    database = create_persistence(_settings(tmp_path)).database
    assert database is not None

    with pytest.raises(sqlite3.IntegrityError), database.transaction() as connection:
        connection.execute(
            """
            INSERT INTO chat_messages(id, session_id, sequence, role, created_at, payload_json)
            VALUES ('message_bad', 'missing', 0, 'user', ?, '{}')
            """,
            (datetime.now(UTC).isoformat(),),
        )

    assert database.integrity().counts["chat_messages"] == 0


def test_changed_migration_checksum_and_busy_database_fail_safely(tmp_path: Path) -> None:
    database = create_persistence(_settings(tmp_path)).database
    assert database is not None
    with database.transaction() as connection:
        connection.execute("UPDATE schema_migrations SET checksum = 'changed' WHERE version = 1")

    with pytest.raises(PersistenceError) as checksum_error:
        database.initialize()
    assert checksum_error.value.code == PersistenceErrorCode.INCOMPATIBLE_SCHEMA

    other_home = tmp_path / "busy"
    busy_database = create_persistence(_settings(other_home)).database
    assert busy_database is not None
    lock = busy_database.connect()
    lock.execute("BEGIN EXCLUSIVE")
    try:
        contender = SQLiteDatabase(busy_database.path, busy_timeout_ms=10)
        with pytest.raises(PersistenceError) as busy_error, contender.transaction():
            pass
        assert busy_error.value.code == PersistenceErrorCode.DATABASE_BUSY
    finally:
        lock.rollback()
        lock.close()


@pytest.mark.parametrize(
    ("object_type", "object_name"),
    (("TABLE", "chat_messages"), ("INDEX", "chat_sessions_updated_at_idx")),
)
def test_schema_validation_rejects_missing_required_objects(
    tmp_path: Path, object_type: str, object_name: str
) -> None:
    database = create_persistence(_settings(tmp_path)).database
    assert database is not None
    with database.transaction() as connection:
        connection.execute(f'DROP {object_type} "{object_name}"')

    with pytest.raises(PersistenceError) as error:
        database.initialize()

    assert error.value.code == PersistenceErrorCode.INCOMPATIBLE_SCHEMA
    assert object_name in str(error.value)


def test_market_repository_preserves_decimal_and_date_values(tmp_path: Path) -> None:
    store = create_persistence(_settings(tmp_path)).market_data
    result = _market_result()

    store.save_history(result)
    loaded = store.get_history(symbol="nvo", now=datetime(2026, 7, 2, tzinfo=UTC))

    assert loaded is not None
    assert loaded.bars[0].close == Decimal("101.2300")
    assert loaded.bars[0].priced_at == date(2026, 7, 1)
    assert loaded.source.retrieved_at == datetime(2026, 7, 1, 12, tzinfo=UTC)
    assert store.clear() == 1


def test_filing_repository_stores_offsets_and_rehydrates_chunk_text(tmp_path: Path) -> None:
    bundle = create_persistence(_settings(tmp_path))
    result = _filing_result(tmp_path)

    bundle.filings.save_result(result)
    with bundle.database.read() as connection:
        payload = json.loads(
            connection.execute("SELECT payload_json FROM filing_results").fetchone()[0]
        )
    loaded = bundle.filings.get_result(cik="320193", now=datetime(2026, 7, 2, tzinfo=UTC))

    assert "text" not in payload["chunks"][0]
    assert loaded is not None
    assert loaded.chunks[0].text == "Revenue increased."
    assert loaded.chunks[0].char_start == 0
    assert loaded.chunks[0].char_end == 18


def test_complex_repository_payloads_round_trip(tmp_path: Path) -> None:
    bundle = create_persistence(_settings(tmp_path))
    now = datetime(2026, 7, 1, tzinfo=UTC)
    cited = CitedResearchRun(
        id="cited_1",
        query="Question",
        answer="Limited answer",
        status=CitedResearchRunStatus.LIMITED,
        created_at=now,
        limitations=("No evidence",),
    )
    run = OrchestratedResearchRun(
        id="run_1",
        query="Research NVO",
        status=OrchestratorRunStatus.PARTIAL,
        created_at=now,
        updated_at=now,
        execution_policy=OrchestratorExecutionPolicy.SEQUENTIAL_LOCAL_SAFE,
        plan=default_orchestrator_plan(),
        selected_company={"id": "company_1", "legal_name": "Test Company", "cik": "320193"},
        selected_security={"id": "security_1", "symbol": "NVO", "currency": "USD"},
        limitations=("Fixture",),
    )

    bundle.cited_runs.save(cited)
    bundle.orchestrator_runs.save(run)
    bundle.runtime_settings.replace(
        RuntimeSettingsOverrides(chat_model="test-model"), base_settings=_settings(tmp_path)
    )

    assert bundle.cited_runs.get("cited_1") == cited
    assert bundle.orchestrator_runs.get("run_1") == run
    assert bundle.runtime_settings.get().chat_model == "test-model"


def test_existing_legacy_json_requires_explicit_migration(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    legacy = ChatSessionStore(storage_path=settings.local_paths.data_dir / "chat_sessions.json")
    session = legacy.create()

    with pytest.raises(PersistenceError) as error:
        create_persistence(settings)

    assert error.value.code == PersistenceErrorCode.STORAGE_MIGRATION_REQUIRED
    result = LegacyJsonImporter(settings).migrate()
    migrated = create_persistence(settings)
    assert migrated.sessions.get(session.id) is not None
    assert result.backup is not None
    assert not legacy.storage_path.exists()
    assert all(path.exists() for path in result.archived_paths)
    assert LegacyJsonImporter(settings).migrate().backup is None


def test_corrupt_legacy_json_never_activates_partial_database(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    legacy_path = settings.local_paths.data_dir / "chat_sessions.json"
    legacy_path.parent.mkdir(parents=True)
    legacy_path.write_text("{broken", encoding="utf-8")

    with pytest.raises(PersistenceError) as error:
        LegacyJsonImporter(settings).migrate()

    assert error.value.code == PersistenceErrorCode.LEGACY_IMPORT_FAILED
    assert legacy_path.exists()
    assert not SQLiteDatabase.from_data_dir(settings.local_paths.data_dir).path.exists()


def test_backup_restore_and_backup_id_validation(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    bundle = create_persistence(settings)
    first = bundle.sessions.create()
    operations = SQLiteOperations(bundle.database, app_home=settings.local_paths.app_home)
    backup = operations.backup()
    second = bundle.sessions.create()

    restored = operations.restore(backup.id)
    reloaded = create_persistence(settings)

    assert restored["restored_backup"]["id"] == backup.id
    assert reloaded.sessions.get(first.id) is not None
    assert reloaded.sessions.get(second.id) is None
    with pytest.raises(PersistenceError) as error:
        operations.restore("../outside")
    assert error.value.code == PersistenceErrorCode.INVALID_BACKUP_ID


def test_restore_blocks_concurrent_writes_until_atomic_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    bundle = create_persistence(settings)
    bundle.sessions.create()
    operations = SQLiteOperations(bundle.database, app_home=settings.local_paths.app_home)
    backup = operations.backup()
    copy_started = Event()
    allow_copy = Event()
    from financial_research_agent.persistence import operations as operations_module

    original_copy = operations_module.shutil.copy2

    def blocked_copy(source: Path, target: Path) -> str:
        copy_started.set()
        assert allow_copy.wait(timeout=5)
        return original_copy(source, target)

    monkeypatch.setattr(operations_module.shutil, "copy2", blocked_copy)
    future: Future[dict[str, object]]
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(operations.restore, backup.id)
        assert copy_started.wait(timeout=5)
        contender = SQLiteDatabase(bundle.database.path, busy_timeout_ms=50)
        with pytest.raises(PersistenceError) as error, contender.transaction():
            pass
        assert error.value.code == PersistenceErrorCode.DATABASE_BUSY
        allow_copy.set()
        future.result(timeout=5)

    assert not bundle.database.maintenance_lock_path.exists()


def test_orchestrator_store_normalizes_resolved_entity_payloads(tmp_path: Path) -> None:
    bundle = create_persistence(_settings(tmp_path))
    company = ResolvedCompany(
        id="company_novo",
        legal_name="Novo Nordisk A/S",
        identifiers=(
            EntityIdentifier(EntityIdentifierType.CIK, "0000353278", source="fixture"),
            EntityIdentifier(EntityIdentifierType.LEI, "549300DAQ1CVT6CXN342", source="fixture"),
        ),
        country_code="DK",
    )
    security = ResolvedSecurity(
        id="security_nvo",
        company_id=company.id,
        ticker="NVO",
        name="Novo Nordisk A/S ADR",
        exchange_mic="XNYS",
        currency="USD",
        isin="US6701002056",
        identifiers=(
            EntityIdentifier(EntityIdentifierType.TICKER, "NVO", source="fixture"),
            EntityIdentifier(EntityIdentifierType.FIGI, "BBG000LYF3S8", source="fixture"),
        ),
    )
    run = _orchestrator_run(
        run_id="run_resolved",
        company=company.to_dict(),
        security=security.to_dict(),
    )

    bundle.orchestrator_runs.save(run)

    with bundle.database.read() as connection:
        stored_run = connection.execute(
            "SELECT company_id, security_id FROM orchestrator_runs WHERE id = ?", (run.id,)
        ).fetchone()
        stored_company = connection.execute(
            "SELECT legal_name, cik FROM companies WHERE id = ?", (company.id,)
        ).fetchone()
        stored_security = connection.execute(
            "SELECT symbol, company_id FROM securities WHERE id = ?", (security.id,)
        ).fetchone()
        company_identifiers = {
            tuple(row)
            for row in connection.execute(
                "SELECT scheme, value FROM company_identifiers WHERE company_id = ?",
                (company.id,),
            )
        }
        security_identifiers = {
            tuple(row)
            for row in connection.execute(
                "SELECT scheme, value FROM security_identifiers WHERE security_id = ?",
                (security.id,),
            )
        }

    assert tuple(stored_run) == (company.id, security.id)
    assert tuple(stored_company) == (company.legal_name, "0000353278")
    assert tuple(stored_security) == (security.ticker, company.id)
    assert company_identifiers == {
        ("cik", "0000353278"),
        ("lei", "549300DAQ1CVT6CXN342"),
    }
    assert security_identifiers == {
        ("ticker:XNYS", "NVO"),
        ("isin", "US6701002056"),
        ("figi", "BBG000LYF3S8"),
    }


def test_entity_identifier_conflict_rolls_back_orchestrator_save(tmp_path: Path) -> None:
    bundle = create_persistence(_settings(tmp_path))
    first = _orchestrator_run(
        run_id="run_first",
        company={"id": "company_first", "legal_name": "First", "cik": "0000000001"},
        security={"id": "security_first", "symbol": "ONE"},
    )
    conflicting = _orchestrator_run(
        run_id="run_conflict",
        company={"id": "company_other", "legal_name": "Other", "cik": "0000000001"},
        security={"id": "security_other", "symbol": "TWO"},
    )
    bundle.orchestrator_runs.save(first)

    with pytest.raises(ValueError, match="CIK is already assigned"):
        bundle.orchestrator_runs.save(conflicting)

    with bundle.database.read() as connection:
        assert (
            connection.execute("SELECT 1 FROM companies WHERE id = 'company_other'").fetchone()
            is None
        )
        assert (
            connection.execute(
                "SELECT 1 FROM orchestrator_runs WHERE id = 'run_conflict'"
            ).fetchone()
            is None
        )


def test_cleanup_is_dry_run_and_reset_preserves_schema(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    bundle = create_persistence(settings)
    bundle.sessions.create()
    operations = SQLiteOperations(bundle.database, app_home=settings.local_paths.app_home)

    preview = operations.cleanup(dataset="chat-sessions", older_than_days=0, confirmed=False)
    applied = operations.cleanup(dataset="chat-sessions", older_than_days=0, confirmed=True)

    assert preview.dry_run is True
    assert preview.matched_records == 1
    assert bundle.sessions.count() == 0
    assert applied.deleted_records == 1
    operations.reset_data()
    assert bundle.database.schema_version() == CURRENT_SCHEMA_VERSION


def test_filing_cleanup_requires_explicit_source_document_permission(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    bundle = create_persistence(settings)
    result = _filing_result(tmp_path)
    bundle.filings.save_result(result)
    operations = SQLiteOperations(bundle.database, app_home=settings.local_paths.app_home)

    with pytest.raises(ValueError, match="include-source-documents"):
        operations.cleanup(dataset="filing-documents", older_than_days=0, confirmed=True)

    cleanup = operations.cleanup(
        dataset="filing-documents",
        older_than_days=0,
        confirmed=True,
        include_source_documents=True,
    )
    assert cleanup.deleted_records == 1
    assert not Path(result.filings[0].local_raw_path).exists()
    assert not Path(result.filings[0].local_text_path).exists()


def test_unfinished_background_jobs_fail_after_restart(tmp_path: Path) -> None:
    bundle = create_persistence(_settings(tmp_path))
    now = datetime(2026, 7, 1, tzinfo=UTC)
    job = BackgroundResearchJob(
        id="job_1",
        query="Research",
        status=BackgroundResearchStatus.RUNNING,
        created_at=now,
        updated_at=now,
        orchestrator_run_id="run_1",
    )
    bundle.background_jobs.save(job)

    assert bundle.background_jobs.fail_unfinished(now=now) == 1
    recovered = bundle.background_jobs.get(job.id)
    assert recovered.status == BackgroundResearchStatus.FAILED
    assert recovered.error_code == "process_restarted"


def _settings(tmp_path: Path) -> Settings:
    return Settings.from_env({"FRA_HOME": str(tmp_path), "FRA_STORAGE_PROVIDER": "sqlite"})


def _orchestrator_run(
    *, run_id: str, company: dict[str, object], security: dict[str, object]
) -> OrchestratedResearchRun:
    now = datetime(2026, 7, 1, tzinfo=UTC)
    return OrchestratedResearchRun(
        id=run_id,
        query="Fixture research",
        status=OrchestratorRunStatus.PARTIAL,
        created_at=now,
        updated_at=now,
        execution_policy=OrchestratorExecutionPolicy.SEQUENTIAL_LOCAL_SAFE,
        plan=default_orchestrator_plan(),
        selected_company=company,
        selected_security=security,
        limitations=("Fixture",),
    )


def _market_result() -> HistoricalPriceResult:
    security = MarketSecurity(symbol="NVO", security_id="security_nvo", currency="USD")
    bar = HistoricalPriceBar(
        security=security,
        priced_at=date(2026, 7, 1),
        open=Decimal("100.10"),
        high=Decimal("102.20"),
        low=Decimal("99.90"),
        close=Decimal("101.2300"),
        volume=1234,
    )
    source = MarketDataSource(
        provider="fixture",
        provider_status="test fixture",
        source_url="https://example.test/prices",
        retrieved_at=datetime(2026, 7, 1, 12, tzinfo=UTC),
        attribution="Test fixture",
        data_as_of=date(2026, 7, 1),
    )
    return HistoricalPriceResult(
        security=security,
        bars=(bar,),
        source=source,
        metrics=calculate_price_metrics((bar,)),
    )


def _filing_result(tmp_path: Path) -> FilingIngestionResult:
    text = "Revenue increased. More text."
    text_path = tmp_path / "data" / "filings" / "text" / "filing.txt"
    raw_path = tmp_path / "data" / "filings" / "raw" / "filing.html"
    text_path.parent.mkdir(parents=True)
    raw_path.parent.mkdir(parents=True)
    text_path.write_text(text, encoding="utf-8")
    raw_path.write_text(f"<p>{text}</p>", encoding="utf-8")
    company = FilingCompany(cik="320193", company_id="company_apple", legal_name="Apple Inc.")
    source = FilingSource(
        provider="sec-edgar",
        provider_status="official",
        source_url="https://www.sec.gov/example",
        retrieved_at=datetime(2026, 7, 1, tzinfo=UTC),
        attribution="SEC",
    )
    filing = FilingDocument(
        id="filing_1",
        company=company,
        form_type="10-K",
        accession_number="0000320193-26-000001",
        filing_date=date(2026, 6, 30),
        report_date=date(2026, 6, 30),
        document_url="https://www.sec.gov/document",
        source_url=source.source_url,
        document_format=FilingDocumentFormat.HTML,
        retrieved_at=source.retrieved_at,
        local_raw_path=str(raw_path),
        local_text_path=str(text_path),
        source=source,
        chunk_ids=("chunk_1",),
    )
    chunk = FilingChunk(
        id="chunk_1",
        filing_id=filing.id,
        chunk_index=0,
        text=text[:18],
        char_start=0,
        char_end=18,
        source_url=source.source_url,
        accession_number=filing.accession_number,
        form_type=filing.form_type,
    )
    return FilingIngestionResult(company=company, filings=(filing,), chunks=(chunk,), source=source)
