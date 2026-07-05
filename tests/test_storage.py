from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from pathlib import Path

import pytest

from financial_research_agent.settings import Settings
from financial_research_agent.storage import (
    LocalStorageManager,
    StorageArea,
    StorageDataset,
    StorageDatasetSpec,
    StorageFormat,
    default_storage_dataset_specs,
)


def test_storage_specs_cover_current_local_stores_and_have_no_secrets(tmp_path: Path) -> None:
    settings = Settings.from_env({"FRA_HOME": str(tmp_path)})
    specs = default_storage_dataset_specs(settings)

    assert {spec.dataset for spec in specs} == {
        StorageDataset.CHAT_SESSIONS,
        StorageDataset.COMPANY_LOOKUP_CACHE,
        StorageDataset.MARKET_DATA_PRICE_BARS,
        StorageDataset.FINANCIAL_STATEMENTS,
        StorageDataset.FILINGS_INDEX,
        StorageDataset.FILING_RAW_DOCUMENTS,
        StorageDataset.FILING_EXTRACTED_TEXT,
        StorageDataset.RETRIEVAL_VECTOR_INDEX,
        StorageDataset.REPORT_RUNS,
        StorageDataset.STORAGE_MIGRATIONS,
    }
    assert all(not spec.contains_secrets for spec in specs)
    assert _spec(specs, StorageDataset.COMPANY_LOOKUP_CACHE).reset_on_data_reset is True
    with pytest.raises(FrozenInstanceError):
        specs[0].label = "mutated"


def test_storage_spec_rejects_paths_outside_area() -> None:
    with pytest.raises(ValueError, match="relative_path"):
        StorageDatasetSpec(
            dataset=StorageDataset.CHAT_SESSIONS,
            label="Bad path",
            area=StorageArea.DATA,
            relative_path="../chat_sessions.json",
            storage_format=StorageFormat.JSON,
            description="invalid",
        )


def test_storage_migrate_creates_layout_and_is_idempotent(tmp_path: Path) -> None:
    settings = Settings.from_env({"FRA_HOME": str(tmp_path)})
    manager = LocalStorageManager.from_settings(settings)

    result = manager.migrate(now=datetime(2026, 7, 4, tzinfo=UTC))
    second = manager.migrate(now=datetime(2026, 7, 5, tzinfo=UTC))

    assert settings.local_paths.data_dir.exists()
    assert settings.local_paths.cache_dir.exists()
    assert settings.local_paths.logs_dir.exists()
    assert settings.local_paths.data_dir.joinpath("filings").exists()
    assert settings.local_paths.data_dir.joinpath("retrieval").exists()
    assert result.applied_migrations[0].id == "0001_local_json_storage_layout"
    assert second.applied_migrations == ()
    assert second.skipped_migrations == ("0001_local_json_storage_layout",)
    payload = json.loads(manager.migrations_path.read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert len(payload["migrations"]) == 1


def test_migration_preserves_existing_local_data(tmp_path: Path) -> None:
    settings = Settings.from_env({"FRA_HOME": str(tmp_path)})
    chat_path = settings.local_paths.data_dir / "chat_sessions.json"
    chat_path.parent.mkdir(parents=True)
    chat_path.write_text(
        json.dumps({"version": 1, "sessions": [{"id": "session_existing"}]}),
        encoding="utf-8",
    )

    LocalStorageManager.from_settings(settings).migrate()

    payload = json.loads(chat_path.read_text(encoding="utf-8"))
    assert payload["sessions"][0]["id"] == "session_existing"


def test_storage_manifest_inspects_records_and_staleness(tmp_path: Path) -> None:
    settings = Settings.from_env(
        {
            "FRA_HOME": str(tmp_path),
            "FRA_COMPANY_LOOKUP_CACHE_TTL_DAYS": "2",
        }
    )
    cache_path = settings.local_paths.cache_dir / "sec_company_tickers.json"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text(
        json.dumps(
            {
                "version": 1,
                "retrieved_at": "2026-07-01T00:00:00+00:00",
                "records": [{"cik": 320193, "ticker": "AAPL", "title": "APPLE INC."}],
            }
        ),
        encoding="utf-8",
    )

    manifest = LocalStorageManager.from_settings(settings).inspect(
        now=datetime(2026, 7, 4, tzinfo=UTC)
    )
    cache_entry = _entry(manifest, StorageDataset.COMPANY_LOOKUP_CACHE)

    assert cache_entry.exists is True
    assert cache_entry.schema_version == 1
    assert cache_entry.record_count == 1
    assert cache_entry.stale is True
    assert cache_entry.expires_at == datetime(2026, 7, 3, tzinfo=UTC)


def test_clear_cache_removes_cache_files_without_data(tmp_path: Path) -> None:
    settings = Settings.from_env({"FRA_HOME": str(tmp_path)})
    cache_path = settings.local_paths.cache_dir / "sec_company_tickers.json"
    data_path = settings.local_paths.data_dir / "market_data_price_bars.json"
    cache_path.parent.mkdir(parents=True)
    data_path.parent.mkdir(parents=True)
    cache_path.write_text("{}", encoding="utf-8")
    data_path.write_text("{}", encoding="utf-8")

    result = LocalStorageManager.from_settings(settings).clear_cache()

    assert result.operation == "clear_cache"
    assert result.deleted_count == 1
    assert not cache_path.exists()
    assert data_path.exists()


def test_reset_local_data_removes_data_and_cache_but_preserves_logs(tmp_path: Path) -> None:
    settings = Settings.from_env({"FRA_HOME": str(tmp_path)})
    chat_path = settings.local_paths.data_dir / "chat_sessions.json"
    raw_path = settings.local_paths.data_dir / "filings" / "raw" / "file.htm"
    cache_path = settings.local_paths.cache_dir / "sec_company_tickers.json"
    log_path = settings.local_paths.logs_dir / "app.log"
    for path in (chat_path, raw_path, cache_path, log_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("content", encoding="utf-8")

    result = LocalStorageManager.from_settings(settings).reset_local_data()

    assert result.operation == "reset_local_data"
    assert not chat_path.exists()
    assert not raw_path.exists()
    assert not cache_path.exists()
    assert log_path.exists()


def _entry(manifest, dataset: StorageDataset):
    matches = [entry for entry in manifest.datasets if entry.spec.dataset == dataset]
    assert len(matches) == 1
    return matches[0]


def _spec(specs: tuple[StorageDatasetSpec, ...], dataset: StorageDataset) -> StorageDatasetSpec:
    matches = [spec for spec in specs if spec.dataset == dataset]
    assert len(matches) == 1
    return matches[0]
