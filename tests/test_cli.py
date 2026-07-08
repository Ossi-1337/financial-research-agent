from __future__ import annotations

import json
from pathlib import Path

import pytest

from financial_research_agent.cli import main


def test_health_command_outputs_json(capsys) -> None:
    exit_code = main(["health"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["app"] == "financial-research-agent"
    assert payload["status"] == "ok"
    assert payload["provider"]["llm_provider"] == "offline-test"


def test_default_command_is_health(capsys) -> None:
    exit_code = main([])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["status"] == "ok"


def test_serve_command_starts_uvicorn(monkeypatch) -> None:
    calls = {}

    def fake_run(app, *, host, port):
        calls["app"] = app
        calls["host"] = host
        calls["port"] = port

    monkeypatch.setattr("financial_research_agent.cli.uvicorn.run", fake_run)

    exit_code = main(["serve", "--host", "127.0.0.2", "--port", "8123"])

    assert exit_code == 0
    assert calls["app"].title == "Financial Research Agent"
    assert calls["host"] == "127.0.0.2"
    assert calls["port"] == 8123


def test_serve_rejects_remote_bind_without_explicit_opt_in(monkeypatch) -> None:
    monkeypatch.delenv("FRA_ALLOW_REMOTE_BIND", raising=False)

    with pytest.raises(SystemExit):
        main(["serve", "--host", "0.0.0.0"])


def test_serve_allows_remote_bind_with_explicit_opt_in(monkeypatch) -> None:
    calls = {}

    def fake_run(app, *, host, port):
        calls["app"] = app
        calls["host"] = host
        calls["port"] = port

    monkeypatch.setenv("FRA_ALLOW_REMOTE_BIND", "true")
    monkeypatch.setattr("financial_research_agent.cli.uvicorn.run", fake_run)

    assert main(["serve", "--host", "0.0.0.0"]) == 0
    assert calls["host"] == "0.0.0.0"


def test_storage_status_command_outputs_manifest(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setenv("FRA_HOME", str(tmp_path))

    exit_code = main(["storage-status", "--pretty"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["provider"] == "local-json"
    assert payload["app_home"] == str(tmp_path)
    assert any(entry["spec"]["id"] == "chat_sessions" for entry in payload["datasets"])


def test_storage_migrate_command_creates_local_layout(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setenv("FRA_HOME", str(tmp_path))

    exit_code = main(["storage-migrate"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["applied_migrations"][0]["id"] == "0001_local_json_storage_layout"
    assert tmp_path.joinpath("data").exists()
    assert tmp_path.joinpath("cache").exists()


def test_cache_clear_command_removes_cache(monkeypatch, tmp_path: Path, capsys) -> None:
    cache_path = tmp_path / "cache" / "sec_company_tickers.json"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("FRA_HOME", str(tmp_path))

    exit_code = main(["cache-clear"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["deleted_count"] == 1
    assert not cache_path.exists()


def test_data_reset_requires_confirmation(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FRA_HOME", str(tmp_path))

    with pytest.raises(SystemExit):
        main(["data-reset"])


def test_data_reset_command_removes_local_data(monkeypatch, tmp_path: Path, capsys) -> None:
    data_path = tmp_path / "data" / "chat_sessions.json"
    data_path.parent.mkdir(parents=True)
    data_path.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("FRA_HOME", str(tmp_path))

    exit_code = main(["data-reset", "--yes"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["operation"] == "reset_local_data"
    assert not data_path.exists()


def test_retrieval_status_command_outputs_index_metadata(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    monkeypatch.setenv("FRA_HOME", str(tmp_path))

    exit_code = main(["retrieval-status"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["provider"] == "local-vector"
    assert payload["record_count"] == 0
    assert payload["storage_path"] == str(tmp_path / "data" / "retrieval" / "vector_index.json")


def test_retrieval_clear_command_removes_index_file(monkeypatch, tmp_path: Path, capsys) -> None:
    index_path = tmp_path / "data" / "retrieval" / "vector_index.json"
    index_path.parent.mkdir(parents=True)
    index_path.write_text('{"version":1,"records":[]}', encoding="utf-8")
    monkeypatch.setenv("FRA_HOME", str(tmp_path))

    exit_code = main(["retrieval-clear"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["cleared_records"] == 0
    assert not index_path.exists()


def test_eval_command_runs_default_offline_harness(capsys) -> None:
    exit_code = main(["eval", "--pretty"])

    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["id"] == "default-offline-fixture"
    assert payload["status"] == "passed"
    assert payload["case_count"] == 3
    assert payload["failed_count"] == 0
