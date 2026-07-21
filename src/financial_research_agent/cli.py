from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence
from pathlib import Path

import uvicorn
from dotenv import load_dotenv

from financial_research_agent.health import build_health_report
from financial_research_agent.persistence import (
    LegacyJsonImporter,
    PersistenceError,
    PersistenceErrorCode,
    SQLiteDatabase,
    SQLiteOperations,
    create_storage_manager,
    existing_legacy_paths,
)
from financial_research_agent.retrieval import LocalVectorIndex
from financial_research_agent.security import validate_bind_host
from financial_research_agent.settings import Settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="financial-research-agent",
        description="Local-first financial research agent foundation.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=[
            "health",
            "serve",
            "storage-status",
            "storage-migrate",
            "storage-check",
            "storage-backup",
            "storage-restore",
            "storage-cleanup",
            "cache-clear",
            "data-reset",
            "retrieval-status",
            "retrieval-clear",
            "eval",
            "scenario-run",
        ],
        default="health",
        help="Command to run. Defaults to health.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    parser.add_argument(
        "scenario_id",
        nargs="?",
        help="Scenario id used by scenario-run.",
    )
    parser.add_argument(
        "--with-local-qa",
        action="store_true",
        help="Run optional source-bounded chat-provider Q&A after deterministic synthesis.",
    )
    parser.add_argument(
        "--no-refresh",
        action="store_true",
        help="Use stored provider data instead of requiring a live scenario refresh.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for the local web UI when using the serve command.",
    )
    parser.add_argument(
        "--port",
        type=_port_value,
        default=8000,
        help="Port for the local web UI when using the serve command.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm destructive local data reset.",
    )
    parser.add_argument("--full", action="store_true", help="Run full SQLite integrity check.")
    parser.add_argument("--backup", help="Backup ID used by storage-restore.")
    parser.add_argument("--dataset", help="Dataset used by storage-cleanup.")
    parser.add_argument(
        "--older-than-days", type=int, help="Retention cutoff used by storage-cleanup."
    )
    parser.add_argument(
        "--include-source-documents",
        action="store_true",
        help="Allow filing source document cleanup.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "health":
        settings = Settings.from_env()
        health = build_health_report(settings)
        _print_json(health, pretty=args.pretty)
        return 0

    if args.command in {
        "storage-status",
        "storage-migrate",
        "storage-check",
        "storage-backup",
        "storage-restore",
        "storage-cleanup",
        "cache-clear",
        "data-reset",
        "retrieval-status",
        "retrieval-clear",
    }:
        settings = Settings.from_env()
        storage = create_storage_manager(settings)
        if args.command == "storage-status":
            _print_json(storage.inspect().to_dict(), pretty=args.pretty)
            return 0
        if args.command == "storage-migrate":
            return _run_storage_command(
                lambda: LegacyJsonImporter(settings).migrate().to_dict(), pretty=args.pretty
            )
        if args.command == "storage-check":

            def check_storage():
                database = _sqlite_database_for_operation(settings)
                return database.integrity(full=args.full).to_dict()

            return _run_storage_command(check_storage, pretty=args.pretty)
        if args.command == "storage-backup":
            return _run_storage_command(
                lambda: _sqlite_operations(settings).backup().to_dict(), pretty=args.pretty
            )
        if args.command == "storage-restore":
            if not args.backup:
                parser.error("storage-restore requires --backup")
            if not args.yes:
                parser.error("storage-restore requires --yes")
            return _run_storage_command(
                lambda: _sqlite_operations(settings).restore(args.backup), pretty=args.pretty
            )
        if args.command == "storage-cleanup":
            if not args.dataset or args.older_than_days is None:
                parser.error("storage-cleanup requires --dataset and --older-than-days")
            return _run_storage_command(
                lambda: (
                    _sqlite_operations(settings)
                    .cleanup(
                        dataset=args.dataset,
                        older_than_days=args.older_than_days,
                        confirmed=args.yes,
                        include_source_documents=args.include_source_documents,
                    )
                    .to_dict()
                ),
                pretty=args.pretty,
            )
        if args.command == "cache-clear":
            _print_json(storage.clear_cache().to_dict(), pretty=args.pretty)
            return 0
        if args.command == "retrieval-status":
            _print_json(
                LocalVectorIndex.from_settings(settings).metadata().to_dict(),
                pretty=args.pretty,
            )
            return 0
        if args.command == "retrieval-clear":
            index = LocalVectorIndex.from_settings(settings)
            cleared_records = index.clear()
            _print_json(
                {
                    "cleared_records": cleared_records,
                    "index": index.metadata().to_dict(),
                },
                pretty=args.pretty,
            )
            return 0
        if not args.yes:
            parser.error("data-reset requires --yes")
        _print_json(storage.reset_local_data().to_dict(), pretty=args.pretty)
        return 0

    if args.command == "serve":
        from financial_research_agent.web import create_app

        settings = Settings.from_env()
        try:
            host = validate_bind_host(
                args.host,
                allow_remote_bind=settings.security.allow_remote_bind,
            )
        except ValueError as exc:
            parser.error(str(exc))
        try:
            app = create_app(settings=settings)
        except PersistenceError as exc:
            _print_json(exc.to_dict(), pretty=args.pretty)
            return 1
        uvicorn.run(app, host=host, port=args.port)
        return 0

    if args.command == "eval":
        from financial_research_agent.evaluation import (
            EvalSuiteStatus,
            run_default_offline_evaluations,
        )

        result = run_default_offline_evaluations()
        _print_json(result.to_dict(), pretty=args.pretty)
        return 0 if result.status == EvalSuiteStatus.PASSED else 1

    if args.command == "scenario-run":
        if not args.scenario_id:
            parser.error("scenario-run requires a scenario id")
        from financial_research_agent.scenarios import (
            ScenarioError,
            ScenarioExecutionStatus,
            ScenarioRunner,
        )
        from financial_research_agent.settings import ProviderTask
        from financial_research_agent.web import create_app

        settings = Settings.from_env()
        try:
            app = create_app(settings=settings)
            selection = settings.provider.selection_for_task(ProviderTask.CHAT)
            runner = ScenarioRunner(
                settings=settings,
                catalog=app.state.scenario_catalog,
                orchestrator=app.state.orchestrator,
                export_service=app.state.report_export_service,
                chat_provider=(
                    app.state.provider_registry.chat_provider(selection.provider)
                    if args.with_local_qa
                    and app.state.provider_registry.has_chat_provider(selection.provider)
                    else None
                ),
                chat_model=selection.model,
            )
            result = asyncio.run(
                runner.run(
                    args.scenario_id,
                    refresh=not args.no_refresh,
                    with_local_qa=args.with_local_qa,
                )
            )
        except (PersistenceError, ScenarioError) as exc:
            _print_json(exc.to_dict(), pretty=args.pretty)
            return 1
        _print_json(result.to_dict(), pretty=args.pretty)
        return 0 if result.status != ScenarioExecutionStatus.FAILED else 1

    parser.error(f"Unsupported command: {args.command}")
    return 2


def _print_json(payload: object, *, pretty: bool) -> None:
    print(json.dumps(payload, indent=2 if pretty else None, sort_keys=True))


def _port_value(value: str) -> int:
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("port must be an integer") from exc
    if port < 1 or port > 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def _sqlite_database_for_operation(settings: Settings) -> SQLiteDatabase:
    if settings.storage.provider != "sqlite":
        raise PersistenceError(
            PersistenceErrorCode.INCOMPATIBLE_SCHEMA,
            "This command requires FRA_STORAGE_PROVIDER=sqlite.",
        )
    database = SQLiteDatabase.from_data_dir(settings.local_paths.data_dir)
    if not database.path.exists() and existing_legacy_paths(settings.local_paths.data_dir):
        raise PersistenceError(
            PersistenceErrorCode.STORAGE_MIGRATION_REQUIRED,
            "Legacy JSON storage exists. Run storage-migrate first.",
        )
    database.initialize()
    return database


def _sqlite_operations(settings: Settings) -> SQLiteOperations:
    return SQLiteOperations(
        _sqlite_database_for_operation(settings), app_home=settings.local_paths.app_home
    )


def _run_storage_command(operation, *, pretty: bool) -> int:
    try:
        payload = operation()
    except (PersistenceError, ValueError) as exc:
        error = (
            exc.to_dict()
            if isinstance(exc, PersistenceError)
            else {
                "error": "invalid_storage_operation",
                "message": str(exc),
            }
        )
        _print_json(error, pretty=pretty)
        return 1
    _print_json(payload, pretty=pretty)
    return 0
