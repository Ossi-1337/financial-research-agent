from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

import uvicorn

from financial_research_agent.health import build_health_report
from financial_research_agent.retrieval import LocalVectorIndex
from financial_research_agent.settings import Settings
from financial_research_agent.storage import LocalStorageManager


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
            "cache-clear",
            "data-reset",
            "retrieval-status",
            "retrieval-clear",
            "eval",
        ],
        default="health",
        help="Command to run. Defaults to health.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
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
        "cache-clear",
        "data-reset",
        "retrieval-status",
        "retrieval-clear",
    }:
        settings = Settings.from_env()
        storage = LocalStorageManager.from_settings(settings)
        if args.command == "storage-status":
            _print_json(storage.inspect().to_dict(), pretty=args.pretty)
            return 0
        if args.command == "storage-migrate":
            _print_json(storage.migrate().to_dict(), pretty=args.pretty)
            return 0
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
        uvicorn.run(create_app(settings=settings), host=args.host, port=args.port)
        return 0

    if args.command == "eval":
        from financial_research_agent.evaluation import (
            EvalSuiteStatus,
            run_default_offline_evaluations,
        )

        result = run_default_offline_evaluations()
        _print_json(result.to_dict(), pretty=args.pretty)
        return 0 if result.status == EvalSuiteStatus.PASSED else 1

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
