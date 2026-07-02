from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

import uvicorn

from financial_research_agent.health import build_health_report
from financial_research_agent.settings import Settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="financial-research-agent",
        description="Local-first financial research agent foundation.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=["health", "serve"],
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "health":
        settings = Settings.from_env()
        health = build_health_report(settings)
        print(json.dumps(health, indent=2 if args.pretty else None, sort_keys=True))
        return 0

    if args.command == "serve":
        from financial_research_agent.web import create_app

        settings = Settings.from_env()
        uvicorn.run(create_app(settings=settings), host=args.host, port=args.port)
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


def _port_value(value: str) -> int:
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("port must be an integer") from exc
    if port < 1 or port > 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port
