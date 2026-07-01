from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

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
        choices=["health"],
        default="health",
        help="Command to run. Defaults to health.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "health":
        settings = Settings.from_env()
        health = build_health_report(settings)
        print(json.dumps(health, indent=2 if args.pretty else None, sort_keys=True))
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2
