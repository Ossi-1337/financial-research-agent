from __future__ import annotations

import json

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
