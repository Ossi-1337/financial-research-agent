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
