from __future__ import annotations

import tomllib
from importlib.resources import files
from pathlib import Path


def test_required_runtime_resources_are_packaged() -> None:
    assert files("financial_research_agent.web").joinpath("static/index.html").is_file()
    assert (
        files("financial_research_agent.persistence.migrations")
        .joinpath("0001_initial.sql")
        .is_file()
    )
    assets = files("financial_research_agent.report_exports").joinpath("assets")
    assert assets.joinpath("NotoSans-Regular.ttf").is_file()
    assert assets.joinpath("NotoSans-Bold.ttf").is_file()
    assert assets.joinpath("OFL.txt").is_file()
    assert (
        files("financial_research_agent.scenarios")
        .joinpath("data/novo_nordisk_context.v1.json")
        .is_file()
    )
    scenario_context = (
        files("financial_research_agent.scenarios")
        .joinpath("data/novo_nordisk_context.v1.json")
        .read_text(encoding="utf-8")
    )
    assert scenario_context.strip().startswith("{")


def test_mcp_sdk_is_optional_but_available_to_development_checks() -> None:
    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads(root.joinpath("pyproject.toml").read_text(encoding="utf-8"))
    optional = config["project"]["optional-dependencies"]

    assert optional["mcp"] == ["mcp>=1.28.1,<2"]
    assert "mcp>=1.28.1,<2" in optional["dev"]


def test_runtime_docker_stage_contains_only_installed_package() -> None:
    root = Path(__file__).resolve().parents[1]
    dockerfile = root.joinpath("Dockerfile").read_text(encoding="utf-8")
    runtime = dockerfile.split("FROM python:3.14-slim AS runtime", maxsplit=1)[1]

    assert "USER fra" in runtime
    assert "FRA_HOME=/data" in runtime
    assert "COPY src" not in runtime
    assert "COPY tests" not in runtime
    assert ".codex" not in runtime
    assert ".env" not in runtime


def test_compose_default_starts_orchestrator_and_specialist_topology() -> None:
    root = Path(__file__).resolve().parents[1]
    compose = root.joinpath("docker-compose.yml").read_text(encoding="utf-8")
    app_image = compose.split("services:", maxsplit=1)[0]

    assert "target: runtime" in app_image
    assert "FRA_LLM_PROVIDER: ${FRA_LLM_PROVIDER:-offline-test}" in compose
    assert 'command: ["chown -R 10001:10001 /data"]' in compose
    assert "condition: service_completed_successfully" in compose
    assert "profiles: [cpu]" in compose
    assert "profiles: [cuda]" in compose
    assert "profiles: [a2a]" not in compose
    assert "a2a-distributed" not in compose
    assert "financial-report-agent:" in compose
    assert "stock-agent:" in compose
    assert "context-agent:" in compose
    assert "synthesis-agent:" in compose
    assert "target: a2a-runtime" not in compose
    assert 'FRA_A2A_ENABLED: "true"' in compose
    assert "ghcr.io/ggml-org/llama.cpp:server}" in compose
    assert "ghcr.io/ggml-org/llama.cpp:server-cuda}" in compose
    assert "profiles: [searxng]" in compose
    assert "FRA_TAVILY_API_KEY" in compose
    assert "FRA_SEARXNG_BASE_URL" in compose
    assert root.joinpath("deploy/searxng/settings.yml").is_file()
