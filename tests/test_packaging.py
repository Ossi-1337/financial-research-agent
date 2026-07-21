from __future__ import annotations

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


def test_compose_default_is_offline_and_model_services_use_profiles() -> None:
    root = Path(__file__).resolve().parents[1]
    compose = root.joinpath("docker-compose.yml").read_text(encoding="utf-8")

    assert "FRA_LLM_PROVIDER: ${FRA_LLM_PROVIDER:-offline-test}" in compose
    assert 'command: ["chown -R 10001:10001 /data"]' in compose
    assert "condition: service_completed_successfully" in compose
    assert "profiles: [cpu]" in compose
    assert "profiles: [cuda]" in compose
    assert "ghcr.io/ggml-org/llama.cpp:server}" in compose
    assert "ghcr.io/ggml-org/llama.cpp:server-cuda}" in compose
