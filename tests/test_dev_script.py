from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from scripts import dev


def test_install_uses_current_python_without_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(dev.subprocess, "run", fake_run)

    assert dev.main(["install"]) == 0
    assert calls == [
        (
            [sys.executable, "-m", "pip", "install", "-e", ".[dev]"],
            {
                "cwd": dev.PROJECT_ROOT,
                "env": None,
                "check": True,
                "shell": False,
                "capture_output": False,
                "text": True,
            },
        )
    ]


@pytest.mark.parametrize(
    ("runtime", "profile", "model_variable"),
    (
        ("cpu", "cpu", "FRA_CPU_MODEL"),
        ("cuda", "cuda", "FRA_CUDA_MODEL"),
    ),
)
def test_docker_up_selects_local_runtime_profile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    runtime: str,
    profile: str,
    model_variable: str,
) -> None:
    calls = []
    monkeypatch.setattr(dev, "_require_docker", lambda: None)
    monkeypatch.setattr(dev, "_require_cuda_runtime", lambda: None)

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if command[-2:] == ["config", "--environment"]:
            return subprocess.CompletedProcess(
                command,
                0,
                f"{model_variable}=dotenv-model\nFRA_HUGGINGFACE_CACHE={tmp_path / 'models'}\n",
                "",
            )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(dev, "_run", fake_run)

    assert dev.main(["docker-up", "--runtime", runtime, "--detach"]) == 0

    assert calls[0][0] == ["docker", "compose", "config", "--environment"]
    command, kwargs = calls[1]
    assert command == ["docker", "compose", "--profile", profile, "up", "--build", "--detach"]
    assert kwargs["env"]["FRA_LLM_PROVIDER"] == "local-openai"
    assert kwargs["env"]["FRA_LLM_MODEL"] == "dotenv-model"
    assert kwargs["env"][model_variable] == "dotenv-model"
    assert kwargs["env"]["FRA_LLM_BASE_URL"] == dev.LOCAL_LLM_BASE_URL
    assert kwargs["env"]["FRA_HUGGINGFACE_CACHE"] == str((tmp_path / "models").resolve())


def test_process_environment_overrides_compose_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FRA_CPU_MODEL", "process-model")

    environment = dev._docker_environment("cpu", configured={"FRA_CPU_MODEL": "dotenv-model"})

    assert environment["FRA_CPU_MODEL"] == "process-model"
    assert environment["FRA_LLM_MODEL"] == "process-model"


def test_relative_cache_path_is_resolved_from_project_root() -> None:
    assert dev._project_path(".docker/models") == (dev.PROJECT_ROOT / ".docker/models").resolve()


def test_docker_up_defaults_to_offline_without_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(dev, "_require_docker", lambda: None)

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(dev, "_run", fake_run)

    assert dev.main(["docker-up"]) == 0

    command, kwargs = calls[0]
    assert command == ["docker", "compose", "up", "--build"]
    assert kwargs["env"]["FRA_LLM_PROVIDER"] == "offline-test"
    assert kwargs["env"]["FRA_LLM_MODEL"] == "offline-test"


def test_missing_docker_returns_actionable_error(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr(dev.shutil, "which", lambda _name: None)

    assert dev.main(["docker-up"]) == 1
    assert "Docker CLI is not installed or not on PATH" in capsys.readouterr().err


def test_check_runs_expected_python_commands(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(dev, "_run", fake_run)

    assert dev.main(["check"]) == 0
    assert calls == [
        [sys.executable, "-m", "ruff", "check", "."],
        [sys.executable, "-m", "ruff", "format", "--check", "."],
        [sys.executable, "-m", "pytest"],
        [sys.executable, "-m", "compileall", "-q", "src", "tests"],
        [sys.executable, "-m", "financial_research_agent", "eval", "--pretty"],
        [sys.executable, "-m", "build"],
    ]


def test_project_root_points_to_repository() -> None:
    assert Path(__file__).resolve().parents[1] == dev.PROJECT_ROOT
