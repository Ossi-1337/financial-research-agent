from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_PYTHON = (3, 14)
CPU_MODEL = "ggml-org/gemma-3-1b-it-GGUF:Q4_K_M"
CUDA_MODEL = "unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF:UD-Q4_K_XL"
LOCAL_LLM_BASE_URL = "http://llama-cpp:8080/v1"


class DevCommandError(RuntimeError):
    pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Financial Research Agent developer commands.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("install", help="Install project and development dependencies.")

    run = commands.add_parser("run", help="Start local web UI without Docker.")
    run.add_argument("--host", default="127.0.0.1")
    run.add_argument("--port", type=int, default=8000)

    commands.add_parser("lint", help="Run Ruff lint and format checks.")
    commands.add_parser("test", help="Run pytest.")
    commands.add_parser("check", help="Run full Python verification and package build.")

    reset = commands.add_parser("reset", help="Reset guarded local application data.")
    reset.add_argument("--yes", action="store_true")

    docker_up = commands.add_parser("docker-up", help="Start Docker Compose stack.")
    docker_up.add_argument("--runtime", choices=("offline", "cpu", "cuda"), default="offline")
    docker_up.add_argument("--detach", action="store_true")
    commands.add_parser("docker-down", help="Stop Docker Compose stack.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _require_supported_python()
        if args.command == "install":
            _run([sys.executable, "-m", "pip", "install", "-e", ".[dev]"])
        elif args.command == "run":
            _run(
                [
                    sys.executable,
                    "-m",
                    "financial_research_agent",
                    "serve",
                    "--host",
                    args.host,
                    "--port",
                    str(args.port),
                ]
            )
        elif args.command == "lint":
            _lint()
        elif args.command == "test":
            _run([sys.executable, "-m", "pytest"])
        elif args.command == "check":
            _check()
        elif args.command == "reset":
            if not args.yes:
                parser.error("reset requires --yes")
            _run([sys.executable, "-m", "financial_research_agent", "data-reset", "--yes"])
        elif args.command == "docker-up":
            _docker_up(runtime=args.runtime, detach=args.detach)
        elif args.command == "docker-down":
            _require_docker()
            _run(["docker", "compose", "down"])
        return 0
    except (DevCommandError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _lint() -> None:
    _run([sys.executable, "-m", "ruff", "check", "."])
    _run([sys.executable, "-m", "ruff", "format", "--check", "."])


def _check() -> None:
    _lint()
    _run([sys.executable, "-m", "pytest"])
    _run([sys.executable, "-m", "compileall", "-q", "src", "tests"])
    _run([sys.executable, "-m", "financial_research_agent", "eval", "--pretty"])
    _run([sys.executable, "-m", "build"])


def _docker_up(*, runtime: str, detach: bool) -> None:
    _require_docker()
    if runtime == "cuda":
        _require_cuda_runtime()
    configured = _compose_environment() if runtime != "offline" else {}
    environment = _docker_environment(runtime, configured=configured)
    command = ["docker", "compose"]
    if runtime != "offline":
        command.extend(("--profile", runtime))
        cache = _project_path(environment.get("FRA_HUGGINGFACE_CACHE", ".docker/huggingface"))
        cache.mkdir(parents=True, exist_ok=True)
        environment["FRA_HUGGINGFACE_CACHE"] = str(cache)
    command.extend(("up", "--build"))
    if detach:
        command.append("--detach")
    _run(command, env=environment)


def _docker_environment(
    runtime: str, *, configured: dict[str, str] | None = None
) -> dict[str, str]:
    environment = os.environ.copy()
    if runtime == "offline":
        environment["FRA_LLM_PROVIDER"] = "offline-test"
        environment["FRA_LLM_MODEL"] = "offline-test"
        return environment
    model_variable = "FRA_CPU_MODEL" if runtime == "cpu" else "FRA_CUDA_MODEL"
    default_model = CPU_MODEL if runtime == "cpu" else CUDA_MODEL
    resolved = configured or {}
    model = environment.get(model_variable) or resolved.get(model_variable) or default_model
    environment[model_variable] = model
    cache = environment.get("FRA_HUGGINGFACE_CACHE") or resolved.get("FRA_HUGGINGFACE_CACHE")
    if cache:
        environment["FRA_HUGGINGFACE_CACHE"] = cache
    environment["FRA_LLM_PROVIDER"] = "local-openai"
    environment["FRA_LLM_MODEL"] = model
    environment["FRA_LLM_BASE_URL"] = LOCAL_LLM_BASE_URL
    environment["FRA_LLM_LOCAL_RUNTIME"] = "llama.cpp"
    return environment


def _compose_environment() -> dict[str, str]:
    result = _run(
        ["docker", "compose", "config", "--environment"],
        capture_output=True,
    )
    return {
        key: value
        for line in result.stdout.splitlines()
        if "=" in line
        for key, value in (line.split("=", maxsplit=1),)
    }


def _project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _require_supported_python() -> None:
    if sys.version_info[:2] != SUPPORTED_PYTHON:
        expected = ".".join(str(part) for part in SUPPORTED_PYTHON)
        actual = f"{sys.version_info.major}.{sys.version_info.minor}"
        raise DevCommandError(f"Python {expected} is required; current interpreter is {actual}.")


def _require_docker() -> None:
    if shutil.which("docker") is None:
        raise DevCommandError("Docker CLI is not installed or not on PATH.")
    try:
        _run(["docker", "info", "--format", "{{.ServerVersion}}"], capture_output=True)
    except subprocess.CalledProcessError as exc:
        raise DevCommandError("Docker daemon is unavailable; start Docker Desktop.") from exc


def _require_cuda_runtime() -> None:
    result = _run(
        ["docker", "info", "--format", "{{json .Runtimes}}"],
        capture_output=True,
    )
    if "nvidia" not in result.stdout.lower():
        raise DevCommandError(
            "Docker NVIDIA runtime is unavailable; use --runtime cpu or configure GPU support."
        )


def _run(
    command: Sequence[str],
    *,
    env: dict[str, str] | None = None,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
        shell=False,
        capture_output=capture_output,
        text=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
