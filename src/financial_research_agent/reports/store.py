from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any, Self

from financial_research_agent.reports.contracts import CitedResearchRun
from financial_research_agent.settings import Settings

REPORT_RUN_STORE_VERSION = 1


class CitedResearchRunStore:
    def __init__(self, *, storage_path: Path | None = None) -> None:
        self.storage_path = storage_path
        self._runs: dict[str, CitedResearchRun] = {}
        self._lock = Lock()
        self._load()

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        return cls(storage_path=settings.local_paths.data_dir / "report_runs.json")

    def save(self, run: CitedResearchRun) -> CitedResearchRun:
        if not isinstance(run, CitedResearchRun):
            raise ValueError("run must be a CitedResearchRun")
        with self._lock:
            self._runs[run.id] = run
            self._save()
        return run

    def get(self, run_id: str) -> CitedResearchRun | None:
        with self._lock:
            return self._runs.get(_require_text("run_id", run_id))

    def list(self) -> tuple[CitedResearchRun, ...]:
        with self._lock:
            return tuple(sorted(self._runs.values(), key=lambda run: run.created_at, reverse=True))

    def count(self) -> int:
        with self._lock:
            return len(self._runs)

    def _load(self) -> None:
        if self.storage_path is None or not self.storage_path.exists():
            return
        try:
            payload = json.loads(self.storage_path.read_text(encoding="utf-8"))
            if payload.get("version") != REPORT_RUN_STORE_VERSION:
                raise ValueError("unsupported report run store version")
            runs = payload.get("runs", ())
            if not isinstance(runs, list):
                raise ValueError("stored report runs must be a list")
            loaded = tuple(CitedResearchRun.from_dict(_payload_mapping(item)) for item in runs)
            self._runs = {run.id: run for run in loaded}
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"Could not load report run store: {self.storage_path}") from exc

    def _save(self) -> None:
        if self.storage_path is None:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": REPORT_RUN_STORE_VERSION,
            "runs": [run.to_dict() for run in self._list_unlocked()],
        }
        temp_path = self.storage_path.with_suffix(f"{self.storage_path.suffix}.tmp")
        temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temp_path.replace(self.storage_path)

    def _list_unlocked(self) -> tuple[CitedResearchRun, ...]:
        return tuple(sorted(self._runs.values(), key=lambda run: run.created_at, reverse=True))


def _payload_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("report run must be an object")
    return value


def _require_text(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    text = value.strip()
    if text == "":
        raise ValueError(f"{name} is required")
    return text
