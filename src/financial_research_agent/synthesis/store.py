from __future__ import annotations

from collections.abc import Iterable
from threading import Lock

from financial_research_agent.synthesis.narrative import NarrativePresentation


class NarrativePresentationStore:
    def __init__(self, items: Iterable[NarrativePresentation] = ()) -> None:
        presentations = tuple(items)
        if any(not isinstance(item, NarrativePresentation) for item in presentations):
            raise ValueError("items must contain NarrativePresentation values")
        self._items = {item.id: item for item in presentations}
        self._lock = Lock()

    def matching(
        self,
        *,
        run_id: str,
        synthesis_sha256: str,
        prompt_id: str,
        prompt_version: str,
        provider: str,
        model: str,
    ) -> NarrativePresentation | None:
        with self._lock:
            matches = tuple(
                item
                for item in self._items.values()
                if item.run_id == run_id
                and item.synthesis_sha256 == synthesis_sha256
                and item.prompt_id == prompt_id
                and item.prompt_version == prompt_version
                and item.provider == provider
                and item.model == model
            )
        return max(matches, key=lambda item: item.created_at, default=None)
