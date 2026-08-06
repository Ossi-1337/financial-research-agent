from __future__ import annotations

import json
from datetime import datetime
from threading import Lock

from financial_research_agent.persistence.database import SQLiteDatabase
from financial_research_agent.web_research.contracts import WebSourceEvidence


class InMemoryWebSourceCache:
    def __init__(self) -> None:
        self._items: dict[str, WebSourceEvidence] = {}
        self._lock = Lock()

    def get(self, canonical_url: str, *, now: datetime) -> WebSourceEvidence | None:
        with self._lock:
            item = self._items.get(canonical_url)
            return item if item is not None and item.expires_at > now else None

    def save(self, source: WebSourceEvidence) -> WebSourceEvidence:
        with self._lock:
            self._items[source.canonical_url] = source
        return source


class SQLiteWebSourceCache:
    def __init__(self, database: SQLiteDatabase) -> None:
        self.database = database

    def get(self, canonical_url: str, *, now: datetime) -> WebSourceEvidence | None:
        with self.database.read() as connection:
            row = connection.execute(
                """
                SELECT payload_json
                FROM web_source_evidence
                WHERE canonical_url = ? AND expires_at > ?
                """,
                (canonical_url, now.isoformat()),
            ).fetchone()
        if row is None:
            return None
        return WebSourceEvidence.from_dict(json.loads(str(row[0])))

    def save(self, source: WebSourceEvidence) -> WebSourceEvidence:
        payload = json.dumps(source.to_dict(), separators=(",", ":"), sort_keys=True)
        with self.database.transaction() as connection:
            connection.execute(
                """
                INSERT INTO web_source_evidence(
                    id, canonical_url, retrieved_at, expires_at, payload_version, payload_json
                ) VALUES (?, ?, ?, ?, 1, ?)
                ON CONFLICT(canonical_url) DO UPDATE SET
                    id = excluded.id,
                    retrieved_at = excluded.retrieved_at,
                    expires_at = excluded.expires_at,
                    payload_version = excluded.payload_version,
                    payload_json = excluded.payload_json
                """,
                (
                    source.id,
                    source.canonical_url,
                    source.retrieved_at.isoformat(),
                    source.expires_at.isoformat(),
                    payload,
                ),
            )
        return source
