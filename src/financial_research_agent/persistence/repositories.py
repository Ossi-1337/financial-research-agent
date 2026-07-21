from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from financial_research_agent.background import BackgroundResearchJob, BackgroundResearchStatus
from financial_research_agent.filings.contracts import FilingIngestionResult
from financial_research_agent.filings.store import filing_ingestion_result_from_dict
from financial_research_agent.llm import MessageRole
from financial_research_agent.market_data.contracts import HistoricalPriceResult
from financial_research_agent.market_data.store import historical_price_result_from_dict
from financial_research_agent.orchestration.contracts import OrchestratedResearchRun
from financial_research_agent.orchestration.store import orchestrated_research_run_from_dict
from financial_research_agent.persistence.database import SQLiteDatabase
from financial_research_agent.reports import CitedResearchRun
from financial_research_agent.runtime_settings import RuntimeSettingsOverrides
from financial_research_agent.settings import Settings
from financial_research_agent.statements.contracts import FinancialStatementResult
from financial_research_agent.statements.store import financial_statement_result_from_dict
from financial_research_agent.web.sessions import (
    ChatMention,
    ChatSession,
    ChatSessionMessage,
    summarize_messages,
)

PAYLOAD_VERSION = 1


class SQLiteChatSessionStore:
    def __init__(
        self, database: SQLiteDatabase, *, recent_turns: int, summary_max_chars: int
    ) -> None:
        self.database = database
        self.storage_path = database.path
        self.recent_turns = recent_turns
        self.summary_max_chars = summary_max_chars

    def create(self) -> ChatSession:
        now = datetime.now(UTC)
        session = ChatSession(id=f"session_{uuid4().hex}", created_at=now, updated_at=now)
        self._save(session)
        return session

    def save(self, session: ChatSession) -> ChatSession:
        if not isinstance(session, ChatSession):
            raise ValueError("session must be a ChatSession")
        self._save(session)
        return session

    def list(self, *, limit: int = 50) -> tuple[ChatSession, ...]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        with self.database.read() as connection:
            ids = tuple(
                str(row["id"])
                for row in connection.execute(
                    "SELECT id FROM chat_sessions ORDER BY updated_at DESC LIMIT ?", (limit,)
                )
            )
        return tuple(session for session_id in ids if (session := self.get(session_id)) is not None)

    def get(self, session_id: str) -> ChatSession | None:
        with self.database.read() as connection:
            row = connection.execute(
                "SELECT * FROM chat_sessions WHERE id = ?", (_text("session_id", session_id),)
            ).fetchone()
            if row is None:
                return None
            messages = tuple(
                ChatSessionMessage.from_dict(json.loads(item["payload_json"]))
                for item in connection.execute(
                    "SELECT payload_json FROM chat_messages WHERE session_id = ? ORDER BY sequence",
                    (session_id,),
                )
            )
        return ChatSession(
            id=str(row["id"]),
            created_at=_datetime(row["created_at"]),
            updated_at=_datetime(row["updated_at"]),
            summary=row["summary"],
            messages=messages,
        )

    def count(self) -> int:
        return _count(self.database, "chat_sessions")

    def delete(self, session_id: str) -> bool:
        with self.database.transaction() as connection:
            cursor = connection.execute(
                "DELETE FROM chat_sessions WHERE id = ?", (_text("session_id", session_id),)
            )
            return cursor.rowcount > 0

    def clear(self) -> int:
        return _clear(self.database, "chat_sessions")

    def append_exchange(
        self,
        *,
        session_id: str,
        user_content: str,
        assistant_content: str,
        provider: str,
        model: str,
        research_run_id: str | None = None,
        mentions: tuple[ChatMention, ...] = (),
        citations=(),
        evidence_snippets=(),
        synthesis_report=None,
    ) -> ChatSession:
        with self.database.transaction():
            session = self.get(session_id)
            if session is None:
                raise KeyError(session_id)
            now = datetime.now(UTC)
            messages = (
                *session.messages,
                ChatSessionMessage(
                    id=f"message_{uuid4().hex}",
                    role=MessageRole.USER,
                    content=user_content,
                    created_at=now,
                    research_run_id=research_run_id,
                    mentions=mentions,
                ),
                ChatSessionMessage(
                    id=f"message_{uuid4().hex}",
                    role=MessageRole.ASSISTANT,
                    content=assistant_content,
                    created_at=datetime.now(UTC),
                    provider=provider,
                    model=model,
                    research_run_id=research_run_id,
                    citations=tuple(citations),
                    evidence_snippets=tuple(evidence_snippets),
                    synthesis_report=synthesis_report,
                ),
            )
            recent_count = self.recent_turns * 2
            older = messages[:-recent_count] if len(messages) > recent_count else ()
            updated = replace(
                session,
                updated_at=datetime.now(UTC),
                messages=messages,
                summary=summarize_messages(older, max_chars=self.summary_max_chars),
            )
            self._save(updated)
        return updated

    def _save(self, session: ChatSession) -> None:
        with self.database.transaction() as connection:
            connection.execute(
                """
                INSERT INTO chat_sessions(id, created_at, updated_at, summary)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    updated_at=excluded.updated_at, summary=excluded.summary
                """,
                (session.id, _iso(session.created_at), _iso(session.updated_at), session.summary),
            )
            connection.execute("DELETE FROM chat_messages WHERE session_id = ?", (session.id,))
            connection.executemany(
                """
                INSERT INTO chat_messages(id, session_id, sequence, role, created_at, payload_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    (
                        message.id,
                        session.id,
                        sequence,
                        message.role.value,
                        _iso(message.created_at),
                        _json(message.to_dict()),
                    )
                    for sequence, message in enumerate(session.messages)
                ),
            )


class SQLiteMarketDataStore:
    def __init__(self, database: SQLiteDatabase, *, stale_after: timedelta) -> None:
        self.database = database
        self.storage_path = database.path
        self.stale_after = stale_after

    def save_history(self, result: HistoricalPriceResult) -> HistoricalPriceResult:
        security_id = _upsert_security(self.database, result.security.to_dict())
        series_id = _stable_id("market_series", result.source.provider, result.security.symbol)
        with self.database.transaction() as connection:
            connection.execute(
                """
                INSERT INTO market_series(id, provider, symbol, security_id, retrieved_at,
                    payload_version, payload_json) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(provider, symbol) DO UPDATE SET security_id=excluded.security_id,
                    retrieved_at=excluded.retrieved_at, payload_version=excluded.payload_version,
                    payload_json=excluded.payload_json
                """,
                (
                    series_id,
                    result.source.provider,
                    result.security.symbol,
                    security_id,
                    _iso(result.source.retrieved_at),
                    PAYLOAD_VERSION,
                    _json(result.to_dict()),
                ),
            )
            connection.execute("DELETE FROM price_bars WHERE series_id = ?", (series_id,))
            connection.executemany(
                """
                INSERT INTO price_bars(series_id, priced_at, open_value, high_value, low_value,
                    close_value, adjusted_close_value, volume) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    (
                        series_id,
                        bar.priced_at.isoformat(),
                        str(bar.open),
                        str(bar.high),
                        str(bar.low),
                        str(bar.close),
                        str(bar.adjusted_close) if bar.adjusted_close is not None else None,
                        bar.volume,
                    )
                    for bar in result.bars
                ),
            )
        return result

    def get_history(self, *, symbol: str, provider: str | None = None, now=None):
        query = "SELECT payload_json FROM market_series WHERE symbol = ?"
        params: list[object] = [symbol.strip().upper()]
        if provider is not None:
            query += " AND provider = ?"
            params.append(provider)
        query += " ORDER BY retrieved_at DESC LIMIT 1"
        with self.database.read() as connection:
            row = connection.execute(query, params).fetchone()
        if row is None:
            return None
        result = historical_price_result_from_dict(json.loads(row["payload_json"]))
        current = now or datetime.now(UTC)
        if result.source.retrieved_at + self.stale_after > current:
            return result
        warning = "Stored market data is stale; refresh before relying on it."
        return replace(
            result,
            source=replace(result.source, freshness_warning=warning),
            warnings=tuple(dict.fromkeys((*result.warnings, warning))),
        )

    def count(self) -> int:
        return _count(self.database, "market_series")

    def list(self) -> tuple[HistoricalPriceResult, ...]:
        with self.database.read() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM market_series ORDER BY retrieved_at DESC"
            )
            return tuple(historical_price_result_from_dict(json.loads(row[0])) for row in rows)

    def clear(self) -> int:
        return _clear(self.database, "market_series")


class SQLiteFinancialStatementStore:
    def __init__(self, database: SQLiteDatabase, *, stale_after: timedelta) -> None:
        self.database = database
        self.storage_path = database.path
        self.stale_after = stale_after

    def save_result(self, result: FinancialStatementResult) -> FinancialStatementResult:
        company_id = _upsert_company(self.database, result.company.to_dict())
        result_id = _stable_id("statement_result", result.source.provider, result.company.cik)
        with self.database.transaction() as connection:
            connection.execute(
                """
                INSERT INTO statement_results(id, provider, cik, company_id, retrieved_at,
                    payload_version, payload_json) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(provider, cik) DO UPDATE SET company_id=excluded.company_id,
                    retrieved_at=excluded.retrieved_at, payload_version=excluded.payload_version,
                    payload_json=excluded.payload_json
                """,
                (
                    result_id,
                    result.source.provider,
                    result.company.cik,
                    company_id,
                    _iso(result.source.retrieved_at),
                    PAYLOAD_VERSION,
                    _json(result.to_dict()),
                ),
            )
            connection.execute("DELETE FROM financial_statements WHERE result_id = ?", (result_id,))
            connection.executemany(
                """
                INSERT INTO financial_statements(id, result_id, statement_type, fiscal_year,
                    period_end, currency, payload_json) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    (
                        item.id,
                        result_id,
                        item.statement_type.value,
                        item.period.fiscal_year,
                        item.period.period_end.isoformat(),
                        item.currency,
                        _json(item.to_dict()),
                    )
                    for item in result.statements
                ),
            )
        return result

    def get_result(self, *, cik: str, provider: str | None = None, now=None):
        row = _latest_provider_payload(self.database, "statement_results", "cik", cik, provider)
        if row is None:
            return None
        result = financial_statement_result_from_dict(json.loads(row["payload_json"]))
        current = now or datetime.now(UTC)
        if result.source.retrieved_at + self.stale_after > current:
            return result
        warning = "Stored financial statements are stale; refresh before relying on them."
        source = replace(result.source, freshness_warning=warning)
        return replace(
            result,
            source=source,
            statements=tuple(replace(item, source=source) for item in result.statements),
            warnings=tuple(dict.fromkeys((*result.warnings, warning))),
        )

    def count(self) -> int:
        return _count(self.database, "statement_results")

    def list(self) -> tuple[FinancialStatementResult, ...]:
        with self.database.read() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM statement_results ORDER BY retrieved_at DESC"
            )
            return tuple(financial_statement_result_from_dict(json.loads(row[0])) for row in rows)

    def clear(self) -> int:
        return _clear(self.database, "statement_results")


class SQLiteFilingStore:
    def __init__(self, database: SQLiteDatabase, *, stale_after: timedelta) -> None:
        self.database = database
        self.storage_path = database.path
        self.stale_after = stale_after

    def save_result(self, result: FilingIngestionResult) -> FilingIngestionResult:
        company_id = _upsert_company(self.database, result.company.to_dict())
        result_id = _stable_id("filing_result", result.source.provider, result.company.cik)
        payload = result.to_dict()
        for chunk in payload["chunks"]:
            chunk.pop("text", None)
        with self.database.transaction() as connection:
            connection.execute(
                """
                INSERT INTO filing_results(id, provider, cik, company_id, retrieved_at,
                    payload_version, payload_json) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(provider, cik) DO UPDATE SET company_id=excluded.company_id,
                    retrieved_at=excluded.retrieved_at, payload_version=excluded.payload_version,
                    payload_json=excluded.payload_json
                """,
                (
                    result_id,
                    result.source.provider,
                    result.company.cik,
                    company_id,
                    _iso(result.source.retrieved_at),
                    PAYLOAD_VERSION,
                    _json(payload),
                ),
            )
            connection.execute("DELETE FROM filings WHERE result_id = ?", (result_id,))
            for filing in result.filings:
                connection.execute(
                    """
                    INSERT INTO filings(
                        id, result_id, accession_number, form_type, filing_date,
                        local_raw_path, local_text_path, payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        filing.id,
                        result_id,
                        filing.accession_number,
                        filing.form_type,
                        filing.filing_date.isoformat(),
                        filing.local_raw_path,
                        filing.local_text_path,
                        _json(filing.to_dict()),
                    ),
                )
            connection.executemany(
                """
                INSERT INTO filing_chunks(id, filing_id, chunk_index, char_start, char_end,
                    section_heading, metadata_json) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    (
                        chunk.id,
                        chunk.filing_id,
                        chunk.chunk_index,
                        chunk.char_start,
                        chunk.char_end,
                        chunk.section_heading,
                        _json(dict(chunk.metadata)),
                    )
                    for chunk in result.chunks
                ),
            )
        return result

    def get_result(self, *, cik: str, provider: str | None = None, now=None):
        row = _latest_provider_payload(self.database, "filing_results", "cik", cik, provider)
        if row is None:
            return None
        payload = json.loads(row["payload_json"])
        filings = {item["id"]: item for item in payload.get("filings", ())}
        for chunk in payload.get("chunks", ()):
            filing = filings.get(chunk.get("filing_id"))
            chunk["text"] = _chunk_text(filing, chunk)
        result = filing_ingestion_result_from_dict(payload)
        current = now or datetime.now(UTC)
        if result.source.retrieved_at + self.stale_after > current:
            return result
        warning = "Stored filing documents are stale; refresh before relying on them."
        source = replace(result.source, freshness_warning=warning)
        return replace(
            result,
            source=source,
            filings=tuple(
                replace(
                    item,
                    source=replace(item.source, freshness_warning=warning),
                    warnings=tuple(dict.fromkeys((*item.warnings, warning))),
                )
                for item in result.filings
            ),
            warnings=tuple(dict.fromkeys((*result.warnings, warning))),
        )

    def count(self) -> int:
        return _count(self.database, "filing_results")

    def list(self) -> tuple[FilingIngestionResult, ...]:
        with self.database.read() as connection:
            keys = tuple(
                (str(row["cik"]), str(row["provider"]))
                for row in connection.execute(
                    "SELECT cik, provider FROM filing_results ORDER BY retrieved_at DESC"
                )
            )
        return tuple(
            result
            for cik, provider in keys
            if (result := self.get_result(cik=cik, provider=provider)) is not None
        )

    def clear(self) -> int:
        return _clear(self.database, "filing_results")


class SQLiteCitedResearchRunStore:
    def __init__(self, database: SQLiteDatabase) -> None:
        self.database = database
        self.storage_path = database.path

    def save(self, run: CitedResearchRun) -> CitedResearchRun:
        if not isinstance(run, CitedResearchRun):
            raise ValueError("run must be a CitedResearchRun")
        with self.database.transaction() as connection:
            connection.execute(
                """
                INSERT INTO cited_runs(id, status, created_at, query, payload_version, payload_json)
                VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET status=excluded.status,
                    payload_version=excluded.payload_version, payload_json=excluded.payload_json
                """,
                (
                    run.id,
                    run.status.value,
                    _iso(run.created_at),
                    run.query,
                    PAYLOAD_VERSION,
                    _json(run.to_dict()),
                ),
            )
            connection.execute("DELETE FROM citations WHERE run_id = ?", (run.id,))
            connection.execute("DELETE FROM evidence_snippets WHERE run_id = ?", (run.id,))
            connection.executemany(
                "INSERT INTO citations(run_id, citation_id, source_url, payload_json) "
                "VALUES (?, ?, ?, ?)",
                (
                    (run.id, item.id, item.source_url, _json(item.to_dict()))
                    for item in run.citations
                ),
            )
            connection.executemany(
                "INSERT INTO evidence_snippets(run_id, evidence_id, source_url, payload_json) "
                "VALUES (?, ?, ?, ?)",
                (
                    (run.id, item.id, item.source_url, _json(item.to_dict()))
                    for item in run.evidence
                ),
            )
        return run

    def get(self, run_id: str):
        with self.database.read() as connection:
            row = connection.execute(
                "SELECT payload_json FROM cited_runs WHERE id = ?", (run_id,)
            ).fetchone()
        return None if row is None else CitedResearchRun.from_dict(json.loads(row["payload_json"]))

    def list(self):
        with self.database.read() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM cited_runs ORDER BY created_at DESC"
            )
            return tuple(
                CitedResearchRun.from_dict(json.loads(row["payload_json"])) for row in rows
            )

    def count(self) -> int:
        return _count(self.database, "cited_runs")

    def clear(self) -> int:
        return _clear(self.database, "cited_runs")


class SQLiteOrchestratorRunStore:
    def __init__(self, database: SQLiteDatabase) -> None:
        self.database = database
        self.storage_path = database.path

    def save(self, run: OrchestratedResearchRun) -> OrchestratedResearchRun:
        with self.database.transaction() as connection:
            company_id = _selected_company(self.database, run.selected_company)
            security_id = _selected_security(self.database, run.selected_security, company_id)
            connection.execute(
                """
                INSERT INTO orchestrator_runs(id, status, created_at, updated_at, company_id,
                    security_id, payload_version, payload_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    status=excluded.status, updated_at=excluded.updated_at,
                    company_id=excluded.company_id, security_id=excluded.security_id,
                    payload_version=excluded.payload_version, payload_json=excluded.payload_json
                """,
                (
                    run.id,
                    run.status.value,
                    _iso(run.created_at),
                    _iso(run.updated_at),
                    company_id,
                    security_id,
                    PAYLOAD_VERSION,
                    _json(run.to_dict()),
                ),
            )
            connection.execute("DELETE FROM agent_handoffs WHERE run_id = ?", (run.id,))
            connection.executemany(
                """
                INSERT INTO agent_handoffs(id, run_id, step_id, kind, status, completed_at,
                    payload_json) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    (
                        item.id,
                        run.id,
                        item.step_id,
                        item.kind.value,
                        item.status.value,
                        _iso(item.completed_at),
                        _json(item.to_dict()),
                    )
                    for item in run.handoffs
                ),
            )
        return run

    def get(self, run_id: str):
        with self.database.read() as connection:
            row = connection.execute(
                "SELECT payload_json FROM orchestrator_runs WHERE id = ?", (run_id,)
            ).fetchone()
        return (
            None
            if row is None
            else orchestrated_research_run_from_dict(json.loads(row["payload_json"]))
        )

    def list(self):
        with self.database.read() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM orchestrator_runs ORDER BY updated_at DESC"
            )
            return tuple(
                orchestrated_research_run_from_dict(json.loads(row["payload_json"])) for row in rows
            )

    def count(self) -> int:
        return _count(self.database, "orchestrator_runs")

    def clear(self) -> int:
        return _clear(self.database, "orchestrator_runs")


class SQLiteRuntimeSettingsStore:
    def __init__(self, database: SQLiteDatabase) -> None:
        self.database = database
        self.storage_path = database.path

    def get(self) -> RuntimeSettingsOverrides:
        with self.database.read() as connection:
            row = connection.execute(
                "SELECT payload_json FROM runtime_settings WHERE id = 1"
            ).fetchone()
        return (
            RuntimeSettingsOverrides()
            if row is None
            else RuntimeSettingsOverrides.from_mapping(json.loads(row["payload_json"]))
        )

    def settings(self, base_settings: Settings) -> Settings:
        return self.get().apply_to(base_settings)

    def update(self, updates: RuntimeSettingsOverrides, *, base_settings: Settings):
        return self.replace(self.get().merged_with(updates), base_settings=base_settings)

    def replace(self, overrides: RuntimeSettingsOverrides, *, base_settings: Settings):
        overrides.apply_to(base_settings)
        with self.database.transaction() as connection:
            connection.execute(
                """
                INSERT INTO runtime_settings(id, updated_at, payload_version, payload_json)
                VALUES (1, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET updated_at=excluded.updated_at,
                    payload_version=excluded.payload_version, payload_json=excluded.payload_json
                """,
                (_iso(datetime.now(UTC)), PAYLOAD_VERSION, _json(overrides.to_dict())),
            )
        return overrides

    def clear(self) -> RuntimeSettingsOverrides:
        with self.database.transaction() as connection:
            connection.execute("DELETE FROM runtime_settings")
        return RuntimeSettingsOverrides()

    def count(self) -> int:
        return _count(self.database, "runtime_settings")


class SQLiteBackgroundJobStore:
    def __init__(self, database: SQLiteDatabase) -> None:
        self.database = database

    def save(self, job: BackgroundResearchJob) -> BackgroundResearchJob:
        with self.database.transaction() as connection:
            connection.execute(
                """
                INSERT INTO background_jobs(id, orchestrator_run_id, status, created_at, updated_at,
                    completed_at, payload_version, payload_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    status=excluded.status, updated_at=excluded.updated_at,
                    completed_at=excluded.completed_at, payload_version=excluded.payload_version,
                    payload_json=excluded.payload_json
                """,
                (
                    job.id,
                    job.orchestrator_run_id,
                    job.status.value,
                    _iso(job.created_at),
                    _iso(job.updated_at),
                    _iso(job.completed_at) if job.completed_at else None,
                    PAYLOAD_VERSION,
                    _json(job.to_dict()),
                ),
            )
        return job

    def get(self, job_id: str) -> BackgroundResearchJob | None:
        with self.database.read() as connection:
            row = connection.execute(
                "SELECT payload_json FROM background_jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return (
            None
            if row is None
            else BackgroundResearchJob.from_dict(json.loads(row["payload_json"]))
        )

    def list(self) -> tuple[BackgroundResearchJob, ...]:
        with self.database.read() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM background_jobs ORDER BY updated_at DESC"
            )
            return tuple(
                BackgroundResearchJob.from_dict(json.loads(row["payload_json"])) for row in rows
            )

    def count(self) -> int:
        return _count(self.database, "background_jobs")

    def clear(self) -> int:
        return _clear(self.database, "background_jobs")

    def fail_unfinished(self, *, now: datetime) -> int:
        jobs = [
            job
            for job in self.list()
            if job.status in {BackgroundResearchStatus.QUEUED, BackgroundResearchStatus.RUNNING}
        ]
        for job in jobs:
            self.save(
                replace(
                    job,
                    status=BackgroundResearchStatus.FAILED,
                    updated_at=now,
                    completed_at=now,
                    error_code="process_restarted",
                    error_message="Background process restarted; job was not resumed.",
                )
            )
        return len(jobs)


def _upsert_company(database: SQLiteDatabase, payload: dict[str, object]) -> str:
    identifiers = _identifier_pairs(payload)
    cik = _optional(payload.get("cik")) or _identifier_value(identifiers, "cik")
    legal_name = _optional(payload.get("legal_name")) or _optional(payload.get("display_name"))
    company_id = (
        _optional(payload.get("company_id"))
        or _optional(payload.get("id"))
        or _stable_id("company", cik or legal_name or _json(payload))
    )
    company_identifiers = tuple(
        dict.fromkeys(
            (
                *(((("cik", cik),)) if cik is not None else ()),
                *(item for item in identifiers if item[0] in {"cik", "lei"}),
            )
        )
    )
    with database.transaction() as connection:
        for scheme, value in company_identifiers:
            existing = connection.execute(
                "SELECT company_id FROM company_identifiers WHERE scheme = ? AND value = ?",
                (scheme, value),
            ).fetchone()
            if existing is not None and existing["company_id"] != company_id:
                raise ValueError(f"{scheme.upper()} is already assigned to a different company")
        connection.execute(
            """
            INSERT INTO companies(id, legal_name, cik, payload_json) VALUES (?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET legal_name=COALESCE(excluded.legal_name, legal_name),
                cik=COALESCE(excluded.cik, cik), payload_json=excluded.payload_json
            """,
            (company_id, legal_name, cik, _json(payload)),
        )
        for scheme, value in company_identifiers:
            connection.execute(
                "INSERT OR IGNORE INTO company_identifiers(company_id, scheme, value) "
                "VALUES (?, ?, ?)",
                (company_id, scheme, value),
            )
    return company_id


def _upsert_security(
    database: SQLiteDatabase, payload: dict[str, object], company_id: str | None = None
) -> str:
    raw_symbol = payload.get("symbol") or payload.get("ticker")
    symbol = _text("symbol", str(raw_symbol or "")).upper()
    security_id = (
        _optional(payload.get("security_id"))
        or _optional(payload.get("id"))
        or _stable_id("security", _optional(payload.get("exchange_mic")) or "unknown", symbol)
    )
    with database.transaction() as connection:
        connection.execute(
            """
            INSERT INTO securities(id, company_id, symbol, exchange_mic, currency, payload_json)
            VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET
                company_id=COALESCE(excluded.company_id, company_id), symbol=excluded.symbol,
                exchange_mic=excluded.exchange_mic, currency=excluded.currency,
                payload_json=excluded.payload_json
            """,
            (
                security_id,
                company_id,
                symbol,
                _optional(payload.get("exchange_mic")),
                _optional(payload.get("currency")),
                _json(payload),
            ),
        )
        ticker_scheme = f"ticker:{_optional(payload.get('exchange_mic')) or 'unknown'}"
        identifiers = _identifier_pairs(payload)
        isin = _optional(payload.get("isin"))
        security_identifiers = tuple(
            dict.fromkeys(
                (
                    (ticker_scheme, symbol),
                    *(((("isin", isin),)) if isin is not None else ()),
                    *(item for item in identifiers if item[0] in {"isin", "figi"}),
                )
            )
        )
        for scheme, value in security_identifiers:
            existing = connection.execute(
                "SELECT security_id FROM security_identifiers WHERE scheme = ? AND value = ?",
                (scheme, value),
            ).fetchone()
            if existing is not None and existing["security_id"] != security_id:
                raise ValueError(f"{scheme.upper()} is already assigned to a different security")
            connection.execute(
                "INSERT OR IGNORE INTO security_identifiers(security_id, scheme, value) "
                "VALUES (?, ?, ?)",
                (security_id, scheme, value),
            )
    return security_id


def _selected_company(database: SQLiteDatabase, payload) -> str | None:
    if payload is None:
        return None
    values = dict(payload)
    values.setdefault("company_id", values.get("id"))
    return _upsert_company(database, values)


def _selected_security(database: SQLiteDatabase, payload, company_id: str | None) -> str | None:
    if payload is None or not ({"symbol", "ticker"} & set(payload)):
        return None
    values = dict(payload)
    values.setdefault("security_id", values.get("id"))
    return _upsert_security(database, values, company_id)


def _identifier_pairs(payload: Mapping[str, object]) -> tuple[tuple[str, str], ...]:
    raw_identifiers = payload.get("identifiers") or ()
    if isinstance(raw_identifiers, (str, bytes)):
        raise ValueError("identifiers must be a collection")
    identifiers = []
    for item in raw_identifiers:
        if not isinstance(item, Mapping):
            raise ValueError("identifier must be an object")
        scheme = _text("identifier type", str(item.get("type") or "")).lower()
        value = _text("identifier value", str(item.get("value") or ""))
        identifiers.append((scheme, value))
    return tuple(dict.fromkeys(identifiers))


def _identifier_value(identifiers: tuple[tuple[str, str], ...], scheme: str) -> str | None:
    return next((value for kind, value in identifiers if kind == scheme), None)


def _latest_provider_payload(
    database: SQLiteDatabase, table: str, key: str, value: str, provider: str | None
):
    query = f'SELECT payload_json FROM "{table}" WHERE "{key}" = ?'
    params: list[object] = [value]
    if provider is not None:
        query += " AND provider = ?"
        params.append(provider)
    query += " ORDER BY retrieved_at DESC LIMIT 1"
    with database.read() as connection:
        return connection.execute(query, params).fetchone()


def _chunk_text(filing: dict[str, Any] | None, chunk: dict[str, Any]) -> str:
    if filing is None:
        raise ValueError("filing chunk references unknown filing")
    text = Path(str(filing["local_text_path"])).read_text(encoding="utf-8")
    start = int(chunk["char_start"])
    end = int(chunk["char_end"])
    if start < 0 or end < start or end > len(text):
        raise ValueError("filing chunk offsets are outside extracted text")
    return text[start:end]


def _count(database: SQLiteDatabase, table: str) -> int:
    with database.read() as connection:
        return int(connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])


def _clear(database: SQLiteDatabase, table: str) -> int:
    with database.transaction() as connection:
        count = int(connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
        connection.execute(f'DELETE FROM "{table}"')
        return count


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _iso(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("datetime must be timezone-aware")
    return value.astimezone(UTC).isoformat()


def _datetime(value: object) -> datetime:
    parsed = datetime.fromisoformat(str(value))
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:24]
    return f"{prefix}_{digest}"


def _optional(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _text(name: str, value: str) -> str:
    text = value.strip()
    if not text:
        raise ValueError(f"{name} is required")
    return text
