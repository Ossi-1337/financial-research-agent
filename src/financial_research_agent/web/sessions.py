from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any, Self
from uuid import uuid4

from financial_research_agent.llm import ChatMessage, MessageRole
from financial_research_agent.reports import Citation, EvidenceSnippet
from financial_research_agent.settings import Settings

SESSION_STORE_VERSION = 1
DEFAULT_SESSION_LIST_LIMIT = 50


@dataclass(frozen=True, slots=True)
class ChatMention:
    id: str
    label: str
    company_id: str
    legal_name: str
    ticker: str | None = None
    cik: str | None = None
    source_provider: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_text("id", self.id))
        object.__setattr__(self, "label", _require_text("label", self.label))
        object.__setattr__(self, "company_id", _require_text("company_id", self.company_id))
        object.__setattr__(self, "legal_name", _require_text("legal_name", self.legal_name))
        object.__setattr__(self, "ticker", _optional_text(self.ticker))
        object.__setattr__(self, "cik", _optional_text(self.cik))
        object.__setattr__(self, "source_provider", _optional_text(self.source_provider))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Self:
        return cls(
            id=_payload_text(payload, "id"),
            label=_payload_text(payload, "label"),
            company_id=_payload_text(payload, "company_id"),
            legal_name=_payload_text(payload, "legal_name"),
            ticker=_payload_optional_text(payload, "ticker"),
            cik=_payload_optional_text(payload, "cik"),
            source_provider=_payload_optional_text(payload, "source_provider"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "label": self.label,
            "company_id": self.company_id,
            "legal_name": self.legal_name,
            "ticker": self.ticker,
            "cik": self.cik,
            "source_provider": self.source_provider,
        }


@dataclass(frozen=True, slots=True)
class ChatSessionMessage:
    id: str
    role: MessageRole
    content: str
    created_at: datetime
    provider: str | None = None
    model: str | None = None
    research_run_id: str | None = None
    mentions: tuple[ChatMention, ...] = ()
    citations: tuple[Citation, ...] = ()
    evidence_snippets: tuple[EvidenceSnippet, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", MessageRole(self.role))
        if self.role not in {MessageRole.USER, MessageRole.ASSISTANT}:
            raise ValueError("chat session messages must be user or assistant messages")
        object.__setattr__(self, "id", _require_text("id", self.id))
        object.__setattr__(self, "content", _message_content(self.role, self.content))
        object.__setattr__(self, "provider", _optional_text(self.provider))
        object.__setattr__(self, "model", _optional_text(self.model))
        object.__setattr__(self, "research_run_id", _optional_text(self.research_run_id))
        object.__setattr__(self, "mentions", _mention_tuple(self.mentions))
        object.__setattr__(self, "citations", _citation_tuple(self.citations))
        object.__setattr__(
            self,
            "evidence_snippets",
            _evidence_snippet_tuple(self.evidence_snippets),
        )
        object.__setattr__(self, "created_at", _aware_datetime("created_at", self.created_at))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Self:
        return cls(
            id=_payload_text(payload, "id"),
            role=MessageRole(_payload_text(payload, "role")),
            content=_payload_string(payload, "content"),
            created_at=_datetime_from_payload(payload, "created_at"),
            provider=_payload_optional_text(payload, "provider"),
            model=_payload_optional_text(payload, "model"),
            research_run_id=_payload_optional_text(payload, "research_run_id"),
            mentions=tuple(
                ChatMention.from_dict(_payload_mapping(item, "mention"))
                for item in _payload_list(payload, "mentions")
            ),
            citations=tuple(
                Citation.from_dict(_payload_mapping(item, "citation"))
                for item in _payload_list(payload, "citations")
            ),
            evidence_snippets=tuple(
                EvidenceSnippet.from_dict(_payload_mapping(item, "evidence_snippet"))
                for item in _payload_list(payload, "evidence_snippets")
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "role": self.role.value,
            "content": self.content,
            "created_at": self.created_at.isoformat(),
            "provider": self.provider,
            "model": self.model,
            "research_run_id": self.research_run_id,
            "mentions": [mention.to_dict() for mention in self.mentions],
            "citations": [citation.to_dict() for citation in self.citations],
            "evidence_snippets": [snippet.to_dict() for snippet in self.evidence_snippets],
        }

    def to_provider_message(self) -> ChatMessage:
        return ChatMessage(role=self.role, content=self.content)


@dataclass(frozen=True, slots=True)
class ChatSession:
    id: str
    created_at: datetime
    updated_at: datetime
    messages: tuple[ChatSessionMessage, ...] = ()
    summary: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_text("id", self.id))
        object.__setattr__(self, "created_at", _aware_datetime("created_at", self.created_at))
        object.__setattr__(self, "updated_at", _aware_datetime("updated_at", self.updated_at))
        object.__setattr__(self, "messages", _message_tuple(self.messages))
        object.__setattr__(self, "summary", _optional_text(self.summary))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Self:
        messages = payload.get("messages", ())
        if not isinstance(messages, list):
            raise ValueError("session messages must be a list")
        return cls(
            id=_payload_text(payload, "id"),
            created_at=_datetime_from_payload(payload, "created_at"),
            updated_at=_datetime_from_payload(payload, "updated_at"),
            messages=tuple(ChatSessionMessage.from_dict(item) for item in messages),
            summary=_payload_optional_text(payload, "summary"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "summary": self.summary,
            "messages": [message.to_dict() for message in self.messages],
        }

    def to_provider_messages(self) -> tuple[ChatMessage, ...]:
        return tuple(message.to_provider_message() for message in self.messages)

    def context_messages(
        self,
        *,
        recent_turns: int,
        summary_max_chars: int,
    ) -> tuple[ChatMessage, ...]:
        recent_messages = _recent_messages(self.messages, recent_turns)
        older_messages = self.messages[: len(self.messages) - len(recent_messages)]
        summary = summarize_messages(older_messages, max_chars=summary_max_chars)
        if summary is None:
            return tuple(message.to_provider_message() for message in recent_messages)
        return (
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=f"Earlier conversation summary (local deterministic): {summary}",
            ),
            *(message.to_provider_message() for message in recent_messages),
        )


class ChatSessionStore:
    def __init__(
        self,
        *,
        storage_path: Path | None = None,
        recent_turns: int = 6,
        summary_max_chars: int = 1200,
    ) -> None:
        if recent_turns <= 0:
            raise ValueError("recent_turns must be positive")
        if summary_max_chars <= 0:
            raise ValueError("summary_max_chars must be positive")
        self.storage_path = storage_path
        self.recent_turns = recent_turns
        self.summary_max_chars = summary_max_chars
        self._sessions: dict[str, ChatSession] = {}
        self._lock = Lock()
        self._load()

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        return cls(
            storage_path=settings.local_paths.data_dir / "chat_sessions.json",
            recent_turns=settings.chat.history_recent_turns,
            summary_max_chars=settings.chat.history_summary_max_chars,
        )

    def create(self) -> ChatSession:
        now = _now()
        session = ChatSession(id=_new_id("session"), created_at=now, updated_at=now)
        with self._lock:
            self._sessions[session.id] = session
            self._save()
        return session

    def list(self, *, limit: int = DEFAULT_SESSION_LIST_LIMIT) -> tuple[ChatSession, ...]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        with self._lock:
            return tuple(
                sorted(
                    self._sessions.values(),
                    key=lambda session: session.updated_at,
                    reverse=True,
                )
            )[:limit]

    def get(self, session_id: str) -> ChatSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def count(self) -> int:
        with self._lock:
            return len(self._sessions)

    def delete(self, session_id: str) -> bool:
        with self._lock:
            deleted = self._sessions.pop(_require_text("session_id", session_id), None) is not None
            if deleted:
                self._save()
            return deleted

    def clear(self) -> int:
        with self._lock:
            deleted = len(self._sessions)
            self._sessions.clear()
            self._save()
            return deleted

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
        citations: tuple[Citation, ...] = (),
        evidence_snippets: tuple[EvidenceSnippet, ...] = (),
    ) -> ChatSession:
        with self._lock:
            session = self._sessions[_require_text("session_id", session_id)]
            created_at = _now()
            messages = (
                *session.messages,
                ChatSessionMessage(
                    id=_new_id("message"),
                    role=MessageRole.USER,
                    content=user_content,
                    created_at=created_at,
                    research_run_id=research_run_id,
                    mentions=mentions,
                ),
                ChatSessionMessage(
                    id=_new_id("message"),
                    role=MessageRole.ASSISTANT,
                    content=assistant_content,
                    created_at=_now(),
                    provider=provider,
                    model=model,
                    research_run_id=research_run_id,
                    citations=citations,
                    evidence_snippets=evidence_snippets,
                ),
            )
            updated = ChatSession(
                id=session.id,
                created_at=session.created_at,
                updated_at=_now(),
                messages=messages,
                summary=summarize_messages(
                    _older_messages(messages, self.recent_turns),
                    max_chars=self.summary_max_chars,
                ),
            )
            self._sessions[session.id] = updated
            self._save()
            return updated

    def _load(self) -> None:
        if self.storage_path is None or not self.storage_path.exists():
            return
        try:
            payload = json.loads(self.storage_path.read_text(encoding="utf-8"))
            if payload.get("version") != SESSION_STORE_VERSION:
                raise ValueError("unsupported chat session store version")
            sessions = payload.get("sessions", ())
            if not isinstance(sessions, list):
                raise ValueError("stored sessions must be a list")
            loaded_sessions = (
                ChatSession.from_dict(_payload_mapping(item, "session")) for item in sessions
            )
            self._sessions = {session.id: session for session in loaded_sessions}
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"Could not load chat session store: {self.storage_path}") from exc

    def _save(self) -> None:
        if self.storage_path is None:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": SESSION_STORE_VERSION,
            "sessions": [session.to_dict() for session in self._list_unlocked()],
        }
        temp_path = self.storage_path.with_suffix(f"{self.storage_path.suffix}.tmp")
        temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temp_path.replace(self.storage_path)

    def _list_unlocked(self) -> tuple[ChatSession, ...]:
        return tuple(
            sorted(self._sessions.values(), key=lambda session: session.updated_at, reverse=True)
        )


def summarize_messages(
    messages: tuple[ChatSessionMessage, ...],
    *,
    max_chars: int,
) -> str | None:
    if not messages:
        return None
    parts = [f"{message.role.value}: {message.content.strip()}" for message in messages]
    summary = " | ".join(part for part in parts if part.strip())
    if summary == "":
        return None
    if len(summary) <= max_chars:
        return summary
    if max_chars <= 3:
        return summary[:max_chars]
    return summary[: max_chars - 3].rstrip() + "..."


def _older_messages(
    messages: tuple[ChatSessionMessage, ...],
    recent_turns: int,
) -> tuple[ChatSessionMessage, ...]:
    recent_messages = _recent_messages(messages, recent_turns)
    return messages[: len(messages) - len(recent_messages)]


def _recent_messages(
    messages: tuple[ChatSessionMessage, ...],
    recent_turns: int,
) -> tuple[ChatSessionMessage, ...]:
    if recent_turns <= 0:
        raise ValueError("recent_turns must be positive")
    max_recent_messages = recent_turns * 2
    return messages[-max_recent_messages:]


def _now() -> datetime:
    return datetime.now(tz=UTC)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def _message_tuple(messages: tuple[ChatSessionMessage, ...]) -> tuple[ChatSessionMessage, ...]:
    result = tuple(messages)
    for index, message in enumerate(result):
        if not isinstance(message, ChatSessionMessage):
            raise ValueError(f"messages[{index}] must be a ChatSessionMessage")
    return result


def _mention_tuple(mentions: tuple[ChatMention, ...]) -> tuple[ChatMention, ...]:
    result = tuple(mentions)
    for index, mention in enumerate(result):
        if not isinstance(mention, ChatMention):
            raise ValueError(f"mentions[{index}] must be a ChatMention")
    return result


def _citation_tuple(citations: tuple[Citation, ...]) -> tuple[Citation, ...]:
    result = tuple(citations)
    for index, citation in enumerate(result):
        if not isinstance(citation, Citation):
            raise ValueError(f"citations[{index}] must be a Citation")
    return result


def _evidence_snippet_tuple(snippets: tuple[EvidenceSnippet, ...]) -> tuple[EvidenceSnippet, ...]:
    result = tuple(snippets)
    for index, snippet in enumerate(result):
        if not isinstance(snippet, EvidenceSnippet):
            raise ValueError(f"evidence_snippets[{index}] must be an EvidenceSnippet")
    return result


def _require_text(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    text = value.strip()
    if text == "":
        raise ValueError(f"{name} is required")
    return text


def _require_content(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("content must be a string")
    text = value.strip()
    if text == "":
        raise ValueError("content is required")
    return text


def _message_content(role: MessageRole, value: str) -> str:
    if role == MessageRole.USER:
        return _require_content(value)
    if not isinstance(value, str):
        raise ValueError("content must be a string")
    return value


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    return text or None


def _aware_datetime(name: str, value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError(f"{name} must be a datetime")
    if value.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value


def _datetime_from_payload(payload: dict[str, Any], name: str) -> datetime:
    try:
        value = datetime.fromisoformat(_payload_text(payload, name))
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO datetime") from exc
    return _aware_datetime(name, value)


def _payload_text(payload: dict[str, Any], name: str) -> str:
    return _require_text(name, _payload_string(payload, name))


def _payload_string(payload: dict[str, Any], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


def _payload_optional_text(payload: dict[str, Any], name: str) -> str | None:
    value = payload.get(name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return _optional_text(value)


def _payload_list(payload: dict[str, Any], name: str) -> list[Any]:
    value = payload.get(name, [])
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def _payload_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value
