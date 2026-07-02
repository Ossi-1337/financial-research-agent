from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Lock
from uuid import uuid4

from financial_research_agent.llm import ChatMessage, MessageRole


@dataclass(frozen=True, slots=True)
class ChatSessionMessage:
    id: str
    role: MessageRole
    content: str
    created_at: datetime
    provider: str | None = None
    model: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", MessageRole(self.role))
        if self.role not in {MessageRole.USER, MessageRole.ASSISTANT}:
            raise ValueError("chat session messages must be user or assistant messages")
        object.__setattr__(self, "id", _require_text("id", self.id))
        object.__setattr__(self, "content", _message_content(self.role, self.content))
        object.__setattr__(self, "provider", _optional_text(self.provider))
        object.__setattr__(self, "model", _optional_text(self.model))
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "role": self.role.value,
            "content": self.content,
            "created_at": self.created_at.isoformat(),
            "provider": self.provider,
            "model": self.model,
        }

    def to_provider_message(self) -> ChatMessage:
        return ChatMessage(role=self.role, content=self.content)


@dataclass(frozen=True, slots=True)
class ChatSession:
    id: str
    created_at: datetime
    messages: tuple[ChatSessionMessage, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_text("id", self.id))
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")
        object.__setattr__(self, "messages", _message_tuple(self.messages))

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "messages": [message.to_dict() for message in self.messages],
        }

    def to_provider_messages(self) -> tuple[ChatMessage, ...]:
        return tuple(message.to_provider_message() for message in self.messages)


class ChatSessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, ChatSession] = {}
        self._lock = Lock()

    def create(self) -> ChatSession:
        session = ChatSession(id=_new_id("session"), created_at=_now())
        with self._lock:
            self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> ChatSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def append_exchange(
        self,
        *,
        session_id: str,
        user_content: str,
        assistant_content: str,
        provider: str,
        model: str,
    ) -> ChatSession:
        with self._lock:
            session = self._sessions[_require_text("session_id", session_id)]
            created_at = _now()
            updated = ChatSession(
                id=session.id,
                created_at=session.created_at,
                messages=(
                    *session.messages,
                    ChatSessionMessage(
                        id=_new_id("message"),
                        role=MessageRole.USER,
                        content=user_content,
                        created_at=created_at,
                    ),
                    ChatSessionMessage(
                        id=_new_id("message"),
                        role=MessageRole.ASSISTANT,
                        content=assistant_content,
                        created_at=_now(),
                        provider=provider,
                        model=model,
                    ),
                ),
            )
            self._sessions[session.id] = updated
            return updated


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
