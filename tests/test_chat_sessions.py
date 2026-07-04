from __future__ import annotations

import json
from pathlib import Path

import pytest

from financial_research_agent.llm import MessageRole
from financial_research_agent.web.sessions import ChatMention, ChatSessionStore, summarize_messages


def test_session_store_persists_sessions_and_messages(tmp_path: Path) -> None:
    storage_path = tmp_path / "chat_sessions.json"
    store = ChatSessionStore(storage_path=storage_path, recent_turns=2, summary_max_chars=200)
    session = store.create()

    updated = store.append_exchange(
        session_id=session.id,
        user_content="What about Novo Nordisk?",
        assistant_content="offline-test response",
        provider="offline-test",
        model="offline-test",
        research_run_id="research_run_1",
        mentions=(
            ChatMention(
                id="sec:company:320193",
                label="AAPL",
                company_id="sec:company:320193",
                legal_name="TEST TOOL OUTPUT APPLE INC.",
                ticker="AAPL",
                cik="320193",
                source_provider="sec",
            ),
        ),
    )
    reloaded = ChatSessionStore(storage_path=storage_path, recent_turns=2, summary_max_chars=200)
    loaded = reloaded.get(session.id)

    assert loaded is not None
    assert loaded.id == updated.id
    assert loaded.updated_at == updated.updated_at
    assert len(loaded.messages) == 2
    assert loaded.messages[0].role == MessageRole.USER
    assert loaded.messages[0].research_run_id == "research_run_1"
    assert loaded.messages[0].mentions[0].label == "AAPL"
    assert loaded.messages[0].mentions[0].cik == "320193"
    assert loaded.messages[1].provider == "offline-test"


def test_session_store_loads_old_messages_without_mentions(tmp_path: Path) -> None:
    storage_path = tmp_path / "chat_sessions.json"
    storage_path.write_text(
        json.dumps(
            {
                "version": 1,
                "sessions": [
                    {
                        "id": "session_old",
                        "created_at": "2026-07-04T12:00:00+00:00",
                        "updated_at": "2026-07-04T12:00:00+00:00",
                        "summary": None,
                        "messages": [
                            {
                                "id": "message_old",
                                "role": "user",
                                "content": "Hello",
                                "created_at": "2026-07-04T12:00:00+00:00",
                                "provider": None,
                                "model": None,
                                "research_run_id": None,
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    store = ChatSessionStore(storage_path=storage_path)
    session = store.get("session_old")

    assert session is not None
    assert session.messages[0].mentions == ()


def test_session_list_sorts_by_updated_at_and_clear_removes_persisted_sessions(
    tmp_path: Path,
) -> None:
    storage_path = tmp_path / "chat_sessions.json"
    store = ChatSessionStore(storage_path=storage_path)
    first = store.create()
    second = store.create()

    listed = store.list()
    deleted = store.clear()
    reloaded = ChatSessionStore(storage_path=storage_path)

    assert store.count() == 0
    assert listed[0].id == second.id
    assert listed[1].id == first.id
    assert deleted == 2
    assert reloaded.list() == ()


def test_delete_removes_one_session(tmp_path: Path) -> None:
    store = ChatSessionStore(storage_path=tmp_path / "chat_sessions.json")
    first = store.create()
    second = store.create()

    assert store.delete(first.id) is True
    assert store.delete("missing") is False
    assert store.get(first.id) is None
    assert store.get(second.id) is not None


def test_context_messages_include_summary_and_recent_turn_window(tmp_path: Path) -> None:
    store = ChatSessionStore(
        storage_path=tmp_path / "chat_sessions.json",
        recent_turns=1,
        summary_max_chars=120,
    )
    session = store.create()
    for index in range(3):
        session = store.append_exchange(
            session_id=session.id,
            user_content=f"user turn {index}",
            assistant_content=f"assistant turn {index}",
            provider="offline-test",
            model="offline-test",
        )

    context = session.context_messages(recent_turns=1, summary_max_chars=120)

    assert context[0].role == MessageRole.SYSTEM
    assert "Earlier conversation summary" in context[0].content
    assert "user turn 0" in context[0].content
    assert [message.content for message in context[1:]] == ["user turn 2", "assistant turn 2"]


def test_summary_truncates_to_configured_length(tmp_path: Path) -> None:
    store = ChatSessionStore(storage_path=tmp_path / "chat_sessions.json", summary_max_chars=32)
    session = store.create()
    message = "x" * 80
    session = store.append_exchange(
        session_id=session.id,
        user_content=message,
        assistant_content="answer",
        provider="offline-test",
        model="offline-test",
    )

    summary = summarize_messages(session.messages, max_chars=32)

    assert summary is not None
    assert len(summary) <= 32
    assert summary.endswith("...")


def test_summary_handles_tiny_max_length(tmp_path: Path) -> None:
    store = ChatSessionStore(storage_path=tmp_path / "chat_sessions.json", summary_max_chars=1)
    session = store.create()
    session = store.append_exchange(
        session_id=session.id,
        user_content="abcdef",
        assistant_content="answer",
        provider="offline-test",
        model="offline-test",
    )

    summary = summarize_messages(session.messages, max_chars=1)

    assert summary == "u"


def test_session_store_rejects_unsupported_storage_version(tmp_path: Path) -> None:
    storage_path = tmp_path / "chat_sessions.json"
    storage_path.write_text(
        json.dumps({"version": 999, "sessions": []}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Could not load chat session store"):
        ChatSessionStore(storage_path=storage_path)
