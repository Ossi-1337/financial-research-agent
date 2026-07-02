from __future__ import annotations

import asyncio
from datetime import datetime

from financial_research_agent.llm import ToolCall
from financial_research_agent.tools import (
    ToolContext,
    ToolErrorCode,
    ToolResultStatus,
    create_default_tool_registry,
)


def test_current_utc_datetime_tool_returns_iso_utc_timestamp() -> None:
    registry = create_default_tool_registry()
    result = asyncio.run(registry.execute(ToolCall(id="call_time", name="current_utc_datetime")))

    timestamp = result.data["utc_datetime"]

    assert result.status == ToolResultStatus.SUCCEEDED
    assert result.tool_call_id == "call_time"
    assert isinstance(timestamp, str)
    assert timestamp.endswith("Z")
    datetime.fromisoformat(timestamp.replace("Z", "+00:00"))


def test_calculate_ratio_tool_succeeds_and_handles_division_by_zero() -> None:
    registry = create_default_tool_registry()

    success = asyncio.run(
        registry.execute(
            ToolCall(
                id="call_ratio",
                name="calculate_ratio",
                arguments={"numerator": 5, "denominator": 2, "precision": 2},
            )
        )
    )
    failure = asyncio.run(
        registry.execute(
            ToolCall(
                id="call_zero",
                name="calculate_ratio",
                arguments={"numerator": 5, "denominator": 0},
            )
        )
    )

    assert success.status == ToolResultStatus.SUCCEEDED
    assert success.data["ratio"] == "2.50"
    assert failure.status == ToolResultStatus.FAILED
    assert failure.error_code == ToolErrorCode.DIVISION_BY_ZERO


def test_resolve_company_stub_returns_no_fake_company_data() -> None:
    registry = create_default_tool_registry()

    result = asyncio.run(
        registry.execute(
            ToolCall(
                id="call_company",
                name="resolve_company_stub",
                arguments={"query": "Novo Nordisk"},
            )
        )
    )

    assert result.status == ToolResultStatus.SUCCEEDED
    assert result.data["query"] == "Novo Nordisk"
    assert result.data["matches"] == ()
    assert result.data["real_entity_resolution_available"] is False
    assert result.warnings == ("Real company entity resolution is planned for Milestone 12.",)


def test_read_local_evidence_tool_reads_injected_mapping_and_reports_missing_items() -> None:
    registry = create_default_tool_registry()
    context = ToolContext(
        local_evidence={
            "ev_1": {
                "title": "Source title",
                "excerpt": "Evidence excerpt.",
            }
        }
    )

    found = asyncio.run(
        registry.execute(
            ToolCall(
                id="call_found",
                name="read_local_evidence",
                arguments={"evidence_id": "ev_1"},
            ),
            context,
        )
    )
    missing = asyncio.run(
        registry.execute(
            ToolCall(
                id="call_missing",
                name="read_local_evidence",
                arguments={"evidence_id": "ev_missing"},
            ),
            context,
        )
    )

    assert found.status == ToolResultStatus.SUCCEEDED
    assert found.data["evidence"]["title"] == "Source title"
    assert missing.status == ToolResultStatus.FAILED
    assert missing.error_code == ToolErrorCode.NOT_FOUND
