from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime

import pytest

from financial_research_agent.orchestration import (
    AgentHandoff,
    OrchestratorHandoffStatus,
    OrchestratorStepKind,
)
from financial_research_agent.synthesis import (
    ConfidenceLabel,
    EvidenceCoverage,
    ScenarioDirection,
    SynthesisAgent,
    SynthesisPoint,
    SynthesisReportStatus,
    SynthesisScenario,
    SynthesisSection,
)

NOW = datetime(2026, 7, 6, 12, tzinfo=UTC)


def test_synthesis_contracts_are_immutable_and_require_conditional_scenarios() -> None:
    point = SynthesisPoint(
        id="point_1",
        section=SynthesisSection.STRENGTHS,
        title="Revenue evidence",
        summary="Revenue improved in the stored statement fixture.",
        confidence=ConfidenceLabel.HIGH,
        evidence_ids=("evidence:revenue",),
        source_handoff_ids=("handoff_report",),
    )

    with pytest.raises(FrozenInstanceError):
        point.title = "changed"
    with pytest.raises(ValueError, match="conditional language"):
        SynthesisScenario(
            id="scenario_bad",
            direction=ScenarioDirection.UPSIDE,
            title="Bad scenario",
            condition="Revenue improves.",
            potential_development="The company improves.",
            confidence=ConfidenceLabel.LOW,
            source_handoff_ids=("handoff_report",),
        )


def test_synthesis_agent_combines_specialist_outputs_with_evidence_and_scenarios() -> None:
    report = SynthesisAgent(now=NOW).synthesize(
        query="Apple financial situation",
        selected_company={"legal_name": "TEST TOOL OUTPUT APPLE INC."},
        selected_security={"ticker": "AAPL"},
        handoffs=(
            _financial_report_handoff(),
            _stock_handoff(),
            _context_handoff(),
        ),
    )
    payload = report.to_dict()

    assert report.status == SynthesisReportStatus.PARTIAL
    assert report.company_name == "TEST TOOL OUTPUT APPLE INC."
    assert report.security_symbol == "AAPL"
    assert report.evidence_coverage in {EvidenceCoverage.MODERATE, EvidenceCoverage.STRONG}
    assert "evidence:revenue" in report.evidence_ids
    assert payload["sections"]["current_situation"]
    assert payload["sections"]["strengths"]
    assert payload["sections"]["opportunities"]
    assert payload["sections"]["risks"]
    assert payload["scenarios"]["upside"]["direction"] == "upside"
    assert payload["scenarios"]["downside"]["direction"] == "downside"
    assert "If " in payload["scenarios"]["upside"]["condition"]
    assert "If " in payload["scenarios"]["downside"]["condition"]
    assert "does not provide buy, sell, hold" in payload["no_recommendation_notice"]


def test_synthesis_agent_remains_useful_when_inputs_are_missing() -> None:
    report = SynthesisAgent(now=NOW).synthesize(
        query="Unknown company financial situation",
        handoffs=(),
    )
    payload = report.to_dict()

    assert report.status == SynthesisReportStatus.NO_DATA
    assert report.evidence_coverage == EvidenceCoverage.NONE
    assert report.overall_confidence == ConfidenceLabel.UNKNOWN
    assert payload["sections"]["current_situation"]
    assert len(payload["sections"]["unknowns"]) >= 3
    assert payload["scenarios"]["upside"]["limitations"]
    assert payload["scenarios"]["downside"]["limitations"]


def _financial_report_handoff() -> AgentHandoff:
    return _handoff(
        id="handoff_report",
        kind=OrchestratorStepKind.FINANCIAL_REPORT_ANALYSIS,
        status=OrchestratorHandoffStatus.PARTIAL,
        output={
            "analysis": {
                "findings": [
                    {
                        "section": "revenue",
                        "title": "Revenue growth",
                        "summary": "Stored statements show revenue growth in the fixture.",
                        "confidence": "high",
                        "evidence_ids": ["evidence:revenue"],
                    },
                    {
                        "section": "risks",
                        "title": "Risk language",
                        "summary": "Stored filing chunks include risk language.",
                        "confidence": "medium",
                        "evidence_ids": ["evidence:risk"],
                    },
                ],
                "limitations": ["Debt data is incomplete."],
            }
        },
        limitations=("Debt data is incomplete.",),
    )


def _stock_handoff() -> AgentHandoff:
    return _handoff(
        id="handoff_stock",
        kind=OrchestratorStepKind.STOCK_PRICE_ANALYSIS,
        status=OrchestratorHandoffStatus.SUCCEEDED,
        output={
            "analysis": {
                "findings": [
                    {
                        "section": "trend",
                        "title": "Positive trend",
                        "summary": "Stored closing prices trend upward in the fixture.",
                        "confidence": "medium",
                        "metric_names": ["close_return_pct", "moving_average_20"],
                        "trend": "up",
                    }
                ]
            }
        },
    )


def _context_handoff() -> AgentHandoff:
    return _handoff(
        id="handoff_context",
        kind=OrchestratorStepKind.CONTEXT_ANALYSIS,
        status=OrchestratorHandoffStatus.SUCCEEDED,
        output={
            "analysis": {
                "findings": [
                    {
                        "scope": "macro",
                        "title": "Demand tailwind",
                        "summary": "Source-linked fixture context describes demand growth.",
                        "confidence": "medium",
                        "source_item_ids": ["source:macro"],
                    }
                ]
            }
        },
    )


def _handoff(
    *,
    id: str,
    kind: OrchestratorStepKind,
    status: OrchestratorHandoffStatus,
    output: dict[str, object],
    limitations: tuple[str, ...] = (),
) -> AgentHandoff:
    return AgentHandoff(
        id=id,
        step_id=kind.value,
        kind=kind,
        status=status,
        started_at=NOW,
        completed_at=NOW,
        output=output,
        limitations=limitations,
        confidence=ConfidenceLabel.MEDIUM.value,
    )
