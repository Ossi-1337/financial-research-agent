from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from pathlib import Path

import pytest

from financial_research_agent.orchestration import (
    AgentHandoff,
    HandoffConfidence,
    OrchestratedResearchRun,
    OrchestratorExecutionPolicy,
    OrchestratorHandoffStatus,
    OrchestratorRunStatus,
    OrchestratorStepKind,
    default_orchestrator_plan,
)
from financial_research_agent.persistence import create_persistence
from financial_research_agent.settings import Settings
from financial_research_agent.synthesis import (
    NARRATIVE_SECTION_ORDER,
    NarrativeParagraph,
    NarrativePresentation,
    NarrativePresentationSection,
    NarrativeSection,
    synthesis_sha256,
)

NOW = datetime(2026, 7, 30, 12, tzinfo=UTC)
REPORT = {
    "id": "report_test",
    "query": "TEST FIXTURE financial research",
    "status": "partial",
    "created_at": NOW.isoformat(),
    "company_name": "TEST COMPANY",
    "security_symbol": "TEST",
    "sections": {
        "current_situation": [
            {
                "id": "point_current",
                "title": "Current situation",
                "summary": "Stored evidence indicates a partial financial picture.",
                "confidence": "medium",
                "evidence_ids": ["evidence:one"],
                "source_handoff_ids": ["handoff_financial"],
                "limitations": [],
            }
        ],
        "strengths": [],
        "weaknesses": [],
        "opportunities": [],
        "risks": [],
        "unknowns": [],
    },
    "scenarios": {},
    "overall_confidence": "medium",
    "evidence_coverage": "limited",
    "evidence_coverage_ratio": 0.5,
    "evidence_ids": ["evidence:one"],
    "warnings": ["TEST stale warning."],
    "limitations": ["TEST partial evidence."],
    "no_recommendation_notice": "Research only; no buy, sell, or hold recommendation.",
}


def test_narrative_contracts_enforce_immutability_order_and_bounds() -> None:
    paragraph = NarrativeParagraph(
        text="The validated report remains partial.",
        source_point_ids=("point_current",),
        evidence_ids=("evidence:one",),
        source_markers=("[S1]",),
    )
    presentation = _presentation()

    with pytest.raises(FrozenInstanceError):
        paragraph.text = "changed"
    with pytest.raises(ValueError, match="canonical section order"):
        NarrativePresentation(
            **{
                **presentation.to_dict(),
                "sections": tuple(reversed(presentation.sections)),
                "created_at": NOW,
            }
        )
    with pytest.raises(ValueError, match="at most 700"):
        NarrativeParagraph(
            text="x" * 701,
            source_point_ids=("point_current",),
        )


def test_sqlite_narrative_store_reads_legacy_matching_artifact(tmp_path: Path) -> None:
    settings = Settings.from_env(
        {
            "FRA_HOME": str(tmp_path),
            "FRA_STORAGE_PROVIDER": "sqlite",
        }
    )
    persistence = create_persistence(settings)
    run = _run()
    presentation = _presentation()
    persistence.orchestrator_runs.save(run)
    assert persistence.database is not None
    with persistence.database.transaction() as connection:
        connection.execute(
            """
            INSERT INTO narrative_presentations(
                id, run_id, report_id, synthesis_sha256, prompt_id, prompt_version,
                provider, model, created_at, payload_version, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                presentation.id,
                presentation.run_id,
                presentation.report_id,
                presentation.synthesis_sha256,
                presentation.prompt_id,
                presentation.prompt_version,
                presentation.provider,
                presentation.model,
                presentation.created_at.isoformat(),
                presentation.schema_version,
                json.dumps(presentation.to_dict(), separators=(",", ":"), sort_keys=True),
            ),
        )

    matching = persistence.narrative_presentations.matching(
        run_id=run.id,
        synthesis_sha256=presentation.synthesis_sha256,
        prompt_id=presentation.prompt_id,
        prompt_version=presentation.prompt_version,
        provider=presentation.provider,
        model=presentation.model,
    )

    assert matching == presentation


def _presentation() -> NarrativePresentation:
    return NarrativePresentation(
        id="narrative_test",
        run_id="run_test",
        report_id="report_test",
        synthesis_sha256=synthesis_sha256(REPORT),
        provider="scripted",
        model="scripted-model",
        created_at=NOW,
        sections=tuple(
            NarrativePresentationSection(
                section=section,
                paragraphs=(
                    (
                        NarrativeParagraph(
                            text="The validated evidence describes a partial current situation.",
                            source_point_ids=("point_current",),
                            evidence_ids=("evidence:one",),
                            source_markers=("[S1]",),
                        ),
                    )
                    if section == NarrativeSection.CURRENT_SITUATION
                    else ()
                ),
            )
            for section in NARRATIVE_SECTION_ORDER
        ),
        warnings=("TEST stale warning.",),
        limitations=("TEST partial evidence.",),
        no_recommendation_notice="Research only; no buy, sell, or hold recommendation.",
    )


def _run() -> OrchestratedResearchRun:
    synthesis = AgentHandoff(
        id="handoff_synthesis",
        step_id="synthesis",
        kind=OrchestratorStepKind.SYNTHESIS,
        status=OrchestratorHandoffStatus.PARTIAL,
        started_at=NOW,
        completed_at=NOW,
        output={"report": REPORT},
        evidence_ids=("evidence:one",),
        confidence=HandoffConfidence.MEDIUM,
    )
    return OrchestratedResearchRun(
        id="run_test",
        query="TEST FIXTURE financial research",
        status=OrchestratorRunStatus.PARTIAL,
        created_at=NOW,
        updated_at=NOW,
        execution_policy=OrchestratorExecutionPolicy.DISTRIBUTED_A2A,
        plan=default_orchestrator_plan(),
        agent_provider="scripted",
        agent_model="scripted-model",
        handoffs=(synthesis,),
        selected_company={"id": "company_test", "legal_name": "TEST COMPANY"},
        selected_security={"id": "security_test", "ticker": "TEST"},
    )
