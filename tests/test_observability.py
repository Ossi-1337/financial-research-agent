from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime

import pytest

from financial_research_agent.observability import (
    RedactionPolicy,
    ReplayMode,
    TraceEventKind,
    build_debug_bundle,
    build_replay_plan,
    build_trace_from_orchestrator_run,
)
from financial_research_agent.orchestration import (
    AgentHandoff,
    HandoffConfidence,
    OrchestratedResearchRun,
    OrchestratorExecutionPolicy,
    OrchestratorHandoffStatus,
    OrchestratorPlanStep,
    OrchestratorRunStatus,
    OrchestratorStepKind,
)
from financial_research_agent.settings import Settings


def test_redaction_policy_removes_secrets_tokens_and_sensitive_paths(tmp_path) -> None:
    policy = RedactionPolicy(
        sensitive_values=("sk-test-secret", "alpha-secret"),
        sensitive_paths=(str(tmp_path),),
        text_preview_chars=80,
    )

    redacted = policy.redact(
        {
            "openai_api_key": "sk-test-secret",
            "authorization": "Bearer alpha-secret",
            "path": str(tmp_path / "data" / "filing.txt"),
            "note": "token=alpha-secret and sk-abcdefghij should not be visible",
            "usage": {"total_tokens": 17, "access_token": "alpha-secret"},
        }
    )

    dumped = str(redacted)
    assert redacted["openai_api_key"] == "[REDACTED]"
    assert "sk-test-secret" not in dumped
    assert "alpha-secret" not in dumped
    assert str(tmp_path) not in dumped
    assert "[LOCAL_PATH]" in dumped
    assert "[REDACTED_API_KEY]" in dumped
    assert redacted["usage"]["total_tokens"] == 17
    assert redacted["usage"]["access_token"] == "[REDACTED]"


def test_trace_contracts_are_immutable_and_build_step_events(tmp_path) -> None:
    run = _run(tmp_path)
    trace = build_trace_from_orchestrator_run(
        run,
        redaction_policy=RedactionPolicy(
            sensitive_values=("secret-key",), sensitive_paths=(str(tmp_path),)
        ),
    )

    first = trace.events[0]
    second = trace.events[1]

    assert trace.run_id == run.id
    assert first.kind == TraceEventKind.PROVIDER_CALL
    assert first.status == "failed"
    assert first.error_code == "provider_unavailable"
    assert "secret-key" not in str(first.to_dict())
    assert second.kind == TraceEventKind.AGENT_OUTPUT
    assert second.evidence_ids == ("fixture:evidence:1",)
    assert second.token_cost_estimate.total_tokens == 17
    with pytest.raises(FrozenInstanceError):
        first.title = "changed"  # type: ignore[misc]


def test_replay_plan_uses_stored_results_without_provider_calls(tmp_path) -> None:
    replay = build_replay_plan(
        _run(tmp_path),
        redaction_policy=RedactionPolicy(
            sensitive_values=("secret-key",), sensitive_paths=(str(tmp_path),)
        ),
    )

    assert replay.replayable is True
    assert replay.steps[0].mode == ReplayMode.STORED_RESULT
    assert replay.steps[0].reason.startswith("Stored handoff output")
    assert replay.steps[0].safe_output["error_code"] == "provider_unavailable"
    assert "secret-key" not in str(replay.to_dict())
    assert "does not call providers" in replay.warnings[0]


def test_debug_bundle_excludes_credentials_and_model_cache(tmp_path) -> None:
    settings = Settings.from_env(
        {
            "FRA_HOME": str(tmp_path / "home"),
            "FRA_OPENAI_API_KEY": "sk-test-secret",
            "FRA_ALPHA_VANTAGE_API_KEY": "alpha-secret",
        }
    )

    bundle = build_debug_bundle(
        _run(tmp_path),
        settings=settings,
        created_at=datetime(2026, 7, 6, tzinfo=UTC),
    ).to_dict()
    dumped = str(bundle)

    assert bundle["schema_version"] == 1
    assert "raw provider credentials" in bundle["excluded_items"]
    assert "model cache files" in bundle["excluded_items"]
    assert "sk-test-secret" not in dumped
    assert "alpha-secret" not in dumped
    assert str(tmp_path / "home") not in dumped


def _run(tmp_path) -> OrchestratedResearchRun:
    started = datetime(2026, 7, 6, 12, 0, tzinfo=UTC)
    completed = datetime(2026, 7, 6, 12, 0, 1, tzinfo=UTC)
    return OrchestratedResearchRun(
        id="orchestrator_run_trace_fixture",
        query="Inspect secret-key path",
        status=OrchestratorRunStatus.FAILED,
        created_at=started,
        updated_at=completed,
        execution_policy=OrchestratorExecutionPolicy.SEQUENTIAL_LOCAL_SAFE,
        plan=(
            OrchestratorPlanStep(
                id="resolve_company",
                kind=OrchestratorStepKind.COMPANY_RESOLUTION,
                title="Resolve company",
                required=True,
            ),
        ),
        handoffs=(
            AgentHandoff(
                id="handoff_1",
                step_id="resolve_company",
                kind=OrchestratorStepKind.COMPANY_RESOLUTION,
                status=OrchestratorHandoffStatus.FAILED,
                started_at=started,
                completed_at=completed,
                input_summary={"query": "secret-key"},
                output={"path": str(tmp_path / "cache" / "source.json")},
                confidence=HandoffConfidence.UNKNOWN,
                error_code="provider_unavailable",
                error_message="Authorization Bearer secret-key failed",
            ),
            AgentHandoff(
                id="handoff_2",
                step_id="financial_report_analysis",
                kind=OrchestratorStepKind.FINANCIAL_REPORT_ANALYSIS,
                status=OrchestratorHandoffStatus.SUCCEEDED,
                started_at=started,
                completed_at=completed,
                output={
                    "analysis": {
                        "summary": "stored deterministic output",
                        "usage": {
                            "input_tokens": 10,
                            "output_tokens": 7,
                            "total_tokens": 17,
                        },
                    }
                },
                evidence_ids=("fixture:evidence:1",),
                confidence=HandoffConfidence.HIGH,
            ),
        ),
    )
