from __future__ import annotations

import asyncio
import json
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from financial_research_agent.llm import ChatMessage, ChatResponse, MessageRole
from financial_research_agent.observability import RedactionPolicy
from financial_research_agent.orchestration import (
    AgentHandoff,
    HandoffConfidence,
    OrchestratedResearchRun,
    OrchestratorExecutionPolicy,
    OrchestratorHandoffStatus,
    OrchestratorRunStatus,
    OrchestratorRunStore,
    OrchestratorStepKind,
    default_orchestrator_plan,
)
from financial_research_agent.report_exports import ReportExportService, ReportExportStore
from financial_research_agent.scenarios import (
    ScenarioCatalog,
    ScenarioCheckStatus,
    ScenarioDefinition,
    ScenarioError,
    ScenarioErrorCode,
    ScenarioExecutionStatus,
    ScenarioRunner,
    create_default_scenario_catalog,
    load_context_snapshot,
)
from financial_research_agent.settings import Settings

NOW = datetime(2026, 7, 21, 12, tzinfo=UTC)


def test_default_scenario_contract_is_immutable_and_has_locked_novo_profile() -> None:
    scenario = create_default_scenario_catalog().get("novo-nordisk")

    assert scenario.expected_cik == "0000353278"
    assert scenario.preferred_ticker == "NVO"
    assert scenario.preferred_exchange == "NYSE"
    assert dict(scenario.filing_form_limits) == {"20-F": 1, "6-K": 1}
    assert scenario.benchmark_symbol == "SPY"
    with pytest.raises(FrozenInstanceError):
        scenario.preferred_ticker = "NONOF"
    with pytest.raises(TypeError):
        scenario.filing_form_limits["20-F"] = 2


def test_scenario_catalog_rejects_duplicates_and_unknown_ids() -> None:
    scenario = create_default_scenario_catalog().get("novo-nordisk")

    with pytest.raises(ValueError, match="unique"):
        ScenarioCatalog((scenario, scenario))
    with pytest.raises(ScenarioError) as error:
        ScenarioCatalog().get("missing")
    assert error.value.code == ScenarioErrorCode.UNKNOWN_SCENARIO


def test_scenario_definition_rejects_unversioned_or_unsafe_resources() -> None:
    payload = create_default_scenario_catalog().get("novo-nordisk").to_dict()

    with pytest.raises(ValueError, match="semantic"):
        ScenarioDefinition(**{**payload, "version": "latest"})
    with pytest.raises(ValueError, match="safe JSON"):
        ScenarioDefinition(**{**payload, "context_resource": "../outside.json"})


def test_tracked_context_snapshot_has_real_dated_company_and_macro_sector_sources() -> None:
    sources = load_context_snapshot(
        "novo_nordisk_context.v1.json",
        scenario_id="novo-nordisk",
        now=NOW,
    )

    assert {item.scope.value for item in sources} >= {"company"}
    assert {item.scope.value for item in sources}.intersection({"macro", "sector"})
    assert all(item.source_url.startswith("https://") for item in sources)
    assert all("fixture" not in item.source_url for item in sources)
    assert all(item.published_at is not None and item.published_at <= NOW for item in sources)


def test_context_snapshot_rejects_future_and_local_fixture_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from financial_research_agent.scenarios import context as context_module

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    payload = {
        "schema_version": 1,
        "scenario_id": "novo-nordisk",
        "source_items": [
            _context_payload(
                "company",
                source_url="https://localhost/company",
                published_at=(NOW + timedelta(days=1)).isoformat(),
            ),
            _context_payload("macro"),
        ],
    }
    (data_dir / "invalid.json").write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(context_module, "files", lambda _package: tmp_path)

    with pytest.raises(ScenarioError) as error:
        load_context_snapshot("invalid.json", scenario_id="novo-nordisk", now=NOW)
    assert error.value.code == ScenarioErrorCode.INVALID_CONTEXT_SNAPSHOT


def test_prepare_requires_live_credentials_but_provider_free_stored_run_does_not(
    tmp_path: Path,
) -> None:
    runner = _runner(tmp_path, settings=Settings.from_env({"FRA_HOME": str(tmp_path)}))

    with pytest.raises(ScenarioError) as error:
        runner.prepare("novo-nordisk")
    assert error.value.code == ScenarioErrorCode.MISSING_MARKET_DATA_CREDENTIALS

    prepared = runner.prepare("novo-nordisk", refresh=False)
    assert prepared.refresh is False
    assert prepared.scenario_id == "novo-nordisk"
    assert prepared.benchmark_symbol == "SPY"
    assert dict(prepared.filing_form_limits) == {"20-F": 1, "6-K": 1}
    assert prepared.context_source_items


def test_scenario_finalize_checks_real_data_shape_and_creates_three_exports(
    tmp_path: Path,
) -> None:
    result = asyncio.run(_runner(tmp_path).finalize(_scenario_run()))

    assert result.status == ScenarioExecutionStatus.COMPLETE
    assert result.export is not None
    assert len(result.export.artifacts) == 3
    assert {check.status for check in result.checks} == {ScenarioCheckStatus.PASSED}
    assert next(check for check in result.checks if check.id == "market_and_benchmark").details == {
        "primary_bar_count": 2,
        "benchmark_bar_count": 2,
    }
    assert next(check for check in result.checks if check.id == "required_filings").details == {
        "form_counts": {"20-F": 1, "6-K": 1}
    }


def test_scenario_checks_reject_empty_benchmark_and_wrong_filing_count(tmp_path: Path) -> None:
    run = _scenario_run(empty_benchmark=True, duplicate_20f=True)

    result = asyncio.run(_runner(tmp_path).finalize(run))
    checks = {check.id: check for check in result.checks}

    assert result.status == ScenarioExecutionStatus.FAILED
    assert checks["market_and_benchmark"].status == ScenarioCheckStatus.FAILED
    assert checks["required_filings"].status == ScenarioCheckStatus.FAILED


def test_optional_local_qa_keeps_only_resolved_source_markers(tmp_path: Path) -> None:
    runner = _runner(tmp_path, chat_provider=FixtureChatProvider())

    result = asyncio.run(runner.finalize(_scenario_run(), with_local_qa=True))

    assert result.status == ScenarioExecutionStatus.COMPLETE
    assert result.local_qa is not None
    assert result.local_qa.source_markers == ("[S1]",)
    assert result.local_qa.status == ScenarioCheckStatus.WARNING
    assert result.local_qa.to_dict()["generation_method"] == "llm_source_bounded"


def test_optional_local_qa_never_blocks_deterministic_completion(tmp_path: Path) -> None:
    result = asyncio.run(_runner(tmp_path).finalize(_scenario_run(), with_local_qa=True))
    local_qa_check = next(check for check in result.checks if check.id == "local_qa")

    assert result.status == ScenarioExecutionStatus.COMPLETE
    assert result.local_qa is None
    assert local_qa_check.status == ScenarioCheckStatus.WARNING


def test_scenario_metadata_persists_and_old_runs_remain_compatible(tmp_path: Path) -> None:
    storage_path = tmp_path / "runs.json"
    store = OrchestratorRunStore(storage_path=storage_path)
    stored = store.save(_scenario_run())
    old_payload = stored.to_dict()
    old_payload.pop("scenario_id")
    old_payload.pop("scenario_version")

    assert store.get(stored.id).scenario_id == "novo-nordisk"
    storage_path.write_text(
        json.dumps({"version": 1, "runs": [old_payload]}),
        encoding="utf-8",
    )
    reloaded_old = OrchestratorRunStore(storage_path=storage_path).get(stored.id)
    assert reloaded_old is not None
    assert reloaded_old.scenario_id is None
    assert reloaded_old.scenario_version is None


class NeverRunOrchestrator:
    async def run(self, _request):
        raise AssertionError("orchestrator should not run in finalize tests")


class FixtureChatProvider:
    async def chat(self, _request):
        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT,
                content="TEST FIXTURE summary [S1]. Unknown marker [S999].",
            ),
            provider="offline-test",
            model="fixture-model",
        )


def _runner(
    tmp_path: Path,
    *,
    settings: Settings | None = None,
    chat_provider=None,
) -> ScenarioRunner:
    return ScenarioRunner(
        settings=settings or Settings.from_env({"FRA_HOME": str(tmp_path)}),
        catalog=create_default_scenario_catalog(),
        orchestrator=NeverRunOrchestrator(),
        export_service=ReportExportService(
            store=ReportExportStore(root=tmp_path / "exports"),
            redaction_policy=RedactionPolicy(),
            now=lambda: NOW,
            id_factory=lambda: "export_scenario_fixture",
        ),
        chat_provider=chat_provider,
        chat_model="fixture-model",
    )


def _scenario_run(
    *,
    empty_benchmark: bool = False,
    duplicate_20f: bool = False,
) -> OrchestratedResearchRun:
    filing_rows = [{"form_type": "20-F"}, {"form_type": "6-K"}]
    if duplicate_20f:
        filing_rows.append({"form_type": "20-F"})
    handoffs = (
        _handoff(
            "market",
            OrchestratorStepKind.MARKET_DATA_REFRESH,
            {
                "history": {"bars": [{"close": "100"}, {"close": "101"}]},
                "benchmark_history": {
                    "bars": [] if empty_benchmark else [{"close": "200"}, {"close": "202"}]
                },
            },
        ),
        _handoff(
            "statements",
            OrchestratorStepKind.FINANCIAL_STATEMENT_REFRESH,
            {
                "statements": {
                    "statements": [
                        _statement_payload(2025),
                        _statement_payload(2024),
                    ]
                }
            },
        ),
        _handoff(
            "filings",
            OrchestratorStepKind.FILING_REFRESH,
            {"filings": {"filings": filing_rows}},
        ),
        _handoff(
            "financial",
            OrchestratorStepKind.FINANCIAL_REPORT_ANALYSIS,
            {
                "analysis": {
                    "citations": [
                        {
                            "id": "citation:novo:revenue",
                            "evidence_id": "evidence:novo:revenue",
                            "source_url": "https://example.test/novo-20f",
                            "retrieved_at": NOW.isoformat(),
                            "section": "Revenue",
                            "quote": "TEST FIXTURE IFRS revenue evidence.",
                            "metadata": {
                                "source_name": "TEST FIXTURE filing",
                                "filing_date": "2026-02-04",
                            },
                        }
                    ],
                    "evidence": [],
                }
            },
            evidence_ids=("evidence:novo:revenue",),
        ),
        _handoff(
            "stock",
            OrchestratorStepKind.STOCK_PRICE_ANALYSIS,
            {
                "analysis": {
                    "chart_series": [
                        {
                            "symbol": "NVO",
                            "points": [
                                {"close": "100", "adjusted_close": "100"},
                                {"close": "105", "adjusted_close": "105"},
                            ],
                        },
                        {
                            "symbol": "SPY",
                            "points": [
                                {"close": "200", "adjusted_close": "200"},
                                {"close": "202", "adjusted_close": "202"},
                            ],
                        },
                    ],
                    "primary_source": {
                        "provider": "TEST market provider",
                        "source_url": "https://example.test/market",
                        "retrieved_at": NOW.isoformat(),
                        "data_as_of": "2026-07-20",
                    },
                }
            },
        ),
        _handoff(
            "context",
            OrchestratorStepKind.CONTEXT_ANALYSIS,
            {
                "analysis": {
                    "source_items": [
                        {
                            "id": "context:novo",
                            "title": "TEST FIXTURE context",
                            "summary": "TEST FIXTURE source-backed context.",
                            "source_url": "https://example.test/context",
                            "source_name": "TEST context source",
                            "source_type": "company_event",
                            "scope": "company",
                            "retrieved_at": NOW.isoformat(),
                            "published_at": NOW.isoformat(),
                        }
                    ]
                }
            },
            evidence_ids=("context:novo",),
        ),
        _handoff(
            "synthesis",
            OrchestratorStepKind.SYNTHESIS,
            {"report": _synthesis_report()},
            evidence_ids=("evidence:novo:revenue", "context:novo"),
        ),
    )
    return OrchestratedResearchRun(
        id="orchestrator_run_novo_fixture",
        query="TEST FIXTURE Novo Nordisk research",
        status=OrchestratorRunStatus.COMPLETE,
        created_at=NOW,
        updated_at=NOW,
        execution_policy=OrchestratorExecutionPolicy.SEQUENTIAL_LOCAL_SAFE,
        plan=default_orchestrator_plan(),
        handoffs=handoffs,
        selected_company={
            "id": "sec:cik:0000353278",
            "legal_name": "TEST FIXTURE NOVO NORDISK A/S",
            "identifiers": [{"type": "cik", "value": "0000353278"}],
        },
        selected_security={
            "id": "sec:ticker:NVO:cik:0000353278",
            "ticker": "NVO",
            "exchange_name": "NYSE",
        },
        synthesis_summary="TEST FIXTURE deterministic Novo summary.",
        scenario_id="novo-nordisk",
        scenario_version="1.0.0",
    )


def _handoff(
    handoff_id: str,
    kind: OrchestratorStepKind,
    output: dict[str, object],
    *,
    evidence_ids: tuple[str, ...] = (),
) -> AgentHandoff:
    return AgentHandoff(
        id=f"handoff:{handoff_id}",
        step_id=kind.value,
        kind=kind,
        status=OrchestratorHandoffStatus.SUCCEEDED,
        started_at=NOW,
        completed_at=NOW,
        output=output,
        evidence_ids=evidence_ids,
        confidence=HandoffConfidence.HIGH,
    )


def _statement_payload(fiscal_year: int) -> dict[str, object]:
    return {
        "id": f"statement:{fiscal_year}",
        "currency": "DKK",
        "period": {"fiscal_year": fiscal_year, "period_type": "annual"},
        "source": {
            "taxonomy_namespaces": ["ifrs-full"],
            "concept_mappings": {"revenue": "ifrs-full:Revenue"},
        },
    }


def _synthesis_report() -> dict[str, object]:
    point = {
        "id": "point:current",
        "title": "Current situation",
        "summary": "TEST FIXTURE deterministic synthesis.",
        "confidence": "high",
        "evidence_ids": ["evidence:novo:revenue"],
        "source_handoff_ids": ["handoff:financial"],
        "limitations": [],
    }
    scenario = {
        "title": "Conditional development",
        "condition": "If TEST FIXTURE conditions change.",
        "potential_development": "Then the outcome may change.",
        "confidence": "low",
        "evidence_ids": ["context:novo"],
        "source_handoff_ids": ["handoff:context"],
        "limitations": [],
    }
    return {
        "id": "synthesis_report_novo_fixture",
        "query": "TEST FIXTURE Novo Nordisk research",
        "status": "complete",
        "created_at": NOW.isoformat(),
        "company_name": "TEST FIXTURE NOVO NORDISK A/S",
        "security_symbol": "NVO",
        "sections": {
            "current_situation": [point],
            "strengths": [],
            "weaknesses": [],
            "opportunities": [],
            "risks": [],
            "unknowns": [],
        },
        "scenarios": {"upside": scenario, "downside": scenario},
        "overall_confidence": "high",
        "evidence_coverage": "substantial",
        "evidence_coverage_ratio": 1.0,
        "evidence_ids": ["evidence:novo:revenue", "context:novo"],
        "warnings": [],
        "limitations": [],
        "no_recommendation_notice": (
            "TEST report does not provide buy, sell, hold, price-target, or personalized "
            "investment advice."
        ),
    }


def _context_payload(
    scope: str,
    *,
    source_url: str | None = None,
    published_at: str | None = None,
) -> dict[str, object]:
    return {
        "id": f"context:{scope}",
        "title": f"TEST FIXTURE {scope}",
        "summary": "TEST FIXTURE context.",
        "source_url": source_url or f"https://example.test/{scope}",
        "source_name": "TEST FIXTURE source",
        "source_type": "company_event" if scope == "company" else "rates",
        "reliability": "official",
        "scope": scope,
        "retrieved_at": NOW.isoformat(),
        "published_at": published_at or NOW.isoformat(),
        "metadata": {"fixture": "false"},
    }
