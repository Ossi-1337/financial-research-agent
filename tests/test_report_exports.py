from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from pathlib import Path

import pypdfium2 as pdfium
import pytest
from fastapi.testclient import TestClient

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
    UnavailableResearchStepDispatcher,
    default_orchestrator_plan,
)
from financial_research_agent.report_exports import (
    MAX_SOURCE_QUOTE_CHARS,
    ReportExportError,
    ReportExportErrorCode,
    ReportExportFormat,
    ReportExportService,
    ReportExportSnapshot,
    ReportExportStore,
    build_report_evidence_index,
    build_report_export_document,
    render_html,
    render_markdown,
    render_pdf,
)
from financial_research_agent.settings import Settings
from financial_research_agent.synthesis import (
    NARRATIVE_SECTION_ORDER,
    NarrativeParagraph,
    NarrativePresentation,
    NarrativePresentationSection,
    NarrativePresentationStore,
    NarrativeSection,
    synthesis_sha256,
)
from financial_research_agent.web import ChatSessionStore, create_app

NOW = datetime(2026, 7, 20, 12, tzinfo=UTC)


def test_export_contracts_are_immutable_and_manifest_round_trips(tmp_path: Path) -> None:
    snapshot = _export_service(tmp_path).export(_run())
    reloaded = ReportExportSnapshot.from_dict(snapshot.to_dict())

    assert reloaded == snapshot
    assert {artifact.format for artifact in snapshot.artifacts} == set(ReportExportFormat)
    assert all(artifact.byte_size > 0 for artifact in snapshot.artifacts)
    assert all(len(artifact.sha256) == 64 for artifact in snapshot.artifacts)
    assert snapshot.content_version == 3
    with pytest.raises(FrozenInstanceError):
        snapshot.export_id = "changed"


def test_legacy_manifest_defaults_to_original_content_version(tmp_path: Path) -> None:
    payload = _export_service(tmp_path).export(_run()).to_dict()
    payload.pop("content_version")

    snapshot = ReportExportSnapshot.from_dict(payload)

    assert snapshot.content_version == 1


def test_content_version_two_manifest_remains_readable(tmp_path: Path) -> None:
    payload = _export_service(tmp_path).export(_run()).to_dict()
    payload["content_version"] = 2

    snapshot = ReportExportSnapshot.from_dict(payload)

    assert snapshot.content_version == 2


def test_source_resolver_deduplicates_sources_and_marks_unknown_evidence() -> None:
    document = _document()
    evidence = build_report_evidence_index(_run(), redaction_policy=RedactionPolicy())

    assert [source.marker for source in document.sources] == ["[S1]", "[S2]", "[S3]", "[S4]"]
    assert document.sources[0].source_url == "https://example.test/filing"
    assert {"ev:financial", "citation:1", "snippet:1"} <= set(document.sources[0].evidence_ids)
    assert len(document.sources[0].quote or "") <= MAX_SOURCE_QUOTE_CHARS
    unresolved = next(source for source in document.sources if not source.resolved)
    assert unresolved.evidence_ids == ("ev:unknown",)
    assert unresolved.source_url is None
    assert "[S1]" in document.current_situation[0].source_markers
    assert unresolved.marker in document.risks[0].source_markers
    assert evidence.evidence_markers["ev:financial"] == ("[S1]",)
    assert evidence.handoff_markers["handoff_context"] == ("[S2]",)
    assert evidence.unresolved_evidence_ids == ("ev:unknown",)
    assert all(len(source.quote or "") <= MAX_SOURCE_QUOTE_CHARS for source in evidence.sources)
    assert [series.symbol for series in document.chart_series] == ["TEST", "SPY"]
    assert document.chart_series[0].points[0].indexed_value == 100
    assert document.chart_series[0].points[-1].indexed_value == 110


def test_renderers_escape_untrusted_text_and_create_unicode_pdf() -> None:
    document = _document()

    markdown = render_markdown(document).decode("utf-8")
    html = render_html(document).decode("utf-8")
    pdf = render_pdf(document)

    assert "<script>" not in markdown
    assert "&lt;script&gt;" in markdown
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "<script" not in html.casefold()
    assert "<script" not in html.split("<style>", 1)[1].split("</style>", 1)[1].casefold()
    assert "Current Situation" in markdown
    assert "Current Situation" in html
    assert "[S1]" in markdown
    assert "[S1]" in html
    assert "LLM provider/model" in markdown
    assert "not used / not used" in markdown
    assert "Indexed Price Development" in markdown
    assert "<svg" in markdown
    assert "TEST: 110.0" in markdown
    assert "Indexed Price Development" in html
    assert "<svg" in html
    assert "SPY: 102.0" in html
    assert pdf.startswith(b"%PDF")
    assert len(pdf) > 10_000
    pdf_document = pdfium.PdfDocument(pdf)
    try:
        pdf_text = "\n".join(
            pdf_document[index].get_textpage().get_text_range()
            for index in range(len(pdf_document))
        )
    finally:
        pdf_document.close()
    assert "Indexed Price Development" in pdf_text
    assert "TEST: 110.0" in pdf_text


def test_new_exports_include_matching_narrative_before_structured_report(
    tmp_path: Path,
) -> None:
    run = _run()
    store = NarrativePresentationStore((_narrative(run),))
    service = ReportExportService(
        store=ReportExportStore(root=tmp_path / "exports"),
        redaction_policy=RedactionPolicy(),
        narrative_store=store,
        now=lambda: NOW,
        id_factory=lambda: "export_narrative",
    )

    snapshot = service.export(run)
    markdown_artifact = snapshot.artifact(ReportExportFormat.MARKDOWN)
    html_artifact = snapshot.artifact(ReportExportFormat.HTML)
    assert markdown_artifact is not None
    assert html_artifact is not None
    markdown = (tmp_path / "exports" / snapshot.export_id / markdown_artifact.filename).read_text(
        encoding="utf-8"
    )
    html = (tmp_path / "exports" / snapshot.export_id / html_artifact.filename).read_text(
        encoding="utf-8"
    )

    assert snapshot.narrative_synthesis_sha256 == _narrative(run).synthesis_sha256
    assert markdown.index("LLM-Generated Narrative") < markdown.index("Current Situation")
    assert "**Narrative provider/model:** local\\-openai / test\\-model" in markdown
    assert "[S1]" in markdown
    assert "LLM-Generated Narrative" in html
    assert "structured report" in html


def test_optional_legacy_narrative_failure_does_not_block_export(tmp_path: Path) -> None:
    class BrokenNarrativeStore:
        def matching(self, **_kwargs):
            raise ValueError("corrupt legacy narrative")

    service = ReportExportService(
        store=ReportExportStore(root=tmp_path / "exports"),
        redaction_policy=RedactionPolicy(),
        narrative_store=BrokenNarrativeStore(),
        now=lambda: NOW,
        id_factory=lambda: "export_without_legacy_narrative",
    )

    snapshot = service.export(_run())

    assert len(snapshot.artifacts) == 3
    assert snapshot.narrative_synthesis_sha256 is None


def test_redaction_removes_secrets_and_local_paths_from_all_artifacts(tmp_path: Path) -> None:
    secret = "sk-test-secret-value"
    local_path = str(tmp_path / "private" / "source.txt")
    run = _run(summary=f"Sensitive {secret} at {local_path}")
    service = ReportExportService(
        store=ReportExportStore(root=tmp_path / "exports"),
        redaction_policy=RedactionPolicy(
            sensitive_values=(secret,),
            sensitive_paths=(str(tmp_path),),
        ),
        now=lambda: NOW,
        id_factory=lambda: "export_redacted",
    )

    snapshot = service.export(run)

    for artifact in snapshot.artifacts:
        content = (tmp_path / "exports" / snapshot.export_id / artifact.filename).read_bytes()
        assert secret.encode() not in content
        assert str(tmp_path).encode() not in content


def test_store_is_atomic_immutable_sorted_and_rejects_traversal(tmp_path: Path) -> None:
    service = _export_service(tmp_path)

    first = service.export(_run())
    second = _export_service(
        tmp_path,
        export_id="export_second",
        now=datetime(2026, 7, 21, tzinfo=UTC),
    ).export(_run())

    assert [snapshot.export_id for snapshot in service.store.list()] == [
        second.export_id,
        first.export_id,
    ]
    assert not any(path.name.startswith(".") for path in service.store.root.iterdir())
    with pytest.raises(ReportExportError) as error:
        service.export(_run())
    assert error.value.code == ReportExportErrorCode.REPORT_EXPORT_FAILED
    with pytest.raises(ValueError, match="invalid report export id"):
        service.store.get("../outside")


def test_api_creates_lists_gets_and_downloads_all_formats(tmp_path: Path) -> None:
    run_store = OrchestratorRunStore(storage_path=tmp_path / "orchestrator_runs.json")
    run = _run()
    run_store.save(run)
    narrative_store = NarrativePresentationStore((_narrative(run),))
    client = TestClient(
        create_app(
            settings=Settings.from_env({"FRA_HOME": str(tmp_path)}),
            session_store=ChatSessionStore(),
            orchestrator_run_store=run_store,
            narrative_store=narrative_store,
        )
    )

    created = client.post("/api/orchestrator/runs/orchestrator_run_fixture/exports")
    payload = created.json()
    export_id = payload["export"]["export_id"]

    assert created.status_code == 201
    assert payload["export"]["narrative_synthesis_sha256"] == _narrative(run).synthesis_sha256
    listed = client.get("/api/report-exports").json()["exports"]
    assert listed[0]["export"]["export_id"] == export_id
    assert client.get(f"/api/report-exports/{export_id}").status_code == 200
    for export_format in ReportExportFormat:
        response = client.get(f"/api/report-exports/{export_id}/files/{export_format.value}")
        assert response.status_code == 200
        assert response.headers["content-disposition"].startswith("attachment;")
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.content
        if export_format == ReportExportFormat.HTML:
            assert "default-src 'none'" in response.headers["content-security-policy"]
        if export_format == ReportExportFormat.PDF:
            assert response.content.startswith(b"%PDF")


def test_api_exports_without_waiting_for_a_narrative_provider(
    tmp_path: Path,
) -> None:
    run_store = OrchestratorRunStore(storage_path=tmp_path / "orchestrator_runs.json")
    run_store.save(_run())
    client = TestClient(
        create_app(
            settings=Settings.from_env({"FRA_HOME": str(tmp_path)}),
            session_store=ChatSessionStore(),
            orchestrator_run_store=run_store,
            research_dispatcher=UnavailableResearchStepDispatcher(),
        )
    )

    created = client.post("/api/orchestrator/runs/orchestrator_run_fixture/exports")
    payload = created.json()

    assert created.status_code == 201
    assert payload["export"]["narrative_synthesis_sha256"] is None
    markdown = client.get(payload["files"]["markdown"])
    assert "LLM-Generated Narrative" not in markdown.text


def test_api_reports_unknown_and_missing_synthesis_without_paths(tmp_path: Path) -> None:
    run_store = OrchestratorRunStore(storage_path=tmp_path / "orchestrator_runs.json")
    run_store.save(_run(include_synthesis=False))
    client = TestClient(
        create_app(
            settings=Settings.from_env({"FRA_HOME": str(tmp_path)}),
            session_store=ChatSessionStore(),
            orchestrator_run_store=run_store,
        )
    )

    unknown_run = client.post("/api/orchestrator/runs/missing/exports")
    unavailable = client.post("/api/orchestrator/runs/orchestrator_run_fixture/exports")
    unknown_export = client.get("/api/report-exports/missing")
    traversal = client.get("/api/report-exports/..%2Foutside")

    assert unknown_run.status_code == 404
    assert unavailable.status_code == 409
    assert unavailable.json()["detail"]["error"] == "synthesis_report_unavailable"
    assert unknown_export.status_code == 404
    assert traversal.status_code == 404
    assert str(tmp_path) not in unavailable.text


def _export_service(
    tmp_path: Path,
    *,
    export_id: str = "export_fixture",
    now: datetime = NOW,
) -> ReportExportService:
    return ReportExportService(
        store=ReportExportStore(root=tmp_path / "exports"),
        redaction_policy=RedactionPolicy(),
        now=lambda: now,
        id_factory=lambda: export_id,
    )


def _document():
    document = build_report_export_document(
        _run(),
        export_id="export_document",
        generated_at=NOW,
        redaction_policy=RedactionPolicy(),
    )
    assert document is not None
    return document


def _run(
    *,
    include_synthesis: bool = True,
    summary: str = "TEST FIXTURE <script>alert('x')</script> Økonomisk situation.",
) -> OrchestratedResearchRun:
    handoffs = [
        _handoff(
            "handoff_financial",
            OrchestratorStepKind.FINANCIAL_REPORT_ANALYSIS,
            {
                "analysis": {
                    "citations": [
                        {
                            "id": "citation:1",
                            "evidence_id": "ev:financial",
                            "source_url": "https://example.test/filing",
                            "retrieved_at": NOW.isoformat(),
                            "section": "Revenue",
                            "quote": "Q" * 500,
                            "metadata": {
                                "source_name": "TEST SEC filing",
                                "filing_date": "2026-06-30",
                            },
                        }
                    ],
                    "evidence": [
                        {
                            "id": "snippet:1",
                            "citation_id": "citation:1",
                            "text": "TEST TOOL OUTPUT revenue evidence.",
                            "source_url": "https://example.test/filing",
                            "retrieved_at": NOW.isoformat(),
                            "section": "Revenue",
                        }
                    ],
                }
            },
            evidence_ids=("ev:financial",),
        ),
        _handoff(
            "handoff_context",
            OrchestratorStepKind.CONTEXT_ANALYSIS,
            {
                "analysis": {
                    "source_items": [
                        {
                            "id": "context:1",
                            "title": "TEST context",
                            "summary": "TEST TOOL OUTPUT context.",
                            "source_url": "https://example.test/context",
                            "source_name": "TEST context source",
                            "source_type": "company_news",
                            "scope": "company",
                            "retrieved_at": NOW.isoformat(),
                            "published_at": "2026-07-19T12:00:00+00:00",
                        },
                        {
                            "id": "context:duplicate",
                            "summary": "TEST duplicate context.",
                            "source_url": "https://example.test/context",
                            "source_name": "TEST context source",
                            "scope": "company",
                            "retrieved_at": NOW.isoformat(),
                        },
                    ]
                }
            },
            evidence_ids=("context:1",),
        ),
        _handoff(
            "handoff_stock",
            OrchestratorStepKind.STOCK_PRICE_ANALYSIS,
            {
                "analysis": {
                    "chart_series": [
                        {
                            "symbol": "TEST",
                            "points": [
                                {
                                    "priced_at": "2026-07-14",
                                    "close": "100",
                                    "adjusted_close": "100",
                                },
                                {
                                    "priced_at": "2026-07-16",
                                    "close": "105",
                                    "adjusted_close": "105",
                                },
                                {
                                    "priced_at": "2026-07-18",
                                    "close": "110",
                                    "adjusted_close": "110",
                                },
                            ],
                        },
                        {
                            "symbol": "SPY",
                            "points": [
                                {
                                    "priced_at": "2026-07-14",
                                    "close": "500",
                                    "adjusted_close": "500",
                                },
                                {
                                    "priced_at": "2026-07-16",
                                    "close": "507.5",
                                    "adjusted_close": "507.5",
                                },
                                {
                                    "priced_at": "2026-07-18",
                                    "close": "510",
                                    "adjusted_close": "510",
                                },
                            ],
                        },
                    ],
                    "primary_source": {
                        "provider": "TEST market provider",
                        "source_url": "https://example.test/market",
                        "retrieved_at": NOW.isoformat(),
                        "data_as_of": "2026-07-18",
                        "attribution": "TEST TOOL OUTPUT market data.",
                    },
                }
            },
        ),
    ]
    if include_synthesis:
        handoffs.append(
            _handoff(
                "handoff_synthesis",
                OrchestratorStepKind.SYNTHESIS,
                {"report": _report(summary)},
                evidence_ids=("ev:financial", "context:1", "ev:unknown"),
            )
        )
    return OrchestratedResearchRun(
        id="orchestrator_run_fixture",
        query="TEST FIXTURE company research",
        status=OrchestratorRunStatus.PARTIAL,
        created_at=NOW,
        updated_at=NOW,
        execution_policy=OrchestratorExecutionPolicy.SEQUENTIAL_LOCAL_SAFE,
        plan=default_orchestrator_plan(),
        agent_provider="local-openai",
        agent_model="test-model",
        handoffs=tuple(handoffs),
        selected_company={"id": "company_fixture", "legal_name": "TEST Økonomi A/S"},
        selected_security={"id": "security_fixture", "ticker": "TEST"},
        synthesis_summary=summary if include_synthesis else None,
        warnings=("TEST fixture warning.",),
        limitations=("TEST fixture partial data.",),
    )


def _handoff(
    handoff_id: str,
    kind: OrchestratorStepKind,
    output: dict[str, object],
    *,
    evidence_ids: tuple[str, ...] = (),
) -> AgentHandoff:
    return AgentHandoff(
        id=handoff_id,
        step_id=kind.value,
        kind=kind,
        status=OrchestratorHandoffStatus.PARTIAL,
        started_at=NOW,
        completed_at=NOW,
        output=output,
        evidence_ids=evidence_ids,
        confidence=HandoffConfidence.LOW,
    )


def _report(summary: str) -> dict[str, object]:
    def point(
        point_id: str,
        title: str,
        evidence_ids: list[str],
        handoff_ids: list[str],
    ) -> dict[str, object]:
        return {
            "id": point_id,
            "title": title,
            "summary": summary,
            "confidence": "low",
            "evidence_ids": evidence_ids,
            "source_handoff_ids": handoff_ids,
            "limitations": ["TEST fixture limitation."],
        }

    return {
        "id": "synthesis_report_fixture",
        "query": "TEST FIXTURE company research",
        "status": "partial",
        "created_at": NOW.isoformat(),
        "company_name": "TEST Økonomi A/S",
        "security_symbol": "TEST",
        "sections": {
            "current_situation": [
                point("point_current", "Current", ["ev:financial"], ["handoff_financial"])
            ],
            "strengths": [point("point_strength", "Strength", ["context:1"], ["handoff_context"])],
            "weaknesses": [],
            "opportunities": [],
            "risks": [point("point_risk", "Risk", ["ev:unknown"], [])],
            "unknowns": [],
        },
        "scenarios": {
            "upside": {
                "title": "Conditional upside",
                "condition": "If fixture conditions improve.",
                "potential_development": "Then fixture performance may improve.",
                "confidence": "low",
                "evidence_ids": ["context:1"],
                "source_handoff_ids": ["handoff_context"],
                "limitations": [],
            },
            "downside": {
                "title": "Conditional downside",
                "condition": "If fixture conditions worsen.",
                "potential_development": "Then fixture performance may weaken.",
                "confidence": "low",
                "evidence_ids": [],
                "source_handoff_ids": ["handoff_stock"],
                "limitations": [],
            },
        },
        "overall_confidence": "low",
        "evidence_coverage": "limited",
        "evidence_coverage_ratio": 0.5,
        "evidence_ids": ["ev:financial", "context:1", "ev:unknown"],
        "warnings": ["TEST synthesis warning."],
        "limitations": ["TEST synthesis limitation."],
        "no_recommendation_notice": (
            "TEST report is source-backed research only and does not provide buy, sell, "
            "hold, price-target, or personalized investment advice."
        ),
    }


def _narrative(run: OrchestratedResearchRun) -> NarrativePresentation:
    report = run.handoffs[-1].output["report"]
    assert isinstance(report, dict)
    return NarrativePresentation(
        id="narrative_export_fixture",
        run_id=run.id,
        report_id="synthesis_report_fixture",
        synthesis_sha256=synthesis_sha256(report),
        provider="local-openai",
        model="test-model",
        created_at=NOW,
        sections=tuple(
            NarrativePresentationSection(
                section=section,
                paragraphs=(
                    (
                        NarrativeParagraph(
                            text="The stored evidence supports only a partial assessment.",
                            source_point_ids=("point_current",),
                            evidence_ids=("ev:financial",),
                            source_markers=("[S1]",),
                        ),
                    )
                    if section == NarrativeSection.CURRENT_SITUATION
                    else ()
                ),
            )
            for section in NARRATIVE_SECTION_ORDER
        ),
        warnings=("TEST narrative warning.",),
        limitations=("TEST narrative limitation.",),
        no_recommendation_notice=("TEST narrative is research only and is not investment advice."),
    )
