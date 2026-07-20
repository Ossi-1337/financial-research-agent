from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from financial_research_agent.observability import RedactionPolicy
from financial_research_agent.orchestration import OrchestratedResearchRun, OrchestratorStepKind

from .contracts import (
    MAX_SOURCE_QUOTE_CHARS,
    ReportExportDocument,
    ReportExportPoint,
    ReportExportScenario,
    ReportSourceReference,
)

_SECTION_NAMES = (
    "current_situation",
    "strengths",
    "weaknesses",
    "opportunities",
    "risks",
    "unknowns",
)


@dataclass(slots=True)
class _SourceCandidate:
    evidence_ids: list[str]
    source_url: str | None
    source_name: str | None = None
    source_date: str | None = None
    retrieved_at: str | None = None
    section: str | None = None
    quote: str | None = None
    handoff_ids: tuple[str, ...] = ()


def build_report_export_document(
    run: OrchestratedResearchRun,
    *,
    export_id: str,
    generated_at: datetime,
    redaction_policy: RedactionPolicy,
) -> ReportExportDocument | None:
    payload = _redacted_run_payload(run, redaction_policy)
    report = _synthesis_report(payload)
    if report is None:
        return None

    candidates = _source_candidates(payload)
    report_evidence_ids = _report_evidence_ids(report)
    known_evidence_ids = {
        evidence_id for candidate in candidates for evidence_id in candidate.evidence_ids
    }
    candidates.extend(
        _SourceCandidate(evidence_ids=[evidence_id], source_url=None)
        for evidence_id in report_evidence_ids
        if evidence_id not in known_evidence_ids
    )
    sources, evidence_markers, handoff_markers = _finalize_sources(candidates)

    sections = _mapping(report.get("sections"))
    scenarios = _mapping(report.get("scenarios"))
    company = _mapping(payload.get("selected_company"))
    security = _mapping(payload.get("selected_security"))

    point_sections = {
        name: _points(
            sections.get(name),
            evidence_markers=evidence_markers,
            handoff_markers=handoff_markers,
        )
        for name in _SECTION_NAMES
    }
    return ReportExportDocument(
        export_id=export_id,
        run_id=_text(payload.get("id"), "run"),
        report_id=_text(report.get("id"), "report"),
        generated_at=generated_at,
        run_created_at=_datetime(payload.get("created_at")),
        run_updated_at=_datetime(payload.get("updated_at")),
        report_created_at=_datetime(report.get("created_at")),
        query=_text(report.get("query") or payload.get("query"), "Research query"),
        run_status=_text(payload.get("status"), "unknown"),
        report_status=_text(report.get("status"), "unknown"),
        company_name=_optional_text(report.get("company_name") or company.get("legal_name")),
        company_id=_optional_text(company.get("id")),
        ticker=_optional_text(report.get("security_symbol") or security.get("ticker")),
        security_id=_optional_text(security.get("id")),
        current_situation=point_sections["current_situation"],
        strengths=point_sections["strengths"],
        weaknesses=point_sections["weaknesses"],
        opportunities=point_sections["opportunities"],
        risks=point_sections["risks"],
        unknowns=point_sections["unknowns"],
        upside_scenario=_scenario(
            scenarios.get("upside"),
            "Upside scenario unavailable.",
            evidence_markers,
            handoff_markers,
        ),
        downside_scenario=_scenario(
            scenarios.get("downside"),
            "Downside scenario unavailable.",
            evidence_markers,
            handoff_markers,
        ),
        overall_confidence=_text(report.get("overall_confidence"), "unknown"),
        evidence_coverage=_text(report.get("evidence_coverage"), "none"),
        evidence_coverage_ratio=_float(report.get("evidence_coverage_ratio")),
        warnings=_texts((*_sequence(payload.get("warnings")), *_sequence(report.get("warnings")))),
        limitations=_texts(
            (*_sequence(payload.get("limitations")), *_sequence(report.get("limitations")))
        ),
        disclaimer=_text(
            report.get("no_recommendation_notice") or payload.get("no_recommendation_notice"),
            "This report is research only and is not financial advice.",
        ),
        sources=sources,
    )


def _redacted_run_payload(
    run: OrchestratedResearchRun,
    policy: RedactionPolicy,
) -> Mapping[str, Any]:
    export_policy = RedactionPolicy(
        sensitive_values=policy.sensitive_values,
        sensitive_paths=policy.sensitive_paths,
        text_preview_chars=100_000,
        collection_preview_items=100_000,
    )
    redacted = export_policy.redact(run.to_dict())
    return _mapping(redacted)


def _synthesis_report(run_payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    for handoff_value in reversed(_sequence(run_payload.get("handoffs"))):
        handoff = _mapping(handoff_value)
        if handoff.get("kind") != OrchestratorStepKind.SYNTHESIS.value:
            continue
        report = _mapping(_mapping(handoff.get("output")).get("report"))
        return report or None
    return None


def _source_candidates(run_payload: Mapping[str, Any]) -> list[_SourceCandidate]:
    candidates: list[_SourceCandidate] = []
    for handoff_value in _sequence(run_payload.get("handoffs")):
        handoff = _mapping(handoff_value)
        handoff_id = _optional_text(handoff.get("id"))
        analysis = _mapping(_mapping(handoff.get("output")).get("analysis"))
        kind = handoff.get("kind")
        if kind == OrchestratorStepKind.FINANCIAL_REPORT_ANALYSIS.value:
            candidates.extend(_financial_sources(analysis, handoff_id))
        elif kind == OrchestratorStepKind.CONTEXT_ANALYSIS.value:
            candidates.extend(_context_sources(analysis, handoff_id))
        elif kind == OrchestratorStepKind.STOCK_PRICE_ANALYSIS.value:
            candidates.extend(_stock_sources(analysis, handoff_id))
    return candidates


def _financial_sources(
    analysis: Mapping[str, Any],
    handoff_id: str | None,
) -> list[_SourceCandidate]:
    evidence = tuple(_mapping(item) for item in _sequence(analysis.get("evidence")))
    by_citation: dict[str, Mapping[str, Any]] = {}
    for item in evidence:
        citation_id = _optional_text(item.get("citation_id"))
        if citation_id:
            by_citation[citation_id] = item

    candidates: list[_SourceCandidate] = []
    used_evidence_ids: set[str] = set()
    for citation_value in _sequence(analysis.get("citations")):
        citation = _mapping(citation_value)
        citation_id = _optional_text(citation.get("id"))
        snippet = by_citation.get(citation_id or "", {})
        evidence_ids = _texts(
            (
                citation.get("evidence_id"),
                citation_id,
                snippet.get("id"),
            )
        )
        used_evidence_ids.update(evidence_ids)
        metadata = _mapping(citation.get("metadata"))
        candidates.append(
            _SourceCandidate(
                evidence_ids=list(evidence_ids),
                source_url=_optional_text(citation.get("source_url") or snippet.get("source_url")),
                source_name=_first_text(metadata, ("source_name", "provider", "source")),
                source_date=_first_text(
                    metadata,
                    ("source_date", "filing_date", "data_as_of", "published_at"),
                ),
                retrieved_at=_optional_text(
                    citation.get("retrieved_at") or snippet.get("retrieved_at")
                ),
                section=_optional_text(citation.get("section") or snippet.get("section")),
                quote=_bounded_quote(citation.get("quote") or snippet.get("text")),
                handoff_ids=(handoff_id,) if handoff_id else (),
            )
        )
    for snippet in evidence:
        snippet_id = _optional_text(snippet.get("id"))
        if snippet_id is None or snippet_id in used_evidence_ids:
            continue
        candidates.append(
            _SourceCandidate(
                evidence_ids=[snippet_id],
                source_url=_optional_text(snippet.get("source_url")),
                retrieved_at=_optional_text(snippet.get("retrieved_at")),
                section=_optional_text(snippet.get("section")),
                quote=_bounded_quote(snippet.get("text")),
                handoff_ids=(handoff_id,) if handoff_id else (),
            )
        )
    return candidates


def _context_sources(
    analysis: Mapping[str, Any],
    handoff_id: str | None,
) -> list[_SourceCandidate]:
    return [
        _SourceCandidate(
            evidence_ids=list(_texts((item.get("id"),))),
            source_url=_optional_text(item.get("source_url")),
            source_name=_optional_text(item.get("source_name")),
            source_date=_optional_text(item.get("published_at")),
            retrieved_at=_optional_text(item.get("retrieved_at")),
            section=_optional_text(item.get("scope") or item.get("source_type")),
            quote=_bounded_quote(item.get("summary")),
            handoff_ids=(handoff_id,) if handoff_id else (),
        )
        for value in _sequence(analysis.get("source_items"))
        if (item := _mapping(value))
    ]


def _stock_sources(
    analysis: Mapping[str, Any],
    handoff_id: str | None,
) -> list[_SourceCandidate]:
    candidates: list[_SourceCandidate] = []
    for key, label in (
        ("primary_source", "primary"),
        ("benchmark_source", "benchmark"),
    ):
        source = _mapping(analysis.get(key))
        if not source:
            continue
        evidence_id = f"{handoff_id or 'stock_analysis'}:{label}_source"
        candidates.append(
            _SourceCandidate(
                evidence_ids=[evidence_id],
                source_url=_optional_text(source.get("source_url")),
                source_name=_optional_text(source.get("provider")),
                source_date=_optional_text(source.get("data_as_of")),
                retrieved_at=_optional_text(source.get("retrieved_at")),
                section=f"stock_{label}",
                quote=_bounded_quote(source.get("attribution")),
                handoff_ids=(handoff_id,) if handoff_id else (),
            )
        )
    return candidates


def _finalize_sources(
    candidates: Iterable[_SourceCandidate],
) -> tuple[
    tuple[ReportSourceReference, ...],
    Mapping[str, tuple[str, ...]],
    Mapping[str, tuple[str, ...]],
]:
    merged: list[_SourceCandidate] = []
    indexes: dict[str, int] = {}
    for candidate in candidates:
        if not candidate.evidence_ids:
            continue
        key = (
            f"url:{candidate.source_url.casefold()}"
            if candidate.source_url
            else f"unresolved:{candidate.evidence_ids[0]}"
        )
        if key not in indexes:
            indexes[key] = len(merged)
            merged.append(candidate)
            continue
        existing = merged[indexes[key]]
        existing.evidence_ids[:] = list(
            dict.fromkeys((*existing.evidence_ids, *candidate.evidence_ids))
        )
        existing.handoff_ids = tuple(dict.fromkeys((*existing.handoff_ids, *candidate.handoff_ids)))
        if existing.quote is None:
            existing.quote = candidate.quote
        if existing.section is None:
            existing.section = candidate.section

    sources: list[ReportSourceReference] = []
    evidence_markers: dict[str, list[str]] = {}
    handoff_markers: dict[str, list[str]] = {}
    for index, candidate in enumerate(merged, start=1):
        marker = f"[S{index}]"
        source = ReportSourceReference(
            marker=marker,
            evidence_ids=tuple(candidate.evidence_ids),
            resolved=candidate.source_url is not None,
            source_url=candidate.source_url,
            source_name=candidate.source_name,
            source_date=candidate.source_date,
            retrieved_at=candidate.retrieved_at,
            section=candidate.section,
            quote=candidate.quote,
        )
        sources.append(source)
        for evidence_id in candidate.evidence_ids:
            evidence_markers.setdefault(evidence_id, []).append(marker)
        for handoff_id in candidate.handoff_ids:
            handoff_markers.setdefault(handoff_id, []).append(marker)
    return (
        tuple(sources),
        {key: tuple(dict.fromkeys(value)) for key, value in evidence_markers.items()},
        {key: tuple(dict.fromkeys(value)) for key, value in handoff_markers.items()},
    )


def _points(
    value: object,
    *,
    evidence_markers: Mapping[str, tuple[str, ...]],
    handoff_markers: Mapping[str, tuple[str, ...]],
) -> tuple[ReportExportPoint, ...]:
    points: list[ReportExportPoint] = []
    for index, item_value in enumerate(_sequence(value), start=1):
        item = _mapping(item_value)
        evidence_ids = _texts(_sequence(item.get("evidence_ids")))
        handoff_ids = _texts(_sequence(item.get("source_handoff_ids")))
        points.append(
            ReportExportPoint(
                id=_text(item.get("id"), f"point-{index}"),
                title=_text(item.get("title"), f"Finding {index}"),
                summary=_text(item.get("summary"), "No summary available."),
                confidence=_text(item.get("confidence"), "unknown"),
                evidence_ids=evidence_ids,
                source_markers=_markers(
                    evidence_ids,
                    handoff_ids,
                    evidence_markers,
                    handoff_markers,
                ),
                limitations=_texts(_sequence(item.get("limitations"))),
            )
        )
    return tuple(points)


def _scenario(
    value: object,
    fallback: str,
    evidence_markers: Mapping[str, tuple[str, ...]],
    handoff_markers: Mapping[str, tuple[str, ...]],
) -> ReportExportScenario:
    scenario = _mapping(value)
    evidence_ids = _texts(_sequence(scenario.get("evidence_ids")))
    handoff_ids = _texts(_sequence(scenario.get("source_handoff_ids")))
    return ReportExportScenario(
        title=_text(scenario.get("title"), fallback),
        condition=_text(scenario.get("condition"), "Condition unavailable."),
        potential_development=_text(
            scenario.get("potential_development"),
            "Potential development unavailable.",
        ),
        confidence=_text(scenario.get("confidence"), "unknown"),
        evidence_ids=evidence_ids,
        source_markers=_markers(
            evidence_ids,
            handoff_ids,
            evidence_markers,
            handoff_markers,
        ),
        limitations=_texts(_sequence(scenario.get("limitations"))),
    )


def _markers(
    evidence_ids: Iterable[str],
    handoff_ids: Iterable[str],
    evidence_markers: Mapping[str, tuple[str, ...]],
    handoff_markers: Mapping[str, tuple[str, ...]],
) -> tuple[str, ...]:
    values = (
        *(
            marker
            for evidence_id in evidence_ids
            for marker in evidence_markers.get(evidence_id, ())
        ),
        *(marker for handoff_id in handoff_ids for marker in handoff_markers.get(handoff_id, ())),
    )
    return tuple(dict.fromkeys(values))


def _report_evidence_ids(report: Mapping[str, Any]) -> tuple[str, ...]:
    values: list[object] = list(_sequence(report.get("evidence_ids")))
    for section in _mapping(report.get("sections")).values():
        for point_value in _sequence(section):
            values.extend(_sequence(_mapping(point_value).get("evidence_ids")))
    for scenario_value in _mapping(report.get("scenarios")).values():
        values.extend(_sequence(_mapping(scenario_value).get("evidence_ids")))
    return _texts(values)


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> tuple[Any, ...]:
    if isinstance(value, list | tuple):
        return tuple(value)
    return ()


def _text(value: object, fallback: str) -> str:
    return _optional_text(value) or fallback


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _texts(values: Iterable[object]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(text for value in values if (text := _optional_text(value)) is not None)
    )


def _bounded_quote(value: object) -> str | None:
    text = _optional_text(value)
    if text is None or len(text) <= MAX_SOURCE_QUOTE_CHARS:
        return text
    return f"{text[: MAX_SOURCE_QUOTE_CHARS - 1].rstrip()}…"


def _first_text(values: Mapping[str, Any], keys: Iterable[str]) -> str | None:
    return next(
        (text for key in keys if (text := _optional_text(values.get(key))) is not None),
        None,
    )


def _datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def _float(value: object) -> float:
    try:
        return float(value)
    except TypeError, ValueError:
        return 0.0
