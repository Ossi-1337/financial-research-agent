from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

from financial_research_agent.orchestration.contracts import (
    AgentHandoff,
    OrchestratorHandoffStatus,
    OrchestratorStepKind,
)
from financial_research_agent.synthesis.contracts import (
    ConfidenceLabel,
    EvidenceCoverage,
    ScenarioDirection,
    SynthesisPoint,
    SynthesisReport,
    SynthesisReportStatus,
    SynthesisScenario,
    SynthesisSection,
)

SUPPORTED_SPECIALIST_KINDS = {
    OrchestratorStepKind.FINANCIAL_REPORT_ANALYSIS,
    OrchestratorStepKind.STOCK_PRICE_ANALYSIS,
    OrchestratorStepKind.CONTEXT_ANALYSIS,
}

SECTION_COUNT_FOR_COVERAGE = 7


@dataclass(frozen=True, slots=True)
class _ScenarioSupport:
    point_count: int
    evidence_ids: tuple[str, ...]
    handoff_ids: tuple[str, ...]


class SynthesisAgent:
    """Builds a bounded deterministic synthesis from persisted specialist handoffs."""

    def __init__(self, *, now: datetime | None = None) -> None:
        self._now = now

    def synthesize(
        self,
        *,
        query: str,
        handoffs: tuple[AgentHandoff, ...],
        selected_company: Mapping[str, object] | None = None,
        selected_security: Mapping[str, object] | None = None,
        created_at: datetime | None = None,
    ) -> SynthesisReport:
        created = created_at or self._now or datetime.now(UTC)
        buckets = _empty_buckets()
        specialist_handoffs = tuple(
            handoff for handoff in handoffs if handoff.kind in SUPPORTED_SPECIALIST_KINDS
        )
        for handoff in specialist_handoffs:
            for point in _points_from_handoff(handoff):
                buckets[point.section].append(point)

        unknowns = _unknown_points(specialist_handoffs)
        if not specialist_handoffs:
            unknowns = (
                *unknowns,
                _point(
                    section=SynthesisSection.UNKNOWNS,
                    title="No specialist outputs",
                    summary=(
                        "No financial report, stock price, or context specialist outputs were "
                        "available, so the report cannot support company-specific conclusions."
                    ),
                    confidence=ConfidenceLabel.UNKNOWN,
                    limitations=("Run specialist analysis or provide source-linked inputs first.",),
                ),
            )

        buckets[SynthesisSection.UNKNOWNS].extend(unknowns)
        if not buckets[SynthesisSection.CURRENT_SITUATION]:
            buckets[SynthesisSection.CURRENT_SITUATION].append(
                _point(
                    section=SynthesisSection.CURRENT_SITUATION,
                    title="Current situation unavailable",
                    summary=(
                        "The available specialist outputs do not provide enough supported "
                        "information to summarize the current situation."
                    ),
                    confidence=ConfidenceLabel.UNKNOWN,
                    limitations=(
                        "Current situation needs financial, market, or context evidence.",
                    ),
                )
            )

        confidence = _overall_confidence(buckets)
        coverage_ratio, coverage = _evidence_coverage(buckets)
        warnings = _unique(
            warning for handoff in specialist_handoffs for warning in handoff.warnings
        )
        limitations = _unique(
            limitation for handoff in specialist_handoffs for limitation in handoff.limitations
        )
        status = _status(specialist_handoffs, coverage)
        company_name = _mapping_text(selected_company, "legal_name") or _mapping_text(
            selected_company,
            "display_name",
        )
        security_symbol = _mapping_text(selected_security, "ticker")

        return SynthesisReport(
            id=f"synthesis_report_{uuid4().hex}",
            query=query,
            status=status,
            created_at=created,
            company_name=company_name,
            security_symbol=security_symbol,
            current_situation=tuple(buckets[SynthesisSection.CURRENT_SITUATION]),
            strengths=tuple(buckets[SynthesisSection.STRENGTHS]),
            weaknesses=tuple(buckets[SynthesisSection.WEAKNESSES]),
            opportunities=tuple(buckets[SynthesisSection.OPPORTUNITIES]),
            risks=tuple(buckets[SynthesisSection.RISKS]),
            upside_scenario=_upside_scenario(buckets),
            downside_scenario=_downside_scenario(buckets),
            unknowns=tuple(buckets[SynthesisSection.UNKNOWNS]),
            overall_confidence=confidence,
            evidence_coverage=coverage,
            evidence_coverage_ratio=coverage_ratio,
            warnings=warnings,
            limitations=limitations,
        )


def _points_from_handoff(handoff: AgentHandoff) -> tuple[SynthesisPoint, ...]:
    if handoff.status in {OrchestratorHandoffStatus.FAILED, OrchestratorHandoffStatus.SKIPPED}:
        return ()
    if handoff.kind == OrchestratorStepKind.FINANCIAL_REPORT_ANALYSIS:
        return _financial_report_points(handoff)
    if handoff.kind == OrchestratorStepKind.STOCK_PRICE_ANALYSIS:
        return _stock_price_points(handoff)
    if handoff.kind == OrchestratorStepKind.CONTEXT_ANALYSIS:
        return _context_points(handoff)
    return ()


def _financial_report_points(handoff: AgentHandoff) -> tuple[SynthesisPoint, ...]:
    analysis = _mapping(_mapping(handoff.output).get("analysis"))
    points: list[SynthesisPoint] = []
    for finding in _list(analysis.get("findings")):
        payload = _mapping(finding)
        section = str(payload.get("section", ""))
        target = _financial_target_section(section)
        title = _payload_text(payload, "title", fallback=f"Financial report {section}")
        summary = _payload_text(payload, "summary", fallback=title)
        evidence_ids = tuple(str(item) for item in _list(payload.get("evidence_ids")))
        limitations = tuple(str(item) for item in _list(payload.get("limitations")))
        confidence = _confidence(str(payload.get("confidence", ConfidenceLabel.UNKNOWN)))
        points.append(
            _point(
                section=SynthesisSection.CURRENT_SITUATION,
                title=title,
                summary=f"Financial report specialist: {summary}",
                confidence=confidence,
                evidence_ids=evidence_ids,
                source_handoff_ids=(handoff.id,),
                limitations=limitations,
            )
        )
        points.append(
            _point(
                section=target,
                title=title,
                summary=summary,
                confidence=confidence,
                evidence_ids=evidence_ids,
                source_handoff_ids=(handoff.id,),
                limitations=limitations,
            )
        )
    return tuple(points)


def _stock_price_points(handoff: AgentHandoff) -> tuple[SynthesisPoint, ...]:
    analysis = _mapping(_mapping(handoff.output).get("analysis"))
    points: list[SynthesisPoint] = []
    for finding in _list(analysis.get("findings")):
        payload = _mapping(finding)
        section = str(payload.get("section", ""))
        title = _payload_text(payload, "title", fallback=f"Stock price {section}")
        summary = _payload_text(payload, "summary", fallback=title)
        confidence = _confidence(str(payload.get("confidence", ConfidenceLabel.UNKNOWN)))
        metric_names = tuple(str(item) for item in _list(payload.get("metric_names")))
        limitations = tuple(str(item) for item in _list(payload.get("limitations")))
        enriched_summary = summary
        if metric_names:
            enriched_summary = f"{summary} Metrics: {', '.join(metric_names)}."
        points.append(
            _point(
                section=SynthesisSection.CURRENT_SITUATION,
                title=title,
                summary=f"Stock price specialist: {enriched_summary}",
                confidence=confidence,
                source_handoff_ids=(handoff.id,),
                limitations=limitations,
            )
        )
        target = _stock_target_section(section, str(payload.get("trend", "")))
        points.append(
            _point(
                section=target,
                title=title,
                summary=enriched_summary,
                confidence=confidence,
                source_handoff_ids=(handoff.id,),
                limitations=limitations,
            )
        )
    return tuple(points)


def _context_points(handoff: AgentHandoff) -> tuple[SynthesisPoint, ...]:
    analysis = _mapping(_mapping(handoff.output).get("analysis"))
    points: list[SynthesisPoint] = []
    for finding in _list(analysis.get("findings")):
        payload = _mapping(finding)
        title = _payload_text(payload, "title", fallback="Context finding")
        summary = _payload_text(payload, "summary", fallback=title)
        confidence = _confidence(str(payload.get("confidence", ConfidenceLabel.UNKNOWN)))
        source_ids = tuple(str(item) for item in _list(payload.get("source_item_ids")))
        limitations = tuple(str(item) for item in _list(payload.get("limitations")))
        target = _context_target_section(title, summary)
        points.append(
            _point(
                section=SynthesisSection.CURRENT_SITUATION,
                title=title,
                summary=f"Context specialist: {summary}",
                confidence=confidence,
                evidence_ids=source_ids,
                source_handoff_ids=(handoff.id,),
                limitations=limitations,
            )
        )
        points.append(
            _point(
                section=target,
                title=title,
                summary=summary,
                confidence=confidence,
                evidence_ids=source_ids,
                source_handoff_ids=(handoff.id,),
                limitations=limitations,
            )
        )
    return tuple(points)


def _unknown_points(handoffs: tuple[AgentHandoff, ...]) -> tuple[SynthesisPoint, ...]:
    points: list[SynthesisPoint] = []
    seen_kinds = {handoff.kind for handoff in handoffs}
    for kind in SUPPORTED_SPECIALIST_KINDS - seen_kinds:
        points.append(
            _point(
                section=SynthesisSection.UNKNOWNS,
                title=f"{kind.value.replace('_', ' ').title()} missing",
                summary=(
                    f"The {kind.value} specialist did not provide output, so related claims "
                    "remain unsupported."
                ),
                confidence=ConfidenceLabel.UNKNOWN,
                limitations=(f"Missing {kind.value} specialist output.",),
            )
        )
    for handoff in handoffs:
        if handoff.status in {
            OrchestratorHandoffStatus.PARTIAL,
            OrchestratorHandoffStatus.SKIPPED,
            OrchestratorHandoffStatus.FAILED,
        }:
            reason = "; ".join(handoff.limitations) or handoff.error_message or handoff.status.value
            points.append(
                _point(
                    section=SynthesisSection.UNKNOWNS,
                    title=f"{handoff.kind.value.replace('_', ' ').title()} limitation",
                    summary=f"{handoff.kind.value} was {handoff.status.value}: {reason}",
                    confidence=ConfidenceLabel.UNKNOWN,
                    source_handoff_ids=(handoff.id,),
                    limitations=handoff.limitations or (reason,),
                )
            )
    return tuple(points)


def _upside_scenario(
    buckets: dict[SynthesisSection, list[SynthesisPoint]],
) -> SynthesisScenario:
    support = _scenario_support(
        (*buckets[SynthesisSection.STRENGTHS], *buckets[SynthesisSection.OPPORTUNITIES])
    )
    if support.handoff_ids:
        condition = (
            "If the cited strengths and opportunities remain durable and the listed risks "
            "are managed, the company could show better operating resilience than the "
            "base evidence currently proves."
        )
        development = (
            "If that condition holds, future analysis may see stronger fundamentals or "
            "sentiment, but this is a conditional scenario rather than a forecast."
        )
    else:
        condition = (
            "If later evidence identifies durable strengths or opportunities, an upside "
            "case could be developed from those sources."
        )
        development = (
            "If such evidence is added, the upside scenario may become more specific; "
            "currently it remains limited by missing support."
        )
    return SynthesisScenario(
        id=f"scenario_upside_{uuid4().hex}",
        direction=ScenarioDirection.UPSIDE,
        title="Conditional upside scenario",
        condition=condition,
        potential_development=development,
        confidence=_scenario_confidence(support.point_count),
        evidence_ids=support.evidence_ids,
        source_handoff_ids=support.handoff_ids,
        limitations=() if support.handoff_ids else ("No supported upside drivers were found.",),
    )


def _downside_scenario(
    buckets: dict[SynthesisSection, list[SynthesisPoint]],
) -> SynthesisScenario:
    support = _scenario_support(
        (*buckets[SynthesisSection.WEAKNESSES], *buckets[SynthesisSection.RISKS])
    )
    if support.handoff_ids:
        condition = (
            "If the cited weaknesses or risks worsen, or if missing evidence hides additional "
            "pressure, the company could develop below the currently supported base case."
        )
        development = (
            "If that condition holds, future analysis may show weaker fundamentals, higher "
            "uncertainty, or softer market confidence, but this is not a prediction."
        )
    else:
        condition = (
            "If later evidence identifies material risks, a downside case should be rebuilt "
            "from those sources."
        )
        development = (
            "If such evidence is added, the downside scenario may become more specific; "
            "currently it remains limited by missing support."
        )
    return SynthesisScenario(
        id=f"scenario_downside_{uuid4().hex}",
        direction=ScenarioDirection.DOWNSIDE,
        title="Conditional downside scenario",
        condition=condition,
        potential_development=development,
        confidence=_scenario_confidence(support.point_count),
        evidence_ids=support.evidence_ids,
        source_handoff_ids=support.handoff_ids,
        limitations=() if support.handoff_ids else ("No supported downside drivers were found.",),
    )


def _financial_target_section(section: str) -> SynthesisSection:
    if section in {"revenue", "margins", "cash_flow"}:
        return SynthesisSection.STRENGTHS
    if section in {"guidance"}:
        return SynthesisSection.OPPORTUNITIES
    if section in {"risks", "accounting_caveats"}:
        return SynthesisSection.RISKS
    if section in {"debt_liquidity"}:
        return SynthesisSection.WEAKNESSES
    return SynthesisSection.CURRENT_SITUATION


def _stock_target_section(section: str, trend: str) -> SynthesisSection:
    if section in {"volatility", "drawdown"}:
        return SynthesisSection.RISKS
    if trend == "up":
        return SynthesisSection.STRENGTHS
    if trend in {"down", "mixed"}:
        return SynthesisSection.WEAKNESSES
    return SynthesisSection.CURRENT_SITUATION


def _context_target_section(title: str, summary: str) -> SynthesisSection:
    text = f"{title} {summary}".casefold()
    risk_terms = {"risk", "pressure", "decline", "weak", "uncertain", "rate", "cost"}
    opportunity_terms = {"opportunity", "growth", "demand", "strong", "improve", "tailwind"}
    if any(term in text for term in risk_terms):
        return SynthesisSection.RISKS
    if any(term in text for term in opportunity_terms):
        return SynthesisSection.OPPORTUNITIES
    return SynthesisSection.CURRENT_SITUATION


def _status(
    handoffs: tuple[AgentHandoff, ...],
    coverage: EvidenceCoverage,
) -> SynthesisReportStatus:
    if not handoffs or coverage == EvidenceCoverage.NONE:
        return SynthesisReportStatus.NO_DATA
    if any(handoff.status != OrchestratorHandoffStatus.SUCCEEDED for handoff in handoffs):
        return SynthesisReportStatus.PARTIAL
    if coverage in {EvidenceCoverage.LIMITED, EvidenceCoverage.MODERATE}:
        return SynthesisReportStatus.PARTIAL
    return SynthesisReportStatus.COMPLETE


def _evidence_coverage(
    buckets: dict[SynthesisSection, list[SynthesisPoint]],
) -> tuple[float, EvidenceCoverage]:
    supported_sections = sum(
        1
        for section in {
            SynthesisSection.CURRENT_SITUATION,
            SynthesisSection.STRENGTHS,
            SynthesisSection.WEAKNESSES,
            SynthesisSection.OPPORTUNITIES,
            SynthesisSection.RISKS,
        }
        if any(point.evidence_ids or point.source_handoff_ids for point in buckets[section])
    )
    scenario_support = 0
    if buckets[SynthesisSection.STRENGTHS] or buckets[SynthesisSection.OPPORTUNITIES]:
        scenario_support += 1
    if buckets[SynthesisSection.WEAKNESSES] or buckets[SynthesisSection.RISKS]:
        scenario_support += 1
    ratio = (supported_sections + scenario_support) / SECTION_COUNT_FOR_COVERAGE
    if ratio >= 0.75:
        return ratio, EvidenceCoverage.STRONG
    if ratio >= 0.5:
        return ratio, EvidenceCoverage.MODERATE
    if ratio > 0:
        return ratio, EvidenceCoverage.LIMITED
    return 0.0, EvidenceCoverage.NONE


def _overall_confidence(
    buckets: dict[SynthesisSection, list[SynthesisPoint]],
) -> ConfidenceLabel:
    values = [
        _confidence_score(point.confidence) for points in buckets.values() for point in points
    ]
    supported = [value for value in values if value > 0]
    if not supported:
        return ConfidenceLabel.UNKNOWN
    average = sum(supported) / len(supported)
    if average >= 2.5:
        return ConfidenceLabel.HIGH
    if average >= 1.75:
        return ConfidenceLabel.MEDIUM
    return ConfidenceLabel.LOW


def _confidence(value: str) -> ConfidenceLabel:
    try:
        return ConfidenceLabel(value)
    except ValueError:
        return ConfidenceLabel.UNKNOWN


def _confidence_score(value: ConfidenceLabel) -> int:
    return {
        ConfidenceLabel.HIGH: 3,
        ConfidenceLabel.MEDIUM: 2,
        ConfidenceLabel.LOW: 1,
        ConfidenceLabel.UNKNOWN: 0,
    }[value]


def _scenario_confidence(point_count: int) -> ConfidenceLabel:
    if point_count >= 3:
        return ConfidenceLabel.MEDIUM
    if point_count > 0:
        return ConfidenceLabel.LOW
    return ConfidenceLabel.UNKNOWN


def _scenario_support(points: tuple[SynthesisPoint, ...]) -> _ScenarioSupport:
    return _ScenarioSupport(
        point_count=len(points),
        evidence_ids=_unique(evidence_id for point in points for evidence_id in point.evidence_ids),
        handoff_ids=_unique(
            handoff_id for point in points for handoff_id in point.source_handoff_ids
        ),
    )


def _point(
    *,
    section: SynthesisSection,
    title: str,
    summary: str,
    confidence: ConfidenceLabel,
    evidence_ids: tuple[str, ...] = (),
    source_handoff_ids: tuple[str, ...] = (),
    limitations: tuple[str, ...] = (),
) -> SynthesisPoint:
    return SynthesisPoint(
        id=f"synthesis_point_{uuid4().hex}",
        section=section,
        title=title,
        summary=summary,
        confidence=confidence,
        evidence_ids=evidence_ids,
        source_handoff_ids=source_handoff_ids,
        limitations=limitations,
    )


def _empty_buckets() -> dict[SynthesisSection, list[SynthesisPoint]]:
    return {section: [] for section in SynthesisSection}


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _list(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _payload_text(payload: Mapping[str, object], name: str, *, fallback: str) -> str:
    value = payload.get(name)
    return value.strip() if isinstance(value, str) and value.strip() else fallback


def _mapping_text(payload: Mapping[str, object] | None, name: str) -> str | None:
    if payload is None:
        return None
    value = payload.get(name)
    return value.strip() if isinstance(value, str) and value.strip() else None


def _unique(values) -> tuple[str, ...]:
    unique_values: list[str] = []
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            unique_values.append(text)
    return tuple(dict.fromkeys(unique_values))
