from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from uuid import uuid4

from financial_research_agent.observability import RedactionPolicy
from financial_research_agent.orchestration import OrchestratedResearchRun
from financial_research_agent.synthesis import (
    NARRATIVE_PROMPT_ID,
    NARRATIVE_PROMPT_VERSION,
    synthesis_sha256,
)

from .builder import build_report_export_document
from .contracts import (
    ReportExportError,
    ReportExportErrorCode,
    ReportExportFormat,
    ReportExportSnapshot,
)
from .renderers import render_html, render_markdown, render_pdf
from .store import ReportExportStore


class ReportExportService:
    def __init__(
        self,
        *,
        store: ReportExportStore,
        redaction_policy: RedactionPolicy,
        now: Callable[[], datetime] | None = None,
        id_factory: Callable[[], str] | None = None,
        narrative_store: object | None = None,
    ) -> None:
        self.store = store
        self.redaction_policy = redaction_policy
        self._now = now or (lambda: datetime.now(UTC))
        self._id_factory = id_factory or (lambda: f"export_{uuid4().hex}")
        self.narrative_store = narrative_store

    def export(self, run: OrchestratedResearchRun) -> ReportExportSnapshot:
        generated_at = self._now()
        export_id = self._id_factory()
        narrative = self._matching_narrative(run)
        document = build_report_export_document(
            run,
            export_id=export_id,
            generated_at=generated_at,
            redaction_policy=self.redaction_policy,
            narrative_presentation=(narrative.to_dict() if narrative is not None else None),
        )
        if document is None:
            raise ReportExportError(
                ReportExportErrorCode.SYNTHESIS_REPORT_UNAVAILABLE,
                "The research run does not contain a synthesis report.",
            )
        try:
            rendered = {
                ReportExportFormat.MARKDOWN: render_markdown(document),
                ReportExportFormat.HTML: render_html(document),
                ReportExportFormat.PDF: render_pdf(document),
            }
            return self.store.save(document, rendered)
        except ReportExportError:
            raise
        except Exception as exc:
            raise ReportExportError(
                ReportExportErrorCode.REPORT_EXPORT_FAILED,
                "The report export could not be generated safely.",
            ) from exc

    def _matching_narrative(self, run: OrchestratedResearchRun):
        if self.narrative_store is None or run.agent_provider is None or run.agent_model is None:
            return None
        try:
            for handoff in reversed(run.handoffs):
                if handoff.kind.value != "synthesis":
                    continue
                report = handoff.output.get("report")
                if isinstance(report, dict):
                    return self.narrative_store.matching(
                        run_id=run.id,
                        synthesis_sha256=synthesis_sha256(report),
                        prompt_id=NARRATIVE_PROMPT_ID,
                        prompt_version=NARRATIVE_PROMPT_VERSION,
                        provider=run.agent_provider,
                        model=run.agent_model,
                    )
                return None
        except Exception:
            return None
        return None
