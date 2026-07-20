from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from uuid import uuid4

from financial_research_agent.observability import RedactionPolicy
from financial_research_agent.orchestration import OrchestratedResearchRun

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
    ) -> None:
        self.store = store
        self.redaction_policy = redaction_policy
        self._now = now or (lambda: datetime.now(UTC))
        self._id_factory = id_factory or (lambda: f"export_{uuid4().hex}")

    def export(self, run: OrchestratedResearchRun) -> ReportExportSnapshot:
        generated_at = self._now()
        export_id = self._id_factory()
        document = build_report_export_document(
            run,
            export_id=export_id,
            generated_at=generated_at,
            redaction_policy=self.redaction_policy,
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
