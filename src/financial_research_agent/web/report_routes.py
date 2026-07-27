from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from financial_research_agent.orchestration import OrchestratorRunStore
from financial_research_agent.report_exports import (
    ReportExportError,
    ReportExportErrorCode,
    ReportExportFormat,
    ReportExportService,
    ReportExportSnapshot,
    ReportExportStore,
)


def create_report_router(
    *,
    orchestrator_runs: OrchestratorRunStore,
    report_exports: ReportExportStore,
    report_export_service: ReportExportService,
) -> APIRouter:
    router = APIRouter()

    @router.post("/api/orchestrator/runs/{run_id}/exports", status_code=201)
    def create_report_export(run_id: str) -> dict[str, object]:
        run = orchestrator_runs.get(run_id)
        if run is None:
            raise HTTPException(
                status_code=404,
                detail={"error": "orchestrator_run_not_found"},
            )
        try:
            snapshot = report_export_service.export(run)
        except ReportExportError as exc:
            if exc.code == ReportExportErrorCode.SYNTHESIS_REPORT_UNAVAILABLE:
                raise HTTPException(
                    status_code=409,
                    detail={"error": exc.code.value, "message": exc.message},
                ) from exc
            raise HTTPException(
                status_code=503,
                detail={
                    "error": ReportExportErrorCode.REPORT_EXPORT_FAILED.value,
                    "message": "The local report export could not be generated.",
                },
            ) from exc
        return report_export_payload(snapshot)

    @router.get("/api/report-exports")
    def list_report_exports() -> dict[str, object]:
        try:
            return {
                "exports": [report_export_payload(snapshot) for snapshot in report_exports.list()]
            }
        except ReportExportError as exc:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": ReportExportErrorCode.REPORT_EXPORT_FAILED.value,
                    "message": "Stored report exports could not be read.",
                },
            ) from exc

    @router.get("/api/report-exports/{export_id}")
    def get_report_export(export_id: str) -> dict[str, object]:
        return report_export_payload(_report_export_or_404(report_exports, export_id))

    @router.get("/api/report-exports/{export_id}/files/{export_format}")
    def download_report_export(export_id: str, export_format: str) -> FileResponse:
        snapshot = _report_export_or_404(report_exports, export_id)
        try:
            selected_format = ReportExportFormat(export_format)
        except ValueError as exc:
            raise HTTPException(
                status_code=404,
                detail={"error": ReportExportErrorCode.REPORT_EXPORT_ARTIFACT_NOT_FOUND.value},
            ) from exc
        artifact = snapshot.artifact(selected_format)
        path = report_exports.artifact_path(snapshot, selected_format)
        if artifact is None or path is None:
            raise HTTPException(
                status_code=404,
                detail={"error": ReportExportErrorCode.REPORT_EXPORT_ARTIFACT_NOT_FOUND.value},
            )
        headers = {"X-Content-Type-Options": "nosniff"}
        if selected_format == ReportExportFormat.HTML:
            headers["Content-Security-Policy"] = (
                "default-src 'none'; style-src 'unsafe-inline'; img-src data:; "
                "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
            )
        return FileResponse(
            path,
            media_type=artifact.mime_type,
            filename=artifact.filename,
            headers=headers,
        )

    return router


def report_export_payload(snapshot: ReportExportSnapshot) -> dict[str, object]:
    export_id = snapshot.export_id
    return {
        "export": snapshot.to_dict(),
        "files": {
            artifact.format.value: (
                f"/api/report-exports/{export_id}/files/{artifact.format.value}"
            )
            for artifact in snapshot.artifacts
        },
    }


def _report_export_or_404(
    report_exports: ReportExportStore,
    export_id: str,
) -> ReportExportSnapshot:
    try:
        snapshot = report_exports.get(export_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail={"error": ReportExportErrorCode.REPORT_EXPORT_NOT_FOUND.value},
        ) from exc
    except ReportExportError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "error": ReportExportErrorCode.REPORT_EXPORT_FAILED.value,
                "message": "Stored report export could not be read.",
            },
        ) from exc
    if snapshot is None:
        raise HTTPException(
            status_code=404,
            detail={"error": ReportExportErrorCode.REPORT_EXPORT_NOT_FOUND.value},
        )
    return snapshot
