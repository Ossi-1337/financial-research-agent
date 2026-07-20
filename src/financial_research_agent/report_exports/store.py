from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections.abc import Mapping
from pathlib import Path
from uuid import uuid4

from financial_research_agent.settings import Settings

from .contracts import (
    ReportExportArtifact,
    ReportExportDocument,
    ReportExportError,
    ReportExportErrorCode,
    ReportExportFormat,
    ReportExportSnapshot,
)


class ReportExportStore:
    def __init__(self, *, root: Path) -> None:
        self.root = root

    @classmethod
    def from_settings(cls, settings: Settings) -> ReportExportStore:
        return cls(root=settings.local_paths.data_dir / "exports")

    def save(
        self,
        document: ReportExportDocument,
        rendered: Mapping[ReportExportFormat, bytes],
    ) -> ReportExportSnapshot:
        if set(rendered) != set(ReportExportFormat):
            raise ValueError("rendered artifacts must include markdown, html, and pdf")
        self.root.mkdir(parents=True, exist_ok=True)
        destination = self.root / document.export_id
        if destination.exists():
            raise FileExistsError("report export snapshot already exists")
        temp = self.root / f".{document.export_id}.tmp-{uuid4().hex}"
        temp.mkdir()
        try:
            basename = _generated_basename(document)
            artifacts: list[ReportExportArtifact] = []
            for export_format in ReportExportFormat:
                content = bytes(rendered[export_format])
                if not content:
                    raise ValueError(f"{export_format.value} artifact must not be empty")
                filename = f"{basename}{export_format.suffix}"
                (temp / filename).write_bytes(content)
                artifacts.append(
                    ReportExportArtifact(
                        format=export_format,
                        filename=filename,
                        mime_type=export_format.mime_type,
                        byte_size=len(content),
                        sha256=hashlib.sha256(content).hexdigest(),
                    )
                )
            snapshot = ReportExportSnapshot(
                export_id=document.export_id,
                run_id=document.run_id,
                report_id=document.report_id,
                created_at=document.generated_at,
                company_name=document.company_name,
                company_id=document.company_id,
                ticker=document.ticker,
                security_id=document.security_id,
                artifacts=tuple(artifacts),
            )
            (temp / "manifest.json").write_text(
                json.dumps(snapshot.to_dict(), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            temp.replace(destination)
            return snapshot
        except Exception:
            if temp.exists():
                shutil.rmtree(temp)
            raise

    def get(self, export_id: str) -> ReportExportSnapshot | None:
        export_dir = self._snapshot_dir(export_id)
        manifest_path = export_dir / "manifest.json"
        if not manifest_path.is_file():
            return None
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("manifest must be an object")
            return ReportExportSnapshot.from_dict(payload)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ReportExportError(
                ReportExportErrorCode.REPORT_EXPORT_FAILED,
                "Stored report export manifest could not be read.",
            ) from exc

    def list(self) -> tuple[ReportExportSnapshot, ...]:
        if not self.root.exists():
            return ()
        snapshots = [
            snapshot
            for child in self.root.iterdir()
            if child.is_dir() and not child.name.startswith(".")
            if (snapshot := self.get(child.name)) is not None
        ]
        return tuple(sorted(snapshots, key=lambda item: item.created_at, reverse=True))

    def artifact_path(
        self,
        snapshot: ReportExportSnapshot,
        export_format: ReportExportFormat,
    ) -> Path | None:
        artifact = snapshot.artifact(export_format)
        if artifact is None:
            return None
        root = self.root.resolve()
        path = (self._snapshot_dir(snapshot.export_id) / artifact.filename).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            return None
        return path

    def _snapshot_dir(self, export_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,159}", export_id):
            raise ValueError("invalid report export id")
        root = self.root.resolve()
        path = (root / export_id).resolve()
        if not path.is_relative_to(root) or path == root:
            raise ValueError("invalid report export path")
        return path


def _generated_basename(document: ReportExportDocument) -> str:
    subject = document.ticker or document.company_name or "research-report"
    slug = re.sub(r"[^a-z0-9]+", "-", subject.casefold()).strip("-")[:48]
    slug = slug or "research-report"
    date_part = document.report_created_at.date().isoformat()
    return f"{slug}-{date_part}-{document.export_id[-8:]}"
