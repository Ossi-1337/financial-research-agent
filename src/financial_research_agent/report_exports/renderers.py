from __future__ import annotations

import html
import io
import re
from importlib.resources import files
from threading import Lock

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

from .contracts import ReportExportDocument, ReportExportPoint, ReportExportScenario

_FONT_LOCK = Lock()
_REGULAR_FONT = "FRA-NotoSans"
_BOLD_FONT = "FRA-NotoSans-Bold"
_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|>])")


def render_markdown(document: ReportExportDocument) -> bytes:
    lines = [
        f"# {_md(document.company_name or document.ticker or 'Financial Research Report')}",
        "",
        _metadata_markdown(document),
        "",
        f"> **Research only:** {_md(document.disclaimer)}",
        "",
    ]
    for title, points in _sections(document):
        lines.extend(_markdown_points(title, points))
    lines.extend(_markdown_scenario("Upside Scenario", document.upside_scenario))
    lines.extend(_markdown_scenario("Downside Scenario", document.downside_scenario))
    lines.extend(_markdown_list("Warnings", document.warnings))
    lines.extend(_markdown_list("Limitations", document.limitations))
    lines.extend(["## Sources", ""])
    if not document.sources:
        lines.extend(["No source references were available.", ""])
    for source in document.sources:
        status = "resolved" if source.resolved else "unresolved"
        lines.append(f"### {_md(source.marker)} {_md(source.source_name or status.title())}")
        lines.append("")
        lines.append(f"- Status: {_md(status)}")
        lines.append(f"- Evidence IDs: {_md(', '.join(source.evidence_ids))}")
        if source.source_url:
            lines.append(f"- URL: {_md(source.source_url)}")
        if source.source_date:
            lines.append(f"- Source date: {_md(source.source_date)}")
        if source.retrieved_at:
            lines.append(f"- Retrieved: {_md(source.retrieved_at)}")
        if source.section:
            lines.append(f"- Section: {_md(source.section)}")
        if source.quote:
            lines.append(f"- Quote: “{_md(source.quote)}”")
        lines.append("")
    return "\n".join(lines).encode("utf-8")


def render_html(document: ReportExportDocument) -> bytes:
    title = _html(document.company_name or document.ticker or "Financial Research Report")
    body: list[str] = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        f"<title>{title} - Financial Research Report</title>",
        "<style>",
        _embedded_css(),
        "</style></head><body>",
        '<main class="report">',
        f"<h1>{title}</h1>",
        _metadata_html(document),
        f'<p class="disclaimer"><strong>Research only:</strong> {_html(document.disclaimer)}</p>',
    ]
    for section_title, points in _sections(document):
        body.append(_html_points(section_title, points))
    body.append(_html_scenario("Upside Scenario", document.upside_scenario))
    body.append(_html_scenario("Downside Scenario", document.downside_scenario))
    body.append(_html_list("Warnings", document.warnings))
    body.append(_html_list("Limitations", document.limitations))
    body.append("<h2>Sources</h2>")
    if not document.sources:
        body.append("<p>No source references were available.</p>")
    for source in document.sources:
        status = "resolved" if source.resolved else "unresolved"
        body.extend(
            (
                '<section class="source">',
                f"<h3>{_html(source.marker)} {_html(source.source_name or status.title())}</h3>",
                "<dl>",
                f"<dt>Status</dt><dd>{_html(status)}</dd>",
                f"<dt>Evidence IDs</dt><dd>{_html(', '.join(source.evidence_ids))}</dd>",
            )
        )
        for label, value in (
            ("URL", source.source_url),
            ("Source date", source.source_date),
            ("Retrieved", source.retrieved_at),
            ("Section", source.section),
            ("Quote", source.quote),
        ):
            if value:
                body.append(f"<dt>{label}</dt><dd>{_html(value)}</dd>")
        body.extend(("</dl>", "</section>"))
    body.extend(
        (
            "<footer><p>Generated deterministically from persisted research data. "
            f"Export ID: {_html(document.export_id)}</p></footer>",
            "</main></body></html>",
        )
    )
    return "".join(body).encode("utf-8")


def render_pdf(document: ReportExportDocument) -> bytes:
    _register_fonts()
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=document.company_name or document.ticker or "Financial Research Report",
        author="Financial Research Agent",
    )
    styles = _pdf_styles()
    story = [
        Spacer(1, 32 * mm),
        Paragraph(
            _html(document.company_name or document.ticker or "Financial Research Report"),
            styles["Title"],
        ),
        Spacer(1, 6 * mm),
        Paragraph("Financial Research Report", styles["Subtitle"]),
        Spacer(1, 12 * mm),
        Paragraph(_metadata_pdf(document), styles["Meta"]),
        Spacer(1, 10 * mm),
        Paragraph(f"<b>Research only:</b> {_html(document.disclaimer)}", styles["Notice"]),
        PageBreak(),
    ]
    for title, points in _sections(document):
        story.append(Paragraph(_html(title), styles["Heading2"]))
        if not points:
            story.append(Paragraph("No findings available.", styles["Muted"]))
        for point in points:
            story.extend(_pdf_point(point, styles))
    for title, scenario in (
        ("Upside Scenario", document.upside_scenario),
        ("Downside Scenario", document.downside_scenario),
    ):
        story.append(Paragraph(title, styles["Heading2"]))
        story.extend(_pdf_scenario(scenario, styles))
    _pdf_text_list(story, "Warnings", document.warnings, styles)
    _pdf_text_list(story, "Limitations", document.limitations, styles)
    story.append(PageBreak())
    story.append(Paragraph("Source Appendix", styles["Heading1"]))
    if not document.sources:
        story.append(Paragraph("No source references were available.", styles["Muted"]))
    for source in document.sources:
        status = "resolved" if source.resolved else "unresolved"
        story.append(
            Paragraph(
                f"<b>{_html(source.marker)} {_html(source.source_name or status.title())}</b>",
                styles["Body"],
            )
        )
        details = [
            f"Status: {status}",
            f"Evidence IDs: {', '.join(source.evidence_ids)}",
            *(
                f"{label}: {value}"
                for label, value in (
                    ("URL", source.source_url),
                    ("Source date", source.source_date),
                    ("Retrieved", source.retrieved_at),
                    ("Section", source.section),
                    ("Quote", source.quote),
                )
                if value
            ),
        ]
        story.append(Paragraph("<br/>".join(_html(item) for item in details), styles["Source"]))
        story.append(Spacer(1, 3 * mm))
    doc.build(
        story,
        onFirstPage=lambda canvas, _: _pdf_footer(canvas, document.export_id),
        onLaterPages=lambda canvas, _: _pdf_footer(canvas, document.export_id),
    )
    return buffer.getvalue()


def _sections(
    document: ReportExportDocument,
) -> tuple[tuple[str, tuple[ReportExportPoint, ...]], ...]:
    return (
        ("Current Situation", document.current_situation),
        ("Strengths", document.strengths),
        ("Weaknesses", document.weaknesses),
        ("Opportunities", document.opportunities),
        ("Risks", document.risks),
        ("Unknowns", document.unknowns),
    )


def _metadata_markdown(document: ReportExportDocument) -> str:
    values = (
        ("Company", document.company_name or "Not available"),
        ("Ticker", document.ticker or "Not available"),
        ("Query", document.query),
        ("Run ID", document.run_id),
        ("Run status", document.run_status),
        ("Report status", document.report_status),
        ("Report created", document.report_created_at.isoformat()),
        ("Export created", document.generated_at.isoformat()),
        ("Confidence", document.overall_confidence),
        (
            "Evidence coverage",
            f"{document.evidence_coverage} ({document.evidence_coverage_ratio:.0%})",
        ),
        ("Generation", document.generation_method),
        ("LLM provider/model", f"{document.llm_provider} / {document.llm_model}"),
    )
    return "\n".join(f"- **{label}:** {_md(value)}" for label, value in values)


def _markdown_points(title: str, points: tuple[ReportExportPoint, ...]) -> list[str]:
    lines = [f"## {title}", ""]
    if not points:
        return [*lines, "No findings available.", ""]
    for point in points:
        markers = f" {' '.join(point.source_markers)}" if point.source_markers else ""
        lines.extend(
            (
                f"### {_md(point.title)}{markers}",
                "",
                _md(point.summary),
                "",
                f"Confidence: {_md(point.confidence)}",
                "",
            )
        )
        if point.limitations:
            lines.append(f"Limitations: {_md('; '.join(point.limitations))}")
            lines.append("")
    return lines


def _markdown_scenario(title: str, scenario: ReportExportScenario) -> list[str]:
    markers = f" {' '.join(scenario.source_markers)}" if scenario.source_markers else ""
    return [
        f"## {title}",
        "",
        f"### {_md(scenario.title)}{markers}",
        "",
        f"**Condition:** {_md(scenario.condition)}",
        "",
        f"**Potential development:** {_md(scenario.potential_development)}",
        "",
        f"**Confidence:** {_md(scenario.confidence)}",
        "",
        *(
            [f"**Limitations:** {_md('; '.join(scenario.limitations))}", ""]
            if scenario.limitations
            else []
        ),
    ]


def _markdown_list(title: str, values: tuple[str, ...]) -> list[str]:
    lines = [f"## {title}", ""]
    lines.extend(f"- {_md(value)}" for value in values)
    if not values:
        lines.append("None recorded.")
    lines.append("")
    return lines


def _metadata_html(document: ReportExportDocument) -> str:
    values = (
        ("Company", document.company_name or "Not available"),
        ("Ticker", document.ticker or "Not available"),
        ("Query", document.query),
        ("Run ID", document.run_id),
        ("Run status", document.run_status),
        ("Report status", document.report_status),
        ("Report created", document.report_created_at.isoformat()),
        ("Export created", document.generated_at.isoformat()),
        ("Confidence", document.overall_confidence),
        (
            "Evidence coverage",
            f"{document.evidence_coverage} ({document.evidence_coverage_ratio:.0%})",
        ),
        ("Generation", document.generation_method),
        ("LLM provider/model", f"{document.llm_provider} / {document.llm_model}"),
    )
    return (
        '<dl class="metadata">'
        + "".join(f"<dt>{label}</dt><dd>{_html(value)}</dd>" for label, value in values)
        + "</dl>"
    )


def _html_points(title: str, points: tuple[ReportExportPoint, ...]) -> str:
    body = [f"<h2>{_html(title)}</h2>"]
    if not points:
        body.append("<p>No findings available.</p>")
    for point in points:
        markers = " ".join(_html(marker) for marker in point.source_markers)
        body.extend(
            (
                '<section class="finding">',
                f'<h3>{_html(point.title)} <span class="markers">{markers}</span></h3>',
                f"<p>{_html(point.summary)}</p>",
                f'<p class="meta">Confidence: {_html(point.confidence)}</p>',
            )
        )
        if point.limitations:
            body.append(f'<p class="meta">Limitations: {_html("; ".join(point.limitations))}</p>')
        body.append("</section>")
    return "".join(body)


def _html_scenario(title: str, scenario: ReportExportScenario) -> str:
    markers = " ".join(_html(marker) for marker in scenario.source_markers)
    limitation = (
        f'<p class="meta">Limitations: {_html("; ".join(scenario.limitations))}</p>'
        if scenario.limitations
        else ""
    )
    return (
        f'<h2>{_html(title)}</h2><section class="scenario">'
        f'<h3>{_html(scenario.title)} <span class="markers">{markers}</span></h3>'
        f"<p><strong>Condition:</strong> {_html(scenario.condition)}</p>"
        f"<p><strong>Potential development:</strong> "
        f"{_html(scenario.potential_development)}</p>"
        f'<p class="meta">Confidence: {_html(scenario.confidence)}</p>'
        f"{limitation}</section>"
    )


def _html_list(title: str, values: tuple[str, ...]) -> str:
    items = "".join(f"<li>{_html(value)}</li>" for value in values)
    return f"<h2>{title}</h2><ul>{items or '<li>None recorded.</li>'}</ul>"


def _embedded_css() -> str:
    return """
@page{size:A4;margin:18mm}*{box-sizing:border-box}body{margin:0;background:#f4f7fb;
color:#172033;font:15px/1.55 "Noto Sans",Arial,sans-serif}.report{max-width:920px;
margin:28px auto;padding:42px;background:#fff;border:1px solid #dce3ed}h1{font-size:30px}
h2{margin-top:30px;padding-bottom:7px;border-bottom:1px solid #dce3ed;font-size:19px}
h3{font-size:15px}.metadata{display:grid;grid-template-columns:170px 1fr;gap:5px 14px}
.metadata dt{font-weight:700}.metadata dd{margin:0;overflow-wrap:anywhere}.disclaimer{padding:14px;
border-left:4px solid #2563eb;background:#eff6ff}.finding,.scenario,.source{break-inside:avoid}
.meta,.markers,footer{color:#5b6678;font-size:13px}.source dl{display:grid;
grid-template-columns:120px 1fr;gap:4px 12px}.source dt{font-weight:700}.source dd{margin:0;
overflow-wrap:anywhere}footer{margin-top:36px;padding-top:12px;border-top:1px solid #dce3ed}
@media(max-width:680px){.report{margin:0;padding:22px;border:0}.metadata,.source dl{
grid-template-columns:1fr}.metadata dd,.source dd{margin-bottom:6px}}@media print{
body{background:#fff}
.report{margin:0;padding:0;border:0}}"""


def _register_fonts() -> None:
    with _FONT_LOCK:
        if _REGULAR_FONT in pdfmetrics.getRegisteredFontNames():
            return
        assets = files("financial_research_agent.report_exports").joinpath("assets")
        with assets.joinpath("NotoSans-Regular.ttf").open("rb") as regular_file:
            pdfmetrics.registerFont(TTFont(_REGULAR_FONT, io.BytesIO(regular_file.read())))
        with assets.joinpath("NotoSans-Bold.ttf").open("rb") as bold_file:
            pdfmetrics.registerFont(TTFont(_BOLD_FONT, io.BytesIO(bold_file.read())))


def _pdf_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "Title": ParagraphStyle(
            "ExportTitle",
            parent=base["Title"],
            fontName=_BOLD_FONT,
            fontSize=26,
            leading=32,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#172033"),
        ),
        "Subtitle": ParagraphStyle(
            "ExportSubtitle",
            parent=base["Normal"],
            fontName=_REGULAR_FONT,
            fontSize=13,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#5b6678"),
        ),
        "Heading1": ParagraphStyle(
            "ExportHeading1",
            parent=base["Heading1"],
            fontName=_BOLD_FONT,
            fontSize=20,
            leading=25,
            spaceAfter=10,
        ),
        "Heading2": ParagraphStyle(
            "ExportHeading2",
            parent=base["Heading2"],
            fontName=_BOLD_FONT,
            fontSize=15,
            leading=20,
            spaceBefore=14,
            spaceAfter=7,
            textColor=colors.HexColor("#172033"),
        ),
        "Body": ParagraphStyle(
            "ExportBody",
            parent=base["BodyText"],
            fontName=_REGULAR_FONT,
            fontSize=9.5,
            leading=14,
            spaceAfter=5,
        ),
        "Meta": ParagraphStyle(
            "ExportMeta",
            parent=base["BodyText"],
            fontName=_REGULAR_FONT,
            fontSize=9,
            leading=14,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#5b6678"),
        ),
        "Notice": ParagraphStyle(
            "ExportNotice",
            parent=base["BodyText"],
            fontName=_REGULAR_FONT,
            fontSize=9.5,
            leading=14,
            backColor=colors.HexColor("#eff6ff"),
            borderColor=colors.HexColor("#2563eb"),
            borderWidth=0.5,
            borderPadding=9,
        ),
        "Muted": ParagraphStyle(
            "ExportMuted",
            parent=base["BodyText"],
            fontName=_REGULAR_FONT,
            fontSize=9,
            textColor=colors.HexColor("#5b6678"),
        ),
        "Source": ParagraphStyle(
            "ExportSource",
            parent=base["BodyText"],
            fontName=_REGULAR_FONT,
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#364152"),
        ),
    }


def _metadata_pdf(document: ReportExportDocument) -> str:
    return "<br/>".join(
        _html(line)
        for line in (
            f"Company: {document.company_name or 'Not available'}",
            f"Ticker: {document.ticker or 'Not available'}",
            f"Run ID: {document.run_id}",
            f"Report status: {document.report_status}",
            f"Report created: {document.report_created_at.isoformat()}",
            f"Export created: {document.generated_at.isoformat()}",
            f"Confidence: {document.overall_confidence}",
            f"Evidence coverage: {document.evidence_coverage} "
            f"({document.evidence_coverage_ratio:.0%})",
            f"Generation: {document.generation_method}",
            f"LLM provider/model: {document.llm_provider} / {document.llm_model}",
        )
    )


def _pdf_point(point: ReportExportPoint, styles: dict[str, ParagraphStyle]) -> list[object]:
    markers = f" {' '.join(point.source_markers)}" if point.source_markers else ""
    values: list[object] = [
        Paragraph(f"<b>{_html(point.title)}</b>{_html(markers)}", styles["Body"]),
        Paragraph(_html(point.summary), styles["Body"]),
        Paragraph(f"Confidence: {_html(point.confidence)}", styles["Muted"]),
    ]
    if point.limitations:
        values.append(
            Paragraph(f"Limitations: {_html('; '.join(point.limitations))}", styles["Muted"])
        )
    values.append(Spacer(1, 2 * mm))
    return values


def _pdf_scenario(
    scenario: ReportExportScenario,
    styles: dict[str, ParagraphStyle],
) -> list[object]:
    markers = f" {' '.join(scenario.source_markers)}" if scenario.source_markers else ""
    values: list[object] = [
        Paragraph(f"<b>{_html(scenario.title)}</b>{_html(markers)}", styles["Body"]),
        Paragraph(f"<b>Condition:</b> {_html(scenario.condition)}", styles["Body"]),
        Paragraph(
            f"<b>Potential development:</b> {_html(scenario.potential_development)}",
            styles["Body"],
        ),
        Paragraph(f"Confidence: {_html(scenario.confidence)}", styles["Muted"]),
    ]
    if scenario.limitations:
        values.append(
            Paragraph(f"Limitations: {_html('; '.join(scenario.limitations))}", styles["Muted"])
        )
    return values


def _pdf_text_list(
    story: list[object],
    title: str,
    values: tuple[str, ...],
    styles: dict[str, ParagraphStyle],
) -> None:
    story.append(Paragraph(title, styles["Heading2"]))
    for value in values or ("None recorded.",):
        story.append(Paragraph(f"• {_html(value)}", styles["Body"]))


def _pdf_footer(canvas: object, export_id: str) -> None:
    canvas.saveState()
    canvas.setFont(_REGULAR_FONT, 7)
    canvas.setFillColor(colors.HexColor("#6b7280"))
    canvas.drawString(18 * mm, 9 * mm, export_id)
    canvas.drawRightString(192 * mm, 9 * mm, f"Page {canvas.getPageNumber()}")
    canvas.restoreState()


def _md(value: object) -> str:
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return _MD_SPECIAL.sub(r"\\\1", text)


def _html(value: object) -> str:
    return html.escape(str(value), quote=True)
