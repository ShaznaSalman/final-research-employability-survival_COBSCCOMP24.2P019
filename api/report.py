"""In-memory PDF report generation for a single prediction."""
from __future__ import annotations

from datetime import datetime, timezone
from html import escape
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.graphics.shapes import Drawing, Line, PolyLine, String
from reportlab.platypus import (Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle)

from core import schema, wording

INK = colors.HexColor("#15203a")
BLUE = colors.HexColor("#315ed4")
MUTED = colors.HexColor("#5a6682")
GRID = colors.HexColor("#dfe5ef")


def _text(value) -> str:
    return escape(str(value)).replace("\n", "<br/>")


def _profile_rows(profile: dict) -> list[list[str]]:
    rows = []
    for key, value in profile.items():
        field = schema.RAW_FIELDS_BY_NAME.get(key)
        label = field.description if field else key.replace("_", " ").capitalize()
        if key == "respondent_status":
            value = "Final-year student" if value == "final_year" else "Graduate"
        elif key == "prep_start":
            value = wording.PREP_START_LABELS.get(value, value)
        elif value in ("yes", "no"):
            value = value.capitalize()
        elif isinstance(value, float):
            value = f"{value:g}"
        rows.append([_text(label), _text(value)])
    return rows


def _survival_chart(curve: dict) -> Drawing | None:
    months, searching = curve.get("months"), curve.get("still_searching")
    if not months or len(months) != len(searching):
        return None

    width, height = 470, 205
    left, right, bottom, top = 48, 12, 32, 16
    graph_w, graph_h = width - left - right, height - bottom - top
    drawing = Drawing(width, height)
    for pct in (0, 25, 50, 75, 100):
        y = bottom + graph_h * pct / 100
        drawing.add(Line(left, y, width - right, y, strokeColor=GRID, strokeWidth=0.6))
        drawing.add(String(left - 8, y - 3, str(pct), fontSize=8, textAnchor="end", fillColor=MUTED))

    max_month = max(months)
    points = []
    for month, prob in zip(months, searching):
        points.extend((left + graph_w * month / max_month,
                       bottom + graph_h * max(0, min(1, prob))))
    drawing.add(PolyLine(points, strokeColor=BLUE, strokeWidth=2.4, fillColor=None))
    drawing.add(Line(left, bottom, width - right, bottom, strokeColor=MUTED, strokeWidth=0.8))
    drawing.add(Line(left, bottom, left, height - top, strokeColor=MUTED, strokeWidth=0.8))
    drawing.add(String(left + graph_w / 2, 7, "Months after graduation", fontSize=9,
                       textAnchor="middle", fillColor=INK))
    drawing.add(String(8, bottom + graph_h / 2, "Still searching (%)", fontSize=8,
                       textAnchor="middle", fillColor=INK, angle=90))
    drawing.add(String(left, bottom - 13, "0", fontSize=8, fillColor=MUTED))
    drawing.add(String(width - right, bottom - 13, str(max_month), fontSize=8,
                       textAnchor="end", fillColor=MUTED))
    return drawing


def _table(rows: list[list[str]], widths: list[float] | None = None) -> Table:
    table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9effc")),
        ("TEXTCOLOR", (0, 0), (-1, 0), INK),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("LEADING", (0, 0), (-1, -1), 12),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.4, GRID),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return table


def build_report(profile: dict, prediction: dict, explanation: dict, suggestions: dict, market: dict,
                 info: dict, disclaimer: str, associational_note: str, pathways: dict | None = None) -> bytes:
    """Return a complete PDF as bytes; the report is never written to disk."""
    buffer = BytesIO()
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], alignment=TA_CENTER,
                              textColor=INK, fontSize=20, leading=24, spaceAfter=5))
    styles.add(ParagraphStyle(name="ReportSubtitle", parent=styles["Normal"], alignment=TA_CENTER,
                              textColor=MUTED, fontSize=8, spaceAfter=14))
    styles.add(ParagraphStyle(name="Section", parent=styles["Heading2"], textColor=BLUE,
                              fontSize=13, leading=16, spaceBefore=12, spaceAfter=6))
    styles.add(ParagraphStyle(name="BodySmall", parent=styles["BodyText"], fontSize=8.5, leading=12))
    styles.add(ParagraphStyle(name="BulletSmall", parent=styles["BodyText"], fontSize=8.5, leading=12,
                              leftIndent=10, firstLineIndent=-7, spaceAfter=4))

    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
                            topMargin=17 * mm, bottomMargin=18 * mm, title="JobSpark results report",
                            author="JobSpark")
    story = [
        Paragraph("Your employability results", styles["ReportTitle"]),
        Paragraph("JobSpark | Generated " + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                  styles["ReportSubtitle"]),
        Paragraph("Your estimate", styles["Section"]),
        Paragraph(_text(prediction["headline"]), styles["BodyText"]),
        Spacer(1, 6),
        _table([
            [Paragraph("<b>Typical estimate</b>", styles["BodySmall"]),
             Paragraph("<b>Likely range</b>", styles["BodySmall"]),
             Paragraph("<b>Confidence</b>", styles["BodySmall"])],
            [Paragraph(_text(prediction["median"]["display"]) + " months", styles["BodySmall"]),
             Paragraph(_text(prediction["low"]["display"]) + " to "
                       + _text(prediction["high"]["display"]) + " months", styles["BodySmall"]),
             Paragraph(_text(prediction["confidence"]), styles["BodySmall"])],
        ], [55 * mm, 55 * mm, 45 * mm]),
        Spacer(1, 6),
        Paragraph(_text(prediction["confidence_reason"]), styles["BodySmall"]),
    ]

    if info.get("is_synthetic", True):
        story.extend([Spacer(1, 6), Paragraph(_text(wording.SYNTHETIC_BANNER), styles["BodySmall"])])
    for caution in prediction.get("cautions", []):
        story.extend([Spacer(1, 4), Paragraph("<b>Note:</b> " + _text(caution), styles["BodySmall"])])

    chart = _survival_chart(prediction.get("curve") or {})
    if chart:
        story.extend([Paragraph("Estimated search over time", styles["Section"]), chart,
                      Paragraph("The curve shows the estimated share of similar graduates still searching "
                                "for their first IT job.", styles["BodySmall"])])
    milestones = prediction.get("milestones") or []
    if milestones:
        story.extend([Spacer(1, 8), _table([
            [Paragraph("<b>Months after graduation</b>", styles["BodySmall"]),
             Paragraph("<b>Estimated employed</b>", styles["BodySmall"]),
             Paragraph("<b>80% uncertainty range</b>", styles["BodySmall"])]
        ] + [[str(m["months"]), f'{m["employed_pct"]}%',
              f'{m["low_pct"]}-{m["high_pct"]}%'] for m in milestones],
            [55 * mm, 50 * mm, 50 * mm])])

    story.append(Paragraph("What is helping and what may be holding you back", styles["Section"]))
    helping = explanation.get("helping") or []
    hurting = explanation.get("hurting") or []
    story.append(Paragraph("<b>Helping:</b>", styles["BodySmall"]))
    for item in helping:
        story.append(Paragraph("• " + _text(item["text"]), styles["BulletSmall"]))
    if not helping:
        story.append(Paragraph("No standout helping factors were identified.", styles["BodySmall"]))
    story.extend([Spacer(1, 4), Paragraph("<b>What may be holding you back:</b>", styles["BodySmall"])])
    for item in hurting:
        story.append(Paragraph("• " + _text(item["text"]), styles["BulletSmall"]))
    if not hurting:
        story.append(Paragraph(_text(wording.LOCAL_NO_HURTING), styles["BodySmall"]))
    story.extend([Spacer(1, 5), Paragraph(_text(explanation.get("note", "")), styles["BodySmall"])])

    items = suggestions.get("suggestions") or []
    story.append(Paragraph("Personalized guidance to explore", styles["Section"]))
    if items:
        for item in items:
            story.append(Paragraph("• " + _text(item["text"]), styles["BulletSmall"]))
        if suggestions.get("note"):
            story.append(Paragraph(_text(suggestions["note"]), styles["BodySmall"]))
    else:
        story.append(Paragraph(_text(suggestions.get("message") or
                                     "No model-based changes met the criteria for a personalized suggestion."),
                               styles["BodySmall"]))

    recommendations = (pathways or {}).get("recommendations") or []
    story.append(Paragraph("Career paths to explore", styles["Section"]))
    if recommendations:
        rows = [[Paragraph("<b>Role profile</b>", styles["BodySmall"]),
                 Paragraph("<b>Skill match</b>", styles["BodySmall"]),
                 Paragraph("<b>Skills to consider strengthening</b>", styles["BodySmall"])]]
        for role in recommendations:
            gap_labels = [gap["label"] for gap in role.get("gaps", [])[:3]]
            rows.append([
                Paragraph(f"<b>{_text(role['role'])}</b><br/>{role.get('ads', 0)} collected ads",
                          styles["BodySmall"]),
                Paragraph(f"{role['match_rating']:.1f} / 5", styles["BodySmall"]),
                Paragraph(_text(", ".join(gap_labels) if gap_labels else
                                "No priority gaps identified from the rated skills."),
                          styles["BodySmall"]),
            ])
        story.append(_table(rows, [55 * mm, 30 * mm, 79 * mm]))
        story.extend([Spacer(1, 4), Paragraph(
            "Role matches summarize overlap between your self-ratings and skills mentioned in collected ads. "
            "They are descriptive guidance, not role-specific hiring predictions.", styles["BodySmall"])])
    else:
        story.append(Paragraph(
            "No role recommendations could be calculated from the available profile and job-ad skill data.",
            styles["BodySmall"]))

    gaps = market.get("top_gaps") or []
    if gaps:
        story.extend([Paragraph("Skills employers value", styles["Section"]),
                      _table([[Paragraph("<b>Skill</b>", styles["BodySmall"]),
                               Paragraph("<b>Your rating</b>", styles["BodySmall"]),
                               Paragraph("<b>Employer rating</b>", styles["BodySmall"])]]
                             + [[_text(g["label"]),
                                 _text(f'{g["your_rating"]:.1f} / 5' if g.get("your_rating") is not None else "Not rated"),
                                 _text(f'{g["employer_importance"]:.1f} / 5'
                                       if g.get("employer_importance") is not None else "Not available")]
                                for g in gaps[:8]],
                             [80 * mm, 40 * mm, 44 * mm])])
    if market.get("note"):
        story.append(Paragraph(_text(market["note"]), styles["BodySmall"]))

    story.append(Paragraph("Profile information used", styles["Section"]))
    rows = [[Paragraph("<b>Profile item</b>", styles["BodySmall"]),
             Paragraph("<b>Your answer</b>", styles["BodySmall"])]]
    rows.extend([[Paragraph(label, styles["BodySmall"]), Paragraph(value, styles["BodySmall"])]
                 for label, value in _profile_rows(profile)])
    story.append(_table(rows, [82 * mm, 82 * mm]))

    story.extend([Spacer(1, 10), Paragraph("<b>Important:</b> " + _text(disclaimer), styles["BodySmall"]),
                  Spacer(1, 4), Paragraph(_text(associational_note), styles["BodySmall"])])

    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(18 * mm, 10 * mm, "JobSpark - decision support only")
        canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {document.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buffer.getvalue()
