"""PDF transcript generation using ReportLab.

Produces a clean, professional PDF with:
  - Page 1+: AI meeting summary (if provided)
  - Remaining pages: Full diarized transcript

Layout modeled after a professional conversation transcript document:
  - Letter-size pages (8.5 x 11 in)
  - Header with title, metadata (date, duration, participants)
  - Speaker labels in bold, timestamps in grey
  - Flowing paragraph text grouped by speaker turns
  - Footer with page numbers
"""

from __future__ import annotations

import re
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    KeepTogether,
)

if TYPE_CHECKING:
    from meet.transcribe import Transcript
    from meet.summarize import MeetingSummary


# ─── Constants ──────────────────────────────────────────────────────────────

_PAGE_W, _PAGE_H = letter  # 612 x 792 pt
_MARGIN_LEFT = 0.75 * inch
_MARGIN_RIGHT = 0.75 * inch
_MARGIN_TOP = 0.75 * inch
_MARGIN_BOTTOM = 0.75 * inch

_COLOR_PRIMARY = HexColor("#1a1a2e")    # Dark navy for headings
_COLOR_SECONDARY = HexColor("#16213e")  # Slightly lighter
_COLOR_SPEAKER = HexColor("#0f3460")    # Speaker names
_COLOR_TIMESTAMP = HexColor("#888888")  # Grey timestamps
_COLOR_TEXT = HexColor("#2c2c2c")       # Body text
_COLOR_ACCENT = HexColor("#e94560")     # Accent / highlights
_COLOR_LIGHT_BG = HexColor("#f5f5f5")   # Light background for summary box


# ─── Styles ─────────────────────────────────────────────────────────────────

def _build_styles():
    """Build the paragraph styles used in the PDF."""
    styles = getSampleStyleSheet()

    s = {}

    s["title"] = ParagraphStyle(
        "PDFTitle",
        parent=styles["Title"],
        fontSize=20,
        leading=26,
        textColor=_COLOR_PRIMARY,
        alignment=TA_LEFT,
        spaceAfter=4,
    )

    s["subtitle"] = ParagraphStyle(
        "PDFSubtitle",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        textColor=_COLOR_TIMESTAMP,
        alignment=TA_LEFT,
        spaceAfter=2,
    )

    s["section_heading"] = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=14,
        leading=18,
        textColor=_COLOR_PRIMARY,
        spaceBefore=16,
        spaceAfter=8,
        borderWidth=0,
        borderPadding=0,
    )

    s["summary_heading"] = ParagraphStyle(
        "SummaryHeading",
        parent=styles["Heading3"],
        fontSize=12,
        leading=16,
        textColor=_COLOR_SECONDARY,
        spaceBefore=10,
        spaceAfter=4,
    )

    s["summary_body"] = ParagraphStyle(
        "SummaryBody",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        textColor=_COLOR_TEXT,
        alignment=TA_JUSTIFY,
        spaceAfter=4,
    )

    s["summary_bullet"] = ParagraphStyle(
        "SummaryBullet",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        textColor=_COLOR_TEXT,
        leftIndent=18,
        firstLineIndent=-12,
        spaceAfter=3,
    )

    s["speaker"] = ParagraphStyle(
        "Speaker",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        textColor=_COLOR_SPEAKER,
        fontName="Helvetica-Bold",
        spaceBefore=10,
        spaceAfter=2,
    )

    s["timestamp"] = ParagraphStyle(
        "Timestamp",
        parent=styles["Normal"],
        fontSize=8,
        leading=10,
        textColor=_COLOR_TIMESTAMP,
        spaceAfter=1,
    )

    s["transcript_text"] = ParagraphStyle(
        "TranscriptText",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        textColor=_COLOR_TEXT,
        alignment=TA_JUSTIFY,
        spaceAfter=2,
    )

    s["footer"] = ParagraphStyle(
        "Footer",
        parent=styles["Normal"],
        fontSize=8,
        leading=10,
        textColor=_COLOR_TIMESTAMP,
        alignment=TA_CENTER,
    )

    return s


# ─── Helpers ────────────────────────────────────────────────────────────────

def _fmt_time(seconds: float) -> str:
    """Format seconds as MM:SS or HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _fmt_duration(seconds: float) -> str:
    """Human-readable duration string."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    parts = []
    if h > 0:
        parts.append(f"{h}h")
    if m > 0:
        parts.append(f"{m}m")
    if s > 0 or not parts:
        parts.append(f"{s}s")
    return " ".join(parts)


def _escape_xml(text: str) -> str:
    """Escape text for ReportLab's XML-based Paragraph markup."""
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    return text


def _extract_date_from_filename(audio_file: str) -> str | None:
    """Try to extract a date from the audio filename (meeting-YYYYMMDD-HHMMSS)."""
    match = re.search(r"(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})", audio_file)
    if match:
        y, mo, d, h, mi, s = match.groups()
        try:
            dt = datetime(int(y), int(mo), int(d), int(h), int(mi), int(s))
            return dt.strftime("%B %d, %Y at %H:%M")
        except ValueError:
            pass
    return None


def _group_speaker_turns(transcript: "Transcript") -> list[dict]:
    """Group consecutive segments from the same speaker into turns.

    Returns a list of dicts:
        {"speaker": str, "start": float, "end": float, "text": str}
    """
    turns: list[dict] = []
    for seg in transcript.segments:
        speaker = seg.speaker or "UNKNOWN"
        text = seg.text.strip()
        if not text:
            continue

        # Merge with previous turn if same speaker
        if turns and turns[-1]["speaker"] == speaker:
            turns[-1]["text"] += " " + text
            turns[-1]["end"] = seg.end
        else:
            turns.append({
                "speaker": speaker,
                "start": seg.start,
                "end": seg.end,
                "text": text,
            })

    return turns


# ─── Summary Markdown → Flowables ──────────────────────────────────────────

def _summary_to_flowables(summary_md: str, styles: dict) -> list:
    """Convert the Markdown summary into ReportLab flowables."""
    flowables: list = []
    lines = summary_md.split("\n")

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # ## Heading
        if stripped.startswith("## "):
            heading_text = _escape_xml(stripped[3:].strip())
            flowables.append(
                Paragraph(heading_text, styles["summary_heading"])
            )

        # - [ ] Checkbox item or - bullet
        elif stripped.startswith("- [ ] ") or stripped.startswith("- [x] "):
            bullet_text = stripped[6:].strip()
            flowables.append(
                Paragraph(
                    f"\u2610 {_escape_xml(bullet_text)}",
                    styles["summary_bullet"],
                )
            )

        elif stripped.startswith("- **") or stripped.startswith("- "):
            bullet_text = stripped[2:].strip()
            # Convert **bold** to <b>bold</b>
            bullet_text = re.sub(
                r"\*\*(.+?)\*\*",
                r"<b>\1</b>",
                _escape_xml(bullet_text),
            )
            # Re-escape after bold conversion — but bold tags should not be escaped
            # Actually we need to escape first, then apply bold conversion
            # Let's redo: escape the raw text, then convert bold markers
            raw = stripped[2:].strip()
            # Split on bold markers and rebuild
            parts = re.split(r"(\*\*.+?\*\*)", raw)
            built = ""
            for part in parts:
                if part.startswith("**") and part.endswith("**"):
                    inner = part[2:-2]
                    built += f"<b>{_escape_xml(inner)}</b>"
                else:
                    built += _escape_xml(part)

            flowables.append(
                Paragraph(f"\u2022 {built}", styles["summary_bullet"])
            )

        else:
            # Regular paragraph text
            flowables.append(
                Paragraph(_escape_xml(stripped), styles["summary_body"])
            )

    return flowables


# ─── Page template with header/footer ──────────────────────────────────────

class _PDFDocTemplate(BaseDocTemplate):
    """Custom doc template with header line and page-number footer."""

    def __init__(self, filename, title: str = "", **kwargs):
        super().__init__(filename, **kwargs)
        self._pdf_title = title

        frame = Frame(
            _MARGIN_LEFT,
            _MARGIN_BOTTOM + 0.3 * inch,  # room for footer
            _PAGE_W - _MARGIN_LEFT - _MARGIN_RIGHT,
            _PAGE_H - _MARGIN_TOP - _MARGIN_BOTTOM - 0.3 * inch,
            id="main",
        )
        self.addPageTemplates([
            PageTemplate(id="main", frames=[frame], onPage=self._draw_page),
        ])

    def _draw_page(self, canvas, doc):
        """Draw header line and footer on every page."""
        canvas.saveState()

        # Footer: page number
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(_COLOR_TIMESTAMP)
        page_text = f"Page {doc.page}"
        canvas.drawCentredString(_PAGE_W / 2, _MARGIN_BOTTOM * 0.5, page_text)

        # Thin line at top of content area
        y_line = _PAGE_H - _MARGIN_TOP + 4
        canvas.setStrokeColor(HexColor("#dddddd"))
        canvas.setLineWidth(0.5)
        canvas.line(_MARGIN_LEFT, y_line, _PAGE_W - _MARGIN_RIGHT, y_line)

        canvas.restoreState()


# ─── Public API ─────────────────────────────────────────────────────────────

def generate_pdf(
    transcript: "Transcript",
    output_path: str | Path,
    summary: "MeetingSummary | None" = None,
    title: str = "Meeting Transcript",
) -> Path:
    """Generate a PDF transcript document.

    Args:
        transcript: The Transcript object with segments and speaker info.
        output_path: Where to write the PDF file.
        summary: Optional AI-generated meeting summary to include as
            the first section.
        title: Document title shown on the first page.

    Returns:
        Path to the generated PDF file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    styles = _build_styles()
    story: list = []

    # ── Title block ──
    story.append(Paragraph(title, styles["title"]))

    # Metadata line(s)
    meta_parts: list[str] = []

    date_str = _extract_date_from_filename(transcript.audio_file)
    if date_str:
        meta_parts.append(f"Date: {date_str}")

    if transcript.duration and transcript.duration > 0:
        meta_parts.append(f"Duration: {_fmt_duration(transcript.duration)}")

    if transcript.speakers:
        speaker_names = ", ".join(s.label or s.id for s in transcript.speakers)
        meta_parts.append(f"Participants: {speaker_names}")

    meta_parts.append("Recording source: AI transcription (meetscribe)")

    for part in meta_parts:
        story.append(Paragraph(_escape_xml(part), styles["subtitle"]))

    story.append(Spacer(1, 12))

    # ── Summary section (if provided) ──
    if summary:
        story.append(Paragraph("AI Meeting Summary", styles["section_heading"]))
        story.append(
            Paragraph(
                f"<i>Generated by {_escape_xml(summary.model)} "
                f"in {summary.elapsed_seconds:.0f}s</i>",
                styles["subtitle"],
            )
        )
        story.append(Spacer(1, 4))

        summary_flowables = _summary_to_flowables(summary.markdown, styles)
        story.extend(summary_flowables)

        story.append(Spacer(1, 16))

    # ── Transcript section ──
    story.append(Paragraph("Full Transcript", styles["section_heading"]))
    story.append(Spacer(1, 4))

    turns = _group_speaker_turns(transcript)

    for turn in turns:
        speaker = turn["speaker"]
        start_ts = _fmt_time(turn["start"])

        # Speaker + timestamp header
        header = (
            f'<font color="{_COLOR_SPEAKER}">'
            f'<b>{_escape_xml(speaker)}</b></font>'
            f'  <font color="{_COLOR_TIMESTAMP}" size="8">{start_ts}</font>'
        )
        story.append(Paragraph(header, styles["speaker"]))

        # Transcript text
        text = _escape_xml(turn["text"])
        story.append(Paragraph(text, styles["transcript_text"]))

    # ── Build PDF ──
    doc = _PDFDocTemplate(
        str(output_path),
        title=title,
        pagesize=letter,
        leftMargin=_MARGIN_LEFT,
        rightMargin=_MARGIN_RIGHT,
        topMargin=_MARGIN_TOP,
        bottomMargin=_MARGIN_BOTTOM,
    )

    doc.build(story)
    return output_path
