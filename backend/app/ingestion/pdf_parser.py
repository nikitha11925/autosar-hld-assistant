"""
PDF parsing using PyMuPDF (fitz).

Responsible for extracting text from AUTOSAR SWS/TR-style PDFs while
preserving structure that later stages depend on:
  - Heading hierarchy (e.g. numbered section titles like "7.2.1 Ports")
  - Page numbers for every extracted text block, so downstream chunks can
    carry an accurate {doc_name, section, page_range} citation

Deliberately out of scope: OCR for scanned pages (see CLAUDE.md scope
boundaries) — this module assumes text-layer PDFs.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field

import fitz  # PyMuPDF

# AUTOSAR SWS/TR documents number sections like "7.2.1 Ports and Interfaces"
# or "A.1 Appendix". Accept a leading numeric/alpha section number followed
# by a heading-looking title (short, not ending in normal sentence punctuation).
_HEADING_NUMBER_RE = re.compile(
    r"^(?P<number>(?:[A-Z]\.)?\d+(?:\.\d+){0,5}\.?)\s+(?P<title>\S.{0,120})$"
)

# A line consisting only of a number (page footer/header artifact) — never a heading.
_NUMBER_ONLY_RE = re.compile(r"^[A-Z0-9.]+$")


@dataclass
class TextBlock:
    """One paragraph/heading-sized unit of text extracted from a page."""

    text: str
    page: int  # 1-indexed
    is_heading: bool
    heading_level: int | None  # 1 = top-level ("7"), 2 = ("7.2"), etc.
    section: str | None  # nearest enclosing heading path, e.g. "7.2.1 Ports"


@dataclass
class ParsedDocument:
    """Full result of parsing one PDF."""

    doc_name: str
    page_count: int
    blocks: list[TextBlock] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _iter_page_spans(page: fitz.Page) -> list[dict]:
    """Flatten a page's text dict into a list of line-level span records."""
    spans: list[dict] = []
    raw = page.get_text("dict")
    for block in raw.get("blocks", []):
        for line in block.get("lines", []):
            line_spans = line.get("spans", [])
            if not line_spans:
                continue
            text = "".join(s.get("text", "") for s in line_spans).strip()
            if not text:
                continue
            sizes = [s.get("size", 0.0) for s in line_spans]
            flags = [s.get("flags", 0) for s in line_spans]
            spans.append(
                {
                    "text": text,
                    "size": max(sizes) if sizes else 0.0,
                    # PyMuPDF span flags: bit 4 (value 16) = bold
                    "bold": any(f & 16 for f in flags),
                }
            )
    return spans


def _body_font_size(all_spans: list[dict]) -> float:
    """Estimate the document's body-text font size as the mode of span sizes."""
    if not all_spans:
        return 10.0
    sizes = [round(s["size"], 1) for s in all_spans]
    try:
        return statistics.mode(sizes)
    except statistics.StatisticsError:
        return statistics.median(sizes)


def _looks_like_heading_text(text: str) -> bool:
    if _NUMBER_ONLY_RE.match(text) and " " not in text:
        return False
    if len(text) > 150:
        return False
    return True


def _classify_line(text: str, size: float, bold: bool, body_size: float) -> tuple[bool, int | None]:
    """Decide whether a line is a heading, and if so at what level.

    A line is treated as a heading if it matches AUTOSAR-style numbering
    ("7.2.1 Title") AND (is visually larger than body text OR bold) —
    numbering alone is not sufficient, since body text can reference
    section numbers inline (e.g. "see section 7.2.1").
    """
    if not _looks_like_heading_text(text):
        return False, None

    match = _HEADING_NUMBER_RE.match(text)
    if not match:
        return False, None

    visually_distinct = size > body_size + 0.5 or bold
    if not visually_distinct:
        return False, None

    number = match.group("number").rstrip(".")
    level = number.count(".") + 1
    return True, level


def parse_pdf(file_path: str, doc_name: str | None = None) -> ParsedDocument:
    """Parse a PDF into heading-tagged, page-tagged text blocks.

    Args:
        file_path: path to the PDF file on disk.
        doc_name: logical document name to stamp on every block/citation.
            Defaults to the file's basename.

    Returns:
        ParsedDocument with one TextBlock per paragraph/heading line, each
        tagged with its 1-indexed page number and the nearest enclosing
        section heading (for citation purposes).
    """
    import os

    if doc_name is None:
        doc_name = os.path.basename(file_path)

    doc = fitz.open(file_path)
    try:
        if doc.is_encrypted:
            raise ValueError(f"'{doc_name}' is encrypted/password-protected and cannot be parsed")

        page_spans: list[list[dict]] = [_iter_page_spans(page) for page in doc]
        all_spans = [s for spans in page_spans for s in spans]
        body_size = _body_font_size(all_spans)

        warnings: list[str] = []
        blocks: list[TextBlock] = []
        heading_stack: list[str] = []  # e.g. ["7 Overview", "7.2 Ports"]

        for page_index, spans in enumerate(page_spans):
            page_number = page_index + 1
            if not spans:
                continue

            for span in spans:
                is_heading, level = _classify_line(
                    span["text"], span["size"], span["bold"], body_size
                )

                if is_heading:
                    # Truncate to the parent depth, padding with empty
                    # placeholders if a level was skipped (e.g. 1 -> 3),
                    # then push this heading at its own level.
                    parent = heading_stack[: level - 1]
                    parent += [""] * (level - 1 - len(parent))
                    heading_stack = parent + [span["text"]]

                section = " > ".join(h for h in heading_stack if h) or None

                blocks.append(
                    TextBlock(
                        text=span["text"],
                        page=page_number,
                        is_heading=is_heading,
                        heading_level=level if is_heading else None,
                        section=section,
                    )
                )

        if not all_spans:
            warnings.append(
                f"'{doc_name}' contains no extractable text — it may be a scanned/image-only "
                "PDF, which is out of scope (OCR is not supported)."
            )

        return ParsedDocument(
            doc_name=doc_name,
            page_count=doc.page_count,
            blocks=blocks,
            warnings=warnings,
        )
    finally:
        doc.close()
