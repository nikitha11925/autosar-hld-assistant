"""Shared pytest fixtures: builds a synthetic AUTOSAR-style PDF on disk."""

from __future__ import annotations

import fitz
import pytest

BODY_SIZE = 10
HEADING_SIZE = 14


def _add_heading(page: fitz.Page, point: tuple[float, float], text: str) -> float:
    page.insert_text(point, text, fontsize=HEADING_SIZE, fontname="hebo")
    return point[1] + HEADING_SIZE + 6


def _add_body(page: fitz.Page, point: tuple[float, float], text: str) -> float:
    page.insert_text(point, text, fontsize=BODY_SIZE, fontname="helv")
    return point[1] + BODY_SIZE + 4


@pytest.fixture()
def sample_pdf_path(tmp_path):
    """A 2-page PDF with a nested heading structure and body paragraphs.

    Structure:
      Page 1:
        7 Overview            (heading, level 1)
          Body paragraph about overview.
        7.1 Ports              (heading, level 2)
          Body paragraph about ports, referencing "section 7.2.1" inline
          (must NOT be misdetected as a heading).
      Page 2:
        7.2 Interfaces         (heading, level 2)
        7.2.1 Client Server    (heading, level 3)
          Body paragraph about client-server interfaces.
    """
    doc = fitz.open()

    page1 = doc.new_page()
    y = 72
    y = _add_heading(page1, (72, y), "7 Overview")
    y = _add_body(page1, (72, y), "This section gives an overview of the module.")
    y = _add_heading(page1, (72, y + 10), "7.1 Ports")
    y = _add_body(
        page1,
        (72, y),
        "Ports are described in detail, see section 7.2.1 for client-server ports.",
    )

    page2 = doc.new_page()
    y = 72
    y = _add_heading(page2, (72, y), "7.2 Interfaces")
    y = _add_heading(page2, (72, y + 10), "7.2.1 Client Server")
    y = _add_body(
        page2, (72, y), "Client-server interfaces define operations with arguments."
    )

    pdf_path = tmp_path / "sample.pdf"
    doc.save(str(pdf_path))
    doc.close()
    return str(pdf_path)


@pytest.fixture()
def encrypted_pdf_path(tmp_path):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Secret content", fontsize=BODY_SIZE, fontname="helv")
    pdf_path = tmp_path / "encrypted.pdf"
    doc.save(str(pdf_path), encryption=fitz.PDF_ENCRYPT_AES_256, owner_pw="owner", user_pw="user")
    doc.close()
    return str(pdf_path)


@pytest.fixture()
def blank_pdf_path(tmp_path):
    doc = fitz.open()
    doc.new_page()
    pdf_path = tmp_path / "blank.pdf"
    doc.save(str(pdf_path))
    doc.close()
    return str(pdf_path)
