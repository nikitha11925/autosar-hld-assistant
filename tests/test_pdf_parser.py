from __future__ import annotations

import pytest

from backend.app.ingestion.pdf_parser import parse_pdf


def test_parses_all_pages(sample_pdf_path):
    parsed = parse_pdf(sample_pdf_path, doc_name="sample.pdf")
    assert parsed.page_count == 2
    assert parsed.doc_name == "sample.pdf"
    assert parsed.warnings == []


def test_detects_headings_and_levels(sample_pdf_path):
    parsed = parse_pdf(sample_pdf_path)
    headings = [b for b in parsed.blocks if b.is_heading]
    heading_texts = [h.text for h in headings]

    assert "7 Overview" in heading_texts
    assert "7.1 Ports" in heading_texts
    assert "7.2 Interfaces" in heading_texts
    assert "7.2.1 Client Server" in heading_texts

    levels = {h.text: h.heading_level for h in headings}
    assert levels["7 Overview"] == 1
    assert levels["7.1 Ports"] == 2
    assert levels["7.2.1 Client Server"] == 3


def test_inline_section_reference_is_not_a_heading(sample_pdf_path):
    """A body sentence mentioning 'section 7.2.1' must not be misdetected."""
    parsed = parse_pdf(sample_pdf_path)
    body_line = next(
        b for b in parsed.blocks if "client-server ports" in b.text
    )
    assert body_line.is_heading is False


def test_body_text_carries_nearest_section_path(sample_pdf_path):
    parsed = parse_pdf(sample_pdf_path)
    overview_body = next(
        b for b in parsed.blocks if "gives an overview" in b.text
    )
    assert overview_body.section == "7 Overview"

    ports_body = next(
        b for b in parsed.blocks if "client-server ports" in b.text
    )
    assert ports_body.section == "7 Overview > 7.1 Ports"


def test_page_numbers_are_1_indexed_and_correct(sample_pdf_path):
    parsed = parse_pdf(sample_pdf_path)
    assert all(b.page >= 1 for b in parsed.blocks)

    overview = next(b for b in parsed.blocks if b.text == "7 Overview")
    assert overview.page == 1

    interfaces = next(b for b in parsed.blocks if b.text == "7.2 Interfaces")
    assert interfaces.page == 2


def test_section_path_carries_across_page_boundary(sample_pdf_path):
    parsed = parse_pdf(sample_pdf_path)
    client_server_body = next(
        b for b in parsed.blocks if "define operations with arguments" in b.text
    )
    assert client_server_body.section == (
        "7 Overview > 7.2 Interfaces > 7.2.1 Client Server"
    )
    assert client_server_body.page == 2


def test_encrypted_pdf_raises(encrypted_pdf_path):
    with pytest.raises(ValueError, match="encrypted"):
        parse_pdf(encrypted_pdf_path)


def test_blank_pdf_warns_no_extractable_text(blank_pdf_path):
    parsed = parse_pdf(blank_pdf_path)
    assert parsed.blocks == []
    assert any("no extractable text" in w for w in parsed.warnings)


def test_defaults_doc_name_to_basename(sample_pdf_path):
    parsed = parse_pdf(sample_pdf_path)
    assert parsed.doc_name.endswith(".pdf")
