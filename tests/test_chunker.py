from __future__ import annotations

from backend.app.ingestion.chunker import chunk_document
from backend.app.ingestion.pdf_parser import ParsedDocument, TextBlock


def _doc(blocks: list[TextBlock], doc_name: str = "doc.pdf") -> ParsedDocument:
    return ParsedDocument(doc_name=doc_name, page_count=max((b.page for b in blocks), default=0), blocks=blocks)


def test_empty_document_produces_no_chunks():
    assert chunk_document(_doc([])) == []


def test_single_section_becomes_one_chunk():
    blocks = [
        TextBlock("7 Overview", page=1, is_heading=True, heading_level=1, section="7 Overview"),
        TextBlock("Body sentence one.", page=1, is_heading=False, heading_level=None, section="7 Overview"),
        TextBlock("Body sentence two.", page=1, is_heading=False, heading_level=None, section="7 Overview"),
    ]
    chunks = chunk_document(_doc(blocks))
    assert len(chunks) == 1
    assert chunks[0].section == "7 Overview"
    assert "Body sentence one." in chunks[0].text
    assert "Body sentence two." in chunks[0].text
    assert chunks[0].page_range == "1"


def test_each_section_gets_its_own_chunk():
    blocks = [
        TextBlock("7 Overview", page=1, is_heading=True, heading_level=1, section="7 Overview"),
        TextBlock("Overview body.", page=1, is_heading=False, heading_level=None, section="7 Overview"),
        TextBlock("7.1 Ports", page=1, is_heading=True, heading_level=2, section="7 Overview > 7.1 Ports"),
        TextBlock("Ports body.", page=2, is_heading=False, heading_level=None, section="7 Overview > 7.1 Ports"),
    ]
    chunks = chunk_document(_doc(blocks))
    assert len(chunks) == 2
    assert chunks[0].section == "7 Overview"
    assert chunks[1].section == "7 Overview > 7.1 Ports"


def test_chunk_page_range_spans_start_to_end():
    blocks = [
        TextBlock("7 Overview", page=1, is_heading=True, heading_level=1, section="7 Overview"),
        TextBlock("Body on page 1.", page=1, is_heading=False, heading_level=None, section="7 Overview"),
        TextBlock("Body on page 2.", page=2, is_heading=False, heading_level=None, section="7 Overview"),
        TextBlock("Body on page 3.", page=3, is_heading=False, heading_level=None, section="7 Overview"),
    ]
    chunks = chunk_document(_doc(blocks))
    assert len(chunks) == 1
    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 3
    assert chunks[0].page_range == "1-3"


def test_long_section_falls_back_to_fixed_windows():
    long_sentence = "word " * 50  # ~250 chars -> ~62 tokens at 4 chars/token
    blocks = [
        TextBlock("7 Overview", page=1, is_heading=True, heading_level=1, section="7 Overview"),
    ] + [
        TextBlock(long_sentence, page=1, is_heading=False, heading_level=None, section="7 Overview")
        for _ in range(20)  # ~1240 tokens total, well over max_tokens=100
    ]
    chunks = chunk_document(_doc(blocks), max_tokens=100)
    assert len(chunks) > 1
    for c in chunks:
        assert c.section == "7 Overview"
        # each fallback chunk should respect (approximately) the max size
        assert len(c.text) // 4 <= 100 + 62  # allow one block's worth of overshoot


def test_chunk_metadata_carries_doc_name():
    blocks = [
        TextBlock("7 Overview", page=1, is_heading=True, heading_level=1, section="7 Overview"),
        TextBlock("Body.", page=1, is_heading=False, heading_level=None, section="7 Overview"),
    ]
    chunks = chunk_document(_doc(blocks, doc_name="my_doc.pdf"))
    assert all(c.doc_name == "my_doc.pdf" for c in chunks)


def test_blocks_with_no_section_are_grouped_together():
    """Preamble text before the first heading has section=None."""
    blocks = [
        TextBlock("Some preamble line.", page=1, is_heading=False, heading_level=None, section=None),
        TextBlock("Another preamble line.", page=1, is_heading=False, heading_level=None, section=None),
    ]
    chunks = chunk_document(_doc(blocks))
    assert len(chunks) == 1
    assert chunks[0].section is None


def test_end_to_end_with_real_pdf(sample_pdf_path):
    from backend.app.ingestion.pdf_parser import parse_pdf

    parsed = parse_pdf(sample_pdf_path)
    chunks = chunk_document(parsed)

    assert len(chunks) >= 3
    for c in chunks:
        assert c.doc_name == parsed.doc_name
        assert c.page_start >= 1
        assert c.page_end >= c.page_start
        assert c.text.strip() != ""
