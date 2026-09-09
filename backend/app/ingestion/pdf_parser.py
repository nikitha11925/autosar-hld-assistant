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
