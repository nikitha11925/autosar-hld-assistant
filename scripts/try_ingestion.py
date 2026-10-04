"""
Manual smoke-test script for the ingestion pipeline.

Usage:
    python scripts/try_ingestion.py path/to/your.pdf

Prints detected headings, section paths, and the resulting chunks so you
can eyeball whether parsing/chunking looks right on a real document before
wiring up embeddings.
"""

from __future__ import annotations

import sys

from backend.app.ingestion.chunker import chunk_document
from backend.app.ingestion.pdf_parser import parse_pdf


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python scripts/try_ingestion.py path/to/your.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]
    parsed = parse_pdf(pdf_path)

    print(f"Doc: {parsed.doc_name}  ({parsed.page_count} pages)")
    if parsed.warnings:
        print("Warnings:")
        for w in parsed.warnings:
            print(f"  - {w}")

    headings = [b for b in parsed.blocks if b.is_heading]
    print(f"\nDetected {len(headings)} headings:")
    for h in headings[:40]:
        print(f"  [p{h.page}] {'  ' * (h.heading_level - 1)}{h.text}")
    if len(headings) > 40:
        print(f"  ... and {len(headings) - 40} more")

    chunks = chunk_document(parsed)
    print(f"\nProduced {len(chunks)} chunks:")
    for c in chunks[:10]:
        preview = c.text[:100].replace("\n", " ")
        print(f"  [p{c.page_range}] section={c.section!r}\n    {preview}...")
    if len(chunks) > 10:
        print(f"  ... and {len(chunks) - 10} more")


if __name__ == "__main__":
    main()
