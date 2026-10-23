"""
Dataset exploration (EDA) for the ingested AUTOSAR specification PDF.

Runs the actual parsing/chunking pipeline against data/raw_pdfs/*.pdf and
reports real, measured statistics:
  - raw document structure (pages, text blocks, detected headings)
  - the heading-detection false-positive rate (numbered list items and
    page-footer artifacts that get misclassified as headings)
  - chunk-size distribution (chars and approximate tokens)
  - section-coverage of chunks (how many chunks have no heading context)

Also saves a small set of plots to eval/results/ so the distributions can
be inspected visually instead of only as printed numbers.

Usage:
    python eval/explore_dataset.py [path/to/doc.pdf]
    (defaults to data/raw_pdfs/AUTOSAR_SWS_OS.pdf if no path given)
"""

from __future__ import annotations

import re
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless; this script only writes files, never opens a window
import matplotlib.pyplot as plt

from backend.app.ingestion.chunker import chunk_document
from backend.app.ingestion.pdf_parser import parse_pdf

RESULTS_DIR = Path(__file__).parent / "results"

# Heuristic to flag headings that are actually numbered list items in body
# prose ("1. More than one core on the same piece of silicon.") or
# page-number running-footer artifacts ("12 13 14 15"), rather than real
# section headings. See PROJECT_OVERVIEW.txt section 4 for the full finding.
_LIST_ITEM_RE = re.compile(r"^\d+\.\s+[a-z]")
_PAGE_FOOTER_RE = re.compile(r"^(\d+\s+)+\d+$")


def find_suspicious_headings(headings: list) -> list[str]:
    return [
        h.text
        for h in headings
        if _LIST_ITEM_RE.match(h.text) or _PAGE_FOOTER_RE.match(h.text)
    ]


def main() -> None:
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw_pdfs/AUTOSAR_SWS_OS.pdf"
    if not Path(pdf_path).exists():
        print(f"'{pdf_path}' not found. Pass a PDF path as an argument.")
        sys.exit(1)

    RESULTS_DIR.mkdir(exist_ok=True)

    parsed = parse_pdf(pdf_path)
    chunks = chunk_document(parsed)
    headings = [b for b in parsed.blocks if b.is_heading]
    suspicious = find_suspicious_headings(headings)

    print(f"=== Document: {parsed.doc_name} ===")
    print(f"pages: {parsed.page_count}")
    print(f"text blocks extracted: {len(parsed.blocks)}")
    print(f"parse warnings: {parsed.warnings or 'none'}")
    print()

    print("=== Heading detection ===")
    print(f"total headings detected: {len(headings)}")
    print(
        f"suspicious (list-item / page-footer) false positives: "
        f"{len(suspicious)} ({100 * len(suspicious) / len(headings):.1f}% of all headings)"
    )
    print()

    print("=== Chunking ===")
    print(f"total chunks: {len(chunks)}")
    char_lens = [len(c.text) for c in chunks]
    token_lens = [len(c.text) // 4 for c in chunks]
    print(
        f"chunk length (chars): min={min(char_lens)} max={max(char_lens)} "
        f"mean={statistics.mean(char_lens):.1f} median={statistics.median(char_lens):.1f} "
        f"stdev={statistics.stdev(char_lens):.1f}"
    )
    print(
        f"chunk length (approx tokens, 4 chars/token): min={min(token_lens)} "
        f"max={max(token_lens)} mean={statistics.mean(token_lens):.1f} "
        f"median={statistics.median(token_lens):.1f}"
    )
    no_section = sum(1 for c in chunks if c.section is None)
    print(f"chunks with no section heading (front matter): {no_section} "
          f"({100 * no_section / len(chunks):.1f}%)")
    multi_page = sum(1 for c in chunks if c.page_end > c.page_start)
    print(f"chunks spanning more than one page: {multi_page} "
          f"({100 * multi_page / len(chunks):.1f}%)")

    # --- Plot 1: chunk length distribution ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(token_lens, bins=30, color="#2563eb", edgecolor="white")
    ax.axvline(statistics.median(token_lens), color="#dc2626", linestyle="--",
               label=f"median = {statistics.median(token_lens):.0f} tokens")
    ax.set_xlabel("Chunk length (approx. tokens)")
    ax.set_ylabel("Number of chunks")
    ax.set_title(f"Chunk size distribution — {parsed.doc_name} ({len(chunks)} chunks)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "chunk_size_distribution.png", dpi=150)
    plt.close(fig)

    # --- Plot 2: heading detection breakdown ---
    fig, ax = plt.subplots(figsize=(5, 4.5))
    clean = len(headings) - len(suspicious)
    ax.bar(["Correctly\ndetected", "False positive\n(list item /\npage footer)"],
           [clean, len(suspicious)], color=["#16a34a", "#dc2626"])
    for i, v in enumerate([clean, len(suspicious)]):
        ax.text(i, v + 3, str(v), ha="center")
    ax.set_ylabel("Number of headings")
    ax.set_title("Heading detection accuracy\n(pdf_parser.py heading classifier)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "heading_detection_accuracy.png", dpi=150)
    plt.close(fig)

    print()
    print(f"Saved plots to {RESULTS_DIR}/")
    print("  - chunk_size_distribution.png")
    print("  - heading_detection_accuracy.png")


if __name__ == "__main__":
    main()
