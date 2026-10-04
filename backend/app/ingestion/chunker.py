"""
Heading-aware chunking.

Splits parsed document content into retrieval-sized chunks, preferring
section/heading boundaries over naive fixed-size windows. Falls back to
~500-token sliding windows only within sections that are themselves too
long to embed as a single chunk.

Every chunk emitted here must carry metadata: {doc_name, section, page_range}
so the RAG layer can produce a citation for any chunk it retrieves.
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.app.ingestion.pdf_parser import ParsedDocument, TextBlock

DEFAULT_MAX_TOKENS = 500
# Approximation used throughout: ~4 characters per token (no tokenizer
# dependency at ingest time). This is a heuristic, not exact GPT/BPE counts.
_CHARS_PER_TOKEN = 4


@dataclass
class Chunk:
    """A retrieval-sized unit of text with citation metadata."""

    text: str
    doc_name: str
    section: str | None
    page_start: int
    page_end: int

    @property
    def page_range(self) -> str:
        if self.page_start == self.page_end:
            return str(self.page_start)
        return f"{self.page_start}-{self.page_end}"


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _group_by_section(blocks: list[TextBlock]) -> list[list[TextBlock]]:
    """Group consecutive blocks that share the same section path.

    Heading blocks start a new group (and are included as the group's
    first block, so their own text is retrievable/citable too).
    """
    groups: list[list[TextBlock]] = []
    current: list[TextBlock] = []
    current_section: str | None = object()  # sentinel, never equals a real section

    for block in blocks:
        starts_new_group = block.is_heading or block.section != current_section
        if starts_new_group and current:
            groups.append(current)
            current = []
        current.append(block)
        current_section = block.section

    if current:
        groups.append(current)

    return groups


def _split_long_group(
    group: list[TextBlock], doc_name: str, max_tokens: int
) -> list[Chunk]:
    """Fixed-window fallback for a section too long to embed as one chunk."""
    chunks: list[Chunk] = []
    window_blocks: list[TextBlock] = []
    window_tokens = 0

    def flush() -> None:
        if not window_blocks:
            return
        text = "\n".join(b.text for b in window_blocks)
        chunks.append(
            Chunk(
                text=text,
                doc_name=doc_name,
                section=window_blocks[0].section,
                page_start=window_blocks[0].page,
                page_end=window_blocks[-1].page,
            )
        )

    for block in group:
        block_tokens = _estimate_tokens(block.text)
        if window_blocks and window_tokens + block_tokens > max_tokens:
            flush()
            window_blocks = []
            window_tokens = 0
        window_blocks.append(block)
        window_tokens += block_tokens

    flush()
    return chunks


def chunk_document(
    parsed: ParsedDocument, max_tokens: int = DEFAULT_MAX_TOKENS
) -> list[Chunk]:
    """Chunk a parsed document, preferring section boundaries.

    Each section (a heading plus the body text under it, up to the next
    heading of any level) becomes one chunk. If a section's text exceeds
    `max_tokens`, it is split further using a fixed-size sliding window,
    per CLAUDE.md's chunking policy.
    """
    non_empty_blocks = [b for b in parsed.blocks if b.text.strip()]
    if not non_empty_blocks:
        return []

    chunks: list[Chunk] = []
    for group in _group_by_section(non_empty_blocks):
        text = "\n".join(b.text for b in group)
        if _estimate_tokens(text) <= max_tokens:
            chunks.append(
                Chunk(
                    text=text,
                    doc_name=parsed.doc_name,
                    section=group[0].section,
                    page_start=group[0].page,
                    page_end=group[-1].page,
                )
            )
        else:
            chunks.extend(_split_long_group(group, parsed.doc_name, max_tokens))

    return chunks
