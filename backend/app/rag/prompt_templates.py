"""
Prompt templates for the RAG Q&A engine.

Holds the strict grounding prompt: the LLM must answer only from retrieved
context and must not use outside knowledge. Kept separate from qa_engine.py
so prompt wording can be iterated on and evaluated independently of
orchestration logic.
"""

from __future__ import annotations

SYSTEM_PROMPT = (
    "You are an AUTOSAR High-Level Design document analysis assistant. "
    "You answer ONLY using the excerpts provided below, taken from an AUTOSAR "
    "specification document. Do not use any outside knowledge of AUTOSAR, "
    "even if you believe you know the answer.\n\n"
    "Rules:\n"
    "1. If the excerpts do not contain enough information to answer the "
    "question, respond with exactly: NOT_FOUND\n"
    "2. Otherwise, answer concisely and only state facts present in the excerpts.\n"
    "3. Do not fabricate section numbers, page numbers, or component names "
    "that are not in the excerpts.\n"
)


def build_user_prompt(question: str, excerpts: list[dict]) -> str:
    """Build the user-turn prompt: numbered excerpts followed by the question.

    `excerpts` items are dicts with keys: index, text, doc_name, section, page_range.
    """
    parts = []
    for ex in excerpts:
        header = f"[Excerpt {ex['index']}] (source: {ex['doc_name']}, section: {ex['section']}, page: {ex['page_range']})"
        parts.append(f"{header}\n{ex['text']}")
    excerpt_block = "\n\n".join(parts)
    return f"{excerpt_block}\n\nQuestion: {question}\n\nAnswer:"
