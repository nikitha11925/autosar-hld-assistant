"""
RAG Q&A orchestration.

Ties together retrieval (vector_store.py) and generation into a single
query path. Applies a similarity-threshold guardrail: if no retrieved chunk
is confident enough, the engine must return an explicit refusal rather than
let the LLM guess (see CLAUDE.md, design principle 1).

Every call returns a single answer object with three fields:
  - answer: str
  - citations: list[{doc_name, section, page_range}]
  - grounded: bool

The underlying LLM call is kept behind a thin interface here so an
API-based model (dev/demo) can later be swapped for a local model
(Llama/Mistral/Qwen-class) without changing callers.
"""
