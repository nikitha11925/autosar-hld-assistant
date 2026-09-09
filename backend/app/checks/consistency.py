"""
Consistency checking.

Flags two categories of issues across an ingested document (or document
set):
  1. Terms referenced but never defined
  2. Terms defined more than once with conflicting descriptions

Prefers deterministic logic (term/definition extraction + comparison) over
pure LLM judgment where possible, per CLAUDE.md — LLM assistance is used
only where deterministic matching is insufficient (e.g. judging whether two
definitions actually conflict).
"""
