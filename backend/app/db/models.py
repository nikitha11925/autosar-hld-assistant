"""
SQLite schema (SQLAlchemy models) for structured storage.

Covers: ingested documents/projects, query audit log entries (query text,
retrieved chunk IDs, answer, groundedness), and extraction results.
Every query and its retrieved chunks must be logged here for auditability
(see CLAUDE.md, design principle 5).
"""
