"""
Structured extraction of AUTOSAR entities.

Uses the LLM to pull components, interfaces, ports, and signals out of
ingested document text into a structured table. Raw LLM JSON output is
never trusted directly — every extraction result is validated against a
schema before being returned from the API layer (see CLAUDE.md, code style
expectations).
"""
