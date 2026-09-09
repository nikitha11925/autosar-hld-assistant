"""
GET /audit

Read-only endpoint exposing the query/retrieval audit log stored in
SQLite (db/models.py) — used by the Streamlit frontend's audit view and by
eval/run_eval.py for scoring past queries.
"""
