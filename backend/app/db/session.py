"""
SQLite engine/session setup (SQLAlchemy).

Provides a session factory used by the API layer and audit logging code.
Kept separate from models.py so engine/connection config can change
independently of schema definitions.
"""
