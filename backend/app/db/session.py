"""
SQLite engine/session setup (SQLAlchemy).

Provides a session factory used by the API layer and audit logging code.
Kept separate from models.py so engine/connection config can change
independently of schema definitions.
"""

from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.app.db.models import Base

DB_PATH = os.environ.get("SQLITE_DB_PATH", "./data/audit.sqlite3")
_engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)
SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)


def init_db() -> None:
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    Base.metadata.create_all(_engine)


def get_session() -> Session:
    return SessionLocal()
