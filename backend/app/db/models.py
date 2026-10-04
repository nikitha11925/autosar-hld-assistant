"""
SQLite schema (SQLAlchemy models) for structured storage.

Covers: ingested documents/projects, query audit log entries (query text,
retrieved chunk IDs, answer, groundedness), and extraction results.
Every query and its retrieved chunks must be logged here for auditability
(see CLAUDE.md, design principle 5).
"""

from __future__ import annotations

import datetime

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Document(Base):
    """One ingested document (= one isolated vector collection)."""

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    doc_name: Mapped[str] = mapped_column(String, unique=True, index=True)
    collection_name: Mapped[str] = mapped_column(String)
    page_count: Mapped[int] = mapped_column(Integer)
    chunk_count: Mapped[int] = mapped_column(Integer)
    ingested_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow
    )

    queries: Mapped[list["QueryLog"]] = relationship(back_populates="document")


class QueryLog(Base):
    """Audit record for one Q&A call: the question, retrieved evidence, and answer."""

    __tablename__ = "query_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"))
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    grounded: Mapped[bool] = mapped_column()
    citations: Mapped[list] = mapped_column(JSON)  # [{doc_name, section, page_range}]
    retrieved_chunk_ids: Mapped[list] = mapped_column(JSON)
    top_similarity_score: Mapped[float] = mapped_column()
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow
    )

    document: Mapped["Document"] = relationship(back_populates="queries")
