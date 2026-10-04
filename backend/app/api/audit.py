"""
GET /audit

Read-only endpoint exposing the query/retrieval audit log stored in
SQLite (db/models.py) — used by the Streamlit frontend's audit view and by
eval/run_eval.py for scoring past queries.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from backend.app.db.models import QueryLog
from backend.app.db.session import get_session

router = APIRouter()


class AuditEntry(BaseModel):
    id: int
    document_id: int
    question: str
    answer: str
    grounded: bool
    citations: list
    top_similarity_score: float
    created_at: str


@router.get("/audit", response_model=list[AuditEntry])
async def get_audit_log(limit: int = 50) -> list[AuditEntry]:
    session = get_session()
    try:
        rows = (
            session.query(QueryLog)
            .order_by(QueryLog.created_at.desc())
            .limit(limit)
            .all()
        )
        return [
            AuditEntry(
                id=r.id,
                document_id=r.document_id,
                question=r.question,
                answer=r.answer,
                grounded=r.grounded,
                citations=r.citations,
                top_similarity_score=r.top_similarity_score,
                created_at=r.created_at.isoformat(),
            )
            for r in rows
        ]
    finally:
        session.close()
