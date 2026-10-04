"""
POST /query

Accepts a natural-language question (+ project/collection identifier),
runs it through rag/qa_engine.py, logs the query + retrieved chunk IDs +
answer to SQLite, and returns the answer object: {answer, citations[],
grounded}.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.app.db.models import Document, QueryLog
from backend.app.db.session import get_session
from backend.app.rag.qa_engine import QAEngine

router = APIRouter()
_engine = QAEngine()


class QueryRequest(BaseModel):
    doc_name: str
    question: str


class CitationResponse(BaseModel):
    doc_name: str
    section: str | None
    page_range: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationResponse]
    grounded: bool


@router.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest) -> QueryResponse:
    session = get_session()
    try:
        document = session.query(Document).filter_by(doc_name=request.doc_name).first()
        if document is None:
            raise HTTPException(
                status_code=404,
                detail=f"Document '{request.doc_name}' has not been ingested yet.",
            )

        try:
            result = _engine.answer(document.collection_name, request.question)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        session.add(
            QueryLog(
                document_id=document.id,
                question=request.question,
                answer=result.answer,
                grounded=result.grounded,
                citations=[
                    {"doc_name": c.doc_name, "section": c.section, "page_range": c.page_range}
                    for c in result.citations
                ],
                retrieved_chunk_ids=result.retrieved_chunk_ids,
                top_similarity_score=result.top_similarity_score,
            )
        )
        session.commit()
    finally:
        session.close()

    return QueryResponse(
        answer=result.answer,
        citations=[
            CitationResponse(doc_name=c.doc_name, section=c.section, page_range=c.page_range)
            for c in result.citations
        ],
        grounded=result.grounded,
    )
