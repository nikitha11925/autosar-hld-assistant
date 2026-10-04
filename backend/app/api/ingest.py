"""
POST /ingest

Accepts an uploaded PDF (+ project/collection identifier), runs it through
pdf_parser.py -> chunker.py -> embed_service.py -> vector_store.py, and
records the document in SQLite. Returns ingestion status (chunk count,
any parse warnings).
"""

from __future__ import annotations

import os
import tempfile
import uuid

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from backend.app.db.models import Document
from backend.app.db.session import get_session
from backend.app.embeddings.embed_service import get_embedding_service
from backend.app.ingestion.chunker import chunk_document
from backend.app.ingestion.pdf_parser import parse_pdf
from backend.app.retrieval.vector_store import StoredChunk, get_vector_store

router = APIRouter()


class IngestResponse(BaseModel):
    doc_name: str
    collection_name: str
    page_count: int
    chunk_count: int
    warnings: list[str]


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)) -> IngestResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        parsed = parse_pdf(tmp_path, doc_name=file.filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        os.unlink(tmp_path)

    chunks = chunk_document(parsed)
    if not chunks:
        raise HTTPException(
            status_code=422,
            detail=f"No extractable text found in '{file.filename}'. "
            + " ".join(parsed.warnings),
        )

    embedding_service = get_embedding_service()
    embeddings = embedding_service.embed_documents([c.text for c in chunks])

    collection_name = f"doc_{uuid.uuid5(uuid.NAMESPACE_URL, parsed.doc_name).hex[:12]}"
    store = get_vector_store()
    # Reset rather than append: re-ingesting the same document (e.g. after an
    # edit, or a repeated upload) must overwrite its chunks, not duplicate them.
    collection = store.reset_collection(collection_name, embedding_service.dimension)

    stored_chunks = [
        StoredChunk(
            chunk_id=f"{collection_name}_{i}",
            text=c.text,
            doc_name=c.doc_name,
            section=c.section,
            page_start=c.page_start,
            page_end=c.page_end,
        )
        for i, c in enumerate(chunks)
    ]
    collection.add(embeddings, stored_chunks)
    store.persist(collection_name)

    session = get_session()
    try:
        existing = session.query(Document).filter_by(doc_name=parsed.doc_name).first()
        if existing is not None:
            existing.collection_name = collection_name
            existing.page_count = parsed.page_count
            existing.chunk_count = len(chunks)
        else:
            session.add(
                Document(
                    doc_name=parsed.doc_name,
                    collection_name=collection_name,
                    page_count=parsed.page_count,
                    chunk_count=len(chunks),
                )
            )
        session.commit()
    finally:
        session.close()

    return IngestResponse(
        doc_name=parsed.doc_name,
        collection_name=collection_name,
        page_count=parsed.page_count,
        chunk_count=len(chunks),
        warnings=parsed.warnings,
    )
