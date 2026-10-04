"""
FastAPI application entrypoint.

Wires together the ingest, query, extract, and audit routers (see
backend/app/api/) into a single ASGI app. Owns startup/shutdown concerns
(e.g. initializing the ChromaDB client, SQLite session) but contains no
business logic itself — that belongs in the ingestion/embeddings/retrieval/
rag/extraction/checks modules.
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()  # must run before any module-level os.environ.get() in api/db/rag imports below

from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.app.api import audit, extract, ingest, query
from backend.app.db.session import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="AUTOSAR HLD Document Analysis Assistant", lifespan=lifespan)

app.include_router(ingest.router)
app.include_router(query.router)
app.include_router(extract.router)
app.include_router(audit.router)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
