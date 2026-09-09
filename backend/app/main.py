"""
FastAPI application entrypoint.

Wires together the ingest, query, extract, and audit routers (see
backend/app/api/) into a single ASGI app. Owns startup/shutdown concerns
(e.g. initializing the ChromaDB client, SQLite session) but contains no
business logic itself — that belongs in the ingestion/embeddings/retrieval/
rag/extraction/checks modules.
"""
