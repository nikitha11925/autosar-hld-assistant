"""
POST /ingest

Accepts an uploaded PDF (+ project/collection identifier), runs it through
pdf_parser.py -> chunker.py -> embed_service.py -> vector_store.py, and
records the document in SQLite. Returns ingestion status (chunk count,
any parse warnings).
"""
