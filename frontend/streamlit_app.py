"""
Streamlit frontend.

Three tabs:
  - Upload/Ingest: upload a PDF, trigger POST /ingest, show ingestion status
  - Ask: chat interface against POST /query, rendering inline citations and
    a grounded/not-grounded badge on every answer
  - Explore: the extraction table (POST /extract) and the consistency
    report (undefined terms, conflicting definitions)

This file is UI-only — it calls the FastAPI backend over HTTP and contains
no business logic of its own.
"""
