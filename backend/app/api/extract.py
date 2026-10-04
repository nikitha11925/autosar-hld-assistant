"""
POST /extract

Runs extraction/entity_extractor.py over an ingested document/project and
returns the schema-validated components/interfaces/ports/signals table.
Also exposes the checks/consistency.py results for the same document set.

Deferred out of the initial 8-hour build (see README "Known limitations") —
route exists so the frontend/API shape is stable, but returns 501 until
entity_extractor.py and checks/consistency.py are implemented.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

router = APIRouter()


@router.post("/extract")
async def extract_entities(doc_name: str) -> None:
    raise HTTPException(
        status_code=501,
        detail="Structured extraction is not yet implemented. See README known limitations.",
    )
