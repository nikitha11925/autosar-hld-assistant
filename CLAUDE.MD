# CLAUDE.md — autosar-hld-assistant

This file orients Claude (or any AI pair-programmer) working in this repository.
Read this before making changes. It defines what this project is, what "good"
looks like, and the constraints that must never be silently dropped.

## What this project is

An AI-assisted **AUTOSAR High-Level Design (HLD) Document Analysis Assistant** —
Case Study 1 of a 5-module "Automotive Engineering AI Platform" reference
architecture. It is being built as a portfolio/interview project, so code
quality, defensibility of design decisions, and evaluation rigor matter more
than feature count.

Core idea: engineers upload AUTOSAR SWS/TR-style PDFs (or any large,
structured HLD document). The system extracts, indexes, and lets engineers
query the content in natural language, with **every answer grounded in a
citation (document, section, page)**. It also extracts a structured
component/interface table and flags consistency issues (undefined or
conflicting terms).

This is deliberately scoped as **one module of a larger platform**. Do not
expand scope toward HARA (safety), TARA (cybersecurity), secure code review,
or UDS diagnostics unless explicitly asked — those are documented as future
modules that reuse this same retrieval/citation core plus a domain-specific
deterministic rules layer.

## Non-negotiable design principles

These come directly from the source reference document and must hold for
every feature added:

1. **Grounded, not generated.** Every answer must be traceable to retrieved
   source text. If retrieval confidence is below threshold, the system must
   say "not found in the provided documents" rather than let the LLM guess.
   Never remove or weaken this guardrail to make demos look more impressive.
2. **One citation minimum per claim.** Citations include document name,
   section, and page (or page range).
3. **Human review is the final step.** The system assists; it never
   auto-approves, auto-modifies source documents, or claims authority over
   architecture decisions.
4. **Project/document isolation.** Each ingested document (or document set)
   lives in its own vector collection. Do not silently merge collections
   across projects.
5. **Everything is auditable.** Every query, the chunks retrieved for it, and
   the answer given must be logged (SQLite is fine for MVP).
6. **Local-first stack.** Design for a locally-hosted or private LLM and
   local vector storage (ChromaDB/FAISS). API-based LLMs are acceptable for
   development/demo convenience, but the code must not hardcode assumptions
   that make a local-model swap hard (keep the LLM call behind a thin
   interface).

## Recommended stack (do not deviate without a stated reason)

- Backend: FastAPI
- Frontend: Streamlit
- Embeddings: sentence-transformers (bge-small-en or e5-small class model)
- Vector store: ChromaDB (persistent, local)
- Structured storage: SQLite (documents, queries, audit log)
- PDF parsing: PyMuPDF
- Packaging: Docker / docker-compose
- LLM: pluggable — default to an API-based model for dev, but keep the
  interface swappable for a local model (Llama/Mistral/Qwen-class)

## Scope boundaries (MVP)

**In scope:**
- PDF ingestion, heading-aware chunking, embeddings, ChromaDB storage
- Cited RAG Q&A with a grounding/refusal guardrail
- Structured extraction: components / interfaces / ports / signals as a
  validated JSON table
- Consistency check: terms referenced but never defined; terms defined more
  than once with conflicting descriptions
- Query + retrieval audit log
- A small hand-built evaluation set (15-20 Q&A pairs) with retrieval
  precision, citation validity, and refusal-correctness metrics reported in
  the README

**Explicitly out of scope (do not build unless asked):**
- OCR for scanned pages
- Full multi-user auth/RBAC (a folder-per-project stub is enough)
- Document revision/version comparison
- Kubernetes, enterprise IAM, Neo4j graph store
- Any of the other 4 case studies (HARA, TARA, secure code review, UDS)

If a request would cross one of these boundaries, flag it rather than
quietly implementing it — scope creep is the main way this kind of project
loses credibility in review.

## Code style / structure expectations

- Keep ingestion, embedding, retrieval, RAG orchestration, extraction, and
  consistency-checking as separate modules behind clear interfaces (see
  directory layout below). This mirrors the platform's layered reference
  architecture and makes it easy to explain in an interview.
- Validate any LLM-produced structured output (JSON) against a schema before
  returning it from an API endpoint — never trust free-text LLM output
  directly as structured data.
- Every RAG answer object should carry: `answer`, `citations[]`, and a
  `grounded: bool` flag, not just a text blob.
- Prefer heading/section-aware chunking over naive fixed-size windows;
  fall back to fixed windows only within very long sections.

## Directory layout

```
autosar-hld-assistant/
├── backend/
│   └── app/
│       ├── main.py
│       ├── ingestion/        (pdf_parser.py, chunker.py)
│       ├── embeddings/       (embed_service.py)
│       ├── retrieval/        (vector_store.py)
│       ├── rag/              (prompt_templates.py, qa_engine.py)
│       ├── extraction/       (entity_extractor.py)
│       ├── checks/           (consistency.py)
│       ├── db/                (models.py, session.py)
│       └── api/               (ingest.py, query.py, extract.py, audit.py)
├── frontend/
│   └── streamlit_app.py
├── data/
│   ├── raw_pdfs/
│   └── chroma_db/
├── eval/
│   ├── qa_test_set.json
│   └── run_eval.py
├── docker-compose.yml
├── Dockerfile.backend
├── Dockerfile.frontend
└── README.md
```

## Definition of done for this MVP

- `docker-compose up` starts backend + frontend and both are usable.
- Uploading a real AUTOSAR PDF successfully ingests and is queryable within
  the same session.
- Every answer in the UI shows at least one citation or an explicit
  "not found" refusal.
- `eval/run_eval.py` runs against the test set and prints precision/
  groundedness numbers.
- README documents the architecture, setup steps, eval results, and a short
  section describing how this module's core would extend to HARA/TARA/UDS
  (design only, not implemented).

## Interview framing to keep in mind while building

This project is explicitly the "Pilot" module in a 5-stage roadmap (Pilot →
Phase 2: HARA+TARA → Phase 3: UDS → Phase 4: Secure Code Review → Phase 5:
Platform Scale). Code and README should make it easy to answer: "why build
this one first?" (lowest decision-risk, produces a reusable knowledge base)
and "how would this extend?" (same retrieval/citation core + a
domain-specific deterministic rules layer added per module).
