# autosar-hld-assistant

AI-assisted AUTOSAR High-Level Design (HLD) Document Analysis Assistant —
Case Study 1 ("Pilot" module) of a 5-module Automotive Engineering AI
Platform reference architecture. See [CLAUDE.MD](CLAUDE.MD) for full scope,
non-negotiable design principles, and directory layout.

**Status: working MVP.** Ingestion, embeddings, retrieval, grounded RAG
Q&A with citations and a refusal guardrail, a minimal Streamlit UI, an
audit log, and a hand-written eval set are all implemented and verified
end-to-end against a real 342-page AUTOSAR specification PDF. Structured
entity extraction and the consistency-checking pass are deferred (see
"Known limitations" below) — they were descoped to fit an 8-hour build
window and are not part of this status.

## Why this project first

Of the platform's five case studies (AUTOSAR HLD, HARA/safety, TARA/
cybersecurity, secure code review, UDS diagnostics), this one is the
documented Pilot: lowest decision-risk, no deep ISO 26262/21434 domain
expertise required to validate correctness, and it produces a reusable
retrieval/citation core that the other four modules would layer
domain-specific deterministic rules on top of (see CLAUDE.md's "Interview
framing" section for how each phase builds on this one).

## Architecture

```
PDF upload
  -> ingestion/pdf_parser.py    (PyMuPDF; heading-aware text + page extraction)
  -> ingestion/chunker.py       (section-boundary chunking, fixed-window fallback)
  -> embeddings/embed_service.py (sentence-transformers, bge-small-en-v1.5)
  -> retrieval/vector_store.py   (FAISS, one isolated collection per document)
  -> db/models.py                (SQLite: documents + query audit log)

Question
  -> rag/qa_engine.py: embed query -> FAISS top-k -> similarity-threshold
     guardrail -> Gemini (behind a swappable LLMClient interface) -> answer
     with citations {doc_name, section, page_range} and a grounded: bool flag
  -> every query + retrieved chunk IDs + answer logged to SQLite (audit.py)

FastAPI (backend/app/main.py) exposes: POST /ingest, POST /query, GET /audit,
POST /extract (not yet implemented, returns 501)
Streamlit (frontend/streamlit_app.py) calls the API over HTTP: Upload/Ingest
and Ask tabs.
```

### Vector store: FAISS, not ChromaDB

CLAUDE.md lists ChromaDB as the default with FAISS as an acceptable
alternative. This build uses FAISS because `chromadb`'s `chroma-hnswlib`
dependency requires a C++ compiler toolchain (MSVC Build Tools) that isn't
installed on the dev machine, and no prebuilt wheel exists for Python 3.13
on Windows. FAISS has prebuilt wheels and installs cleanly. The vector
store is wrapped behind `retrieval/vector_store.py`'s `VectorStore`/
`Collection` interface, so swapping back to ChromaDB only touches that one
file. Each ingested document gets its own isolated FAISS index + metadata
file under `data/chroma_db/` (directory name kept from the original plan).

### LLM: Gemini (free tier), behind a swappable interface

CLAUDE.md requires the LLM call to stay behind a thin interface so a local
model can be swapped in later without touching callers. This build uses
Google's Gemini API (`gemini-2.5-flash`) for dev/demo speed — it's called
through `rag/qa_engine.py`'s `LLMClient` abstract interface, with
`GeminiClient` as the concrete implementation. A local model
(Llama/Mistral/Qwen-class via e.g. Ollama) would be a second `LLMClient`
subclass; no other code changes.

## Grounding / refusal guardrail

This is the project's core non-negotiable principle (CLAUDE.md #1): every
answer must be traceable to retrieved source text, and the system must
refuse rather than guess when it isn't. Implemented as two layers in
`rag/qa_engine.py`:

1. **Similarity threshold**: if the top retrieved chunk's cosine similarity
   is below `DEFAULT_SIMILARITY_THRESHOLD` (0.35), the engine refuses
   without ever calling the LLM.
2. **LLM-level refusal**: the system prompt (`rag/prompt_templates.py`)
   instructs the model to answer only from the provided excerpts and
   return the literal string `NOT_FOUND` if they're insufficient; the
   engine maps that to the same refusal path.

Both paths are covered by unit tests (`tests/test_qa_engine.py`) using a
fake LLM client that asserts the LLM is never even called when the
similarity guardrail already decided to refuse.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate        # or source .venv/bin/activate on Linux/Mac
pip install -r requirements.txt -r requirements-dev.txt

cp .env.example .env
# edit .env: set GEMINI_API_KEY (get a free key at aistudio.google.com)

# run the backend
PYTHONPATH=. uvicorn backend.app.main:app --reload --port 8000

# in a second terminal, run the frontend
PYTHONPATH=. streamlit run frontend/streamlit_app.py
```

Then open the Streamlit UI, upload an AUTOSAR-style PDF in the
"Upload / Ingest" tab, and ask questions in the "Ask" tab.

### Docker

```bash
docker-compose up --build
```

Backend on :8000, frontend on :8501. Requires `.env` with `GEMINI_API_KEY`
set (docker-compose reads it via `env_file`).

## Gemini free-tier rate limits (known operational constraint)

The free tier enforces both a per-minute and a per-day request quota, and
these vary by model and can change. During this build, `gemini-3.8-flash`
hit a 20-requests/day cap; `gemini-2.5-flash` was used instead and has
comfortably handled eval + manual testing. If you hit `503`/`429` errors,
check `GEMINI_MODEL` in `.env` and https://ai.google.dev/gemini-api/docs/rate-limits.
`GeminiClient.generate()` retries transient `503`s automatically and backs
off ~15s on a `429`; `eval/run_eval.py` paces requests 20s apart for the
same reason.

## Eval results

Run against the real AUTOSAR `AUTOSAR_SWS_OS.pdf` specification (342 pages,
R20-11, publicly available from autosar.org), ingested as
`data/raw_pdfs/AUTOSAR_SWS_OS.pdf` -> 549 chunks. Test set:
`eval/qa_test_set.json` (8 hand-written cases: 5 answerable-from-document
questions with an expected citation section, 3 clearly out-of-scope
questions that must be refused).

```
$ python eval/run_eval.py

--- Eval summary ---
Total cases: 8
Refusal correctness: 3/3 (100%)
Groundedness (answered when expected): 4/5 (80%)
Citation presence on grounded answers: 4/4 (100%)
Retrieval section-match precision: 3/5 (60%)
```

- **Refusal correctness (100%)**: all three out-of-scope questions (e.g.
  "What is the capital of France?") were correctly refused with
  `grounded: false` and no fabricated citations — confirms CLAUDE.md
  principle 1 holds under real (not just mocked) conditions.
- **Groundedness (80%)**: one real miss, traced to a chunking quality
  issue — see "Known limitations" below. Every other in-scope question was
  answered with real citations.
- **Citation presence (100%)**: every grounded answer carried at least one
  citation with document/section/page — principle 2 (one citation minimum
  per claim) holds for all answered cases.
- **Section-match precision (60%)**: the retrieval doesn't always surface
  the exact section a human would pick first, though the content is still
  factually correct in all 4 grounded answers — this metric is stricter
  than "was the answer right."

## Known limitations

- **Structured extraction and consistency checking are not implemented.**
  `extraction/entity_extractor.py` and `checks/consistency.py` remain
  stubs; `POST /extract` returns `501`. These were explicitly descoped to
  fit an 8-hour build window — not silently dropped. Both are designed to
  slot into the existing pipeline (retrieval + a validated-JSON-schema
  layer) without architectural changes.
- **Heading false-positives from numbered list items.** `pdf_parser.py`'s
  heading detector occasionally misclassifies numbered prose lines (e.g.
  "1. More than one core on the same piece of silicon.") as sub-headings
  when they're visually bold/larger, fragmenting the parent section into
  many low-signal chunks and reducing that content's retrieval rank. This
  caused the one groundedness miss in the eval run above (a real question
  about multi-core CPU hardware assumptions, whose answer lives in exactly
  such a fragmented section). Fix direction: tighten
  `_classify_line()` in `pdf_parser.py` to require more than a bare digit
  + period before treating a line as a heading number.
  - Also produces occasional noisy headings from page-number running
    footers (e.g. "12 13 14 15").
- **No OCR** (explicitly out of scope per CLAUDE.md) — scanned/image-only
  PDFs are detected and produce a warning, not a crash.
- **No auth/RBAC** — single-user, local-only by design for this MVP.
- **Document re-ingestion overwrites, not versions.** Re-ingesting the same
  `doc_name` replaces its vector collection and SQLite row; there's no
  revision history (explicitly out of scope per CLAUDE.md).

## How this extends to HARA / TARA / UDS (design only)

Per the source reference document's roadmap, every later module reuses
this same retrieval + citation + audit-log core and adds a domain-specific
deterministic rules layer on top:

- **HARA (Phase 2)**: same ingestion/retrieval core over safety
  documents; adds a rule-based S/E/C -> ASIL decision table (deterministic,
  not LLM-judged) and a safety-goal/requirement drafting flow gated by the
  same grounding guardrail.
- **TARA (Phase 2)**: same core over architecture/interface docs; adds a
  threat-pattern library lookup (retrieval, not generation) plus guided
  risk scoring.
- **UDS (Phase 3)**: same core over ISO 14229 + OEM diagnostic specs; adds
  deterministic UDS request/response structure validation rather than
  LLM-generated message bytes.
- **Secure code review (Phase 4)**: same citation/audit pattern applied to
  source code + coding-standard documents, with retrieval scoped to
  authorized repositories only, and the LLM call kept local given source
  IP sensitivity.

In every case, the swappable `LLMClient` interface, per-document FAISS
collection isolation, and SQLite audit logging introduced here carry over
unchanged.
