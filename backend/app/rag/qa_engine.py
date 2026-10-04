"""
RAG Q&A orchestration.

Ties together retrieval (vector_store.py) and generation into a single
query path. Applies a similarity-threshold guardrail: if no retrieved chunk
is confident enough, the engine must return an explicit refusal rather than
let the LLM guess (see CLAUDE.md, design principle 1).

Every call returns a single answer object with three fields:
  - answer: str
  - citations: list[{doc_name, section, page_range}]
  - grounded: bool

The underlying LLM call is kept behind a thin interface (LLMClient) here so
an API-based model (dev/demo) can later be swapped for a local model
(Llama/Mistral/Qwen-class) without changing callers.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

from backend.app.embeddings.embed_service import EmbeddingService, get_embedding_service
from backend.app.rag.prompt_templates import SYSTEM_PROMPT, build_user_prompt
from backend.app.retrieval.vector_store import Match, VectorStore, get_vector_store

# Cosine similarity threshold (vectors are normalized, so IndexFlatIP score
# is cosine similarity in [-1, 1]). Below this, retrieved context is treated
# as too weak to ground an answer on and the engine refuses rather than
# letting the LLM guess.
DEFAULT_SIMILARITY_THRESHOLD = 0.35
DEFAULT_TOP_K = 5
REFUSAL_TEXT = "Not found in the provided documents."


@dataclass
class Citation:
    doc_name: str
    section: str | None
    page_range: str


@dataclass
class Answer:
    answer: str
    citations: list[Citation]
    grounded: bool
    retrieved_chunk_ids: list[str]
    top_similarity_score: float


class LLMClient(ABC):
    """Thin interface so the LLM provider can be swapped (API model <-> local model)."""

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError


class GeminiClient(LLMClient):
    """Default dev/demo LLM client, calling Google's Gemini API."""

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        from google import genai
        from google.genai import types

        self._types = types
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Add it to your .env file "
                "(see .env.example)."
            )
        self._client = genai.Client(api_key=key)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        from google.genai.errors import ClientError, ServerError

        import time

        last_error: Exception | None = None
        for attempt in range(4):
            try:
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=user_prompt,
                    config=self._types.GenerateContentConfig(
                        system_instruction=system_prompt, temperature=0.0
                    ),
                )
                return (response.text or "").strip()
            except ServerError as exc:
                # Transient overload (e.g. 503 UNAVAILABLE) — short retry clears most of these.
                last_error = exc
                if attempt < 3:
                    time.sleep(2 * (attempt + 1))
            except ClientError as exc:
                # The free tier enforces a low requests-per-minute quota (429
                # RESOURCE_EXHAUSTED); back off longer since the limit is per-minute.
                last_error = exc
                if getattr(exc, "code", None) == 429 and attempt < 3:
                    time.sleep(15)
                else:
                    raise RuntimeError(f"Gemini API request failed: {exc}") from exc
        raise RuntimeError(f"Gemini API is temporarily unavailable: {last_error}") from last_error


class QAEngine:
    def __init__(
        self,
        vector_store: VectorStore | None = None,
        embedding_service: EmbeddingService | None = None,
        llm_client: LLMClient | None = None,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        self.vector_store = vector_store or get_vector_store()
        self.embedding_service = embedding_service or get_embedding_service()
        self._llm_client = llm_client
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k

    @property
    def llm_client(self) -> LLMClient:
        if self._llm_client is None:
            self._llm_client = GeminiClient()
        return self._llm_client

    def answer(self, collection_name: str, question: str) -> Answer:
        query_embedding = self.embedding_service.embed_query(question)
        collection = self.vector_store.get_or_create_collection(
            collection_name, self.embedding_service.dimension
        )
        matches = collection.query(query_embedding, top_k=self.top_k)

        top_score = matches[0].score if matches else -1.0
        if not matches or top_score < self.similarity_threshold:
            return Answer(
                answer=REFUSAL_TEXT,
                citations=[],
                grounded=False,
                retrieved_chunk_ids=[m.chunk.chunk_id for m in matches],
                top_similarity_score=top_score,
            )

        excerpts = [
            {
                "index": i + 1,
                "text": m.chunk.text,
                "doc_name": m.chunk.doc_name,
                "section": m.chunk.section or "(no section heading)",
                "page_range": self._page_range(m),
            }
            for i, m in enumerate(matches)
        ]
        user_prompt = build_user_prompt(question, excerpts)
        raw_answer = self.llm_client.generate(SYSTEM_PROMPT, user_prompt)

        if raw_answer.strip() == "NOT_FOUND" or not raw_answer.strip():
            return Answer(
                answer=REFUSAL_TEXT,
                citations=[],
                grounded=False,
                retrieved_chunk_ids=[m.chunk.chunk_id for m in matches],
                top_similarity_score=top_score,
            )

        citations = [
            Citation(
                doc_name=m.chunk.doc_name,
                section=m.chunk.section,
                page_range=self._page_range(m),
            )
            for m in matches
        ]
        return Answer(
            answer=raw_answer,
            citations=citations,
            grounded=True,
            retrieved_chunk_ids=[m.chunk.chunk_id for m in matches],
            top_similarity_score=top_score,
        )

    @staticmethod
    def _page_range(match: Match) -> str:
        chunk = match.chunk
        if chunk.page_start == chunk.page_end:
            return str(chunk.page_start)
        return f"{chunk.page_start}-{chunk.page_end}"
