"""
Tests for the RAG grounding/refusal guardrail — CLAUDE.md's non-negotiable
design principle 1. Uses fake embedding/LLM clients so these tests don't
depend on network access or heavy ML dependencies.
"""

from __future__ import annotations

import pytest

from backend.app.rag.qa_engine import LLMClient, QAEngine, REFUSAL_TEXT
from backend.app.retrieval.vector_store import StoredChunk, VectorStore

faiss = pytest.importorskip("faiss")


class FakeEmbeddingService:
    """Returns fixed 3-d vectors so similarity is fully controllable in tests."""

    dimension = 3

    def __init__(self, query_vector):
        self._query_vector = query_vector

    def embed_query(self, text: str):
        return self._query_vector

    def embed_documents(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]


class FakeLLMClient(LLMClient):
    def __init__(self, response: str):
        self.response = response
        self.called = False

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.called = True
        return self.response


def _make_store_with_one_chunk(tmp_path):
    store = VectorStore(persist_dir=str(tmp_path))
    collection = store.get_or_create_collection("c1", dimension=3)
    collection.add(
        embeddings=[[1.0, 0.0, 0.0]],
        chunks=[
            StoredChunk(
                chunk_id="chunk-1",
                text="The OS module supports multi-core CPUs.",
                doc_name="doc.pdf",
                section="4.5 CPU Core features",
                page_start=22,
                page_end=22,
            )
        ],
    )
    return store


def test_refuses_when_similarity_below_threshold(tmp_path):
    store = _make_store_with_one_chunk(tmp_path)
    # Orthogonal query vector -> cosine similarity ~0, well below threshold.
    llm = FakeLLMClient("this should never be used")
    engine = QAEngine(
        vector_store=store,
        embedding_service=FakeEmbeddingService([0.0, 1.0, 0.0]),
        llm_client=llm,
    )

    result = engine.answer("c1", "unrelated question")

    assert result.grounded is False
    assert result.answer == REFUSAL_TEXT
    assert result.citations == []
    assert llm.called is False  # guardrail short-circuits before calling the LLM


def test_refuses_when_llm_signals_not_found(tmp_path):
    store = _make_store_with_one_chunk(tmp_path)
    llm = FakeLLMClient("NOT_FOUND")
    engine = QAEngine(
        vector_store=store,
        embedding_service=FakeEmbeddingService([1.0, 0.0, 0.0]),
        llm_client=llm,
    )

    result = engine.answer("c1", "does this document mention warp drives?")

    assert result.grounded is False
    assert result.answer == REFUSAL_TEXT
    assert llm.called is True


def test_answers_with_citations_when_grounded(tmp_path):
    store = _make_store_with_one_chunk(tmp_path)
    llm = FakeLLMClient("The OS module supports multi-core CPUs.")
    engine = QAEngine(
        vector_store=store,
        embedding_service=FakeEmbeddingService([1.0, 0.0, 0.0]),
        llm_client=llm,
    )

    result = engine.answer("c1", "does the OS support multi-core?")

    assert result.grounded is True
    assert result.answer == "The OS module supports multi-core CPUs."
    assert len(result.citations) == 1
    assert result.citations[0].doc_name == "doc.pdf"
    assert result.citations[0].section == "4.5 CPU Core features"
    assert result.citations[0].page_range == "22"


def test_refuses_when_collection_is_empty(tmp_path):
    store = VectorStore(persist_dir=str(tmp_path))
    llm = FakeLLMClient("should not be called")
    engine = QAEngine(
        vector_store=store,
        embedding_service=FakeEmbeddingService([1.0, 0.0, 0.0]),
        llm_client=llm,
    )

    result = engine.answer("empty-collection", "anything")

    assert result.grounded is False
    assert llm.called is False
