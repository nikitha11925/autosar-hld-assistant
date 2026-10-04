"""Tests for the FAISS-backed vector store: isolation, persistence, retrieval order."""

from __future__ import annotations

import pytest

from backend.app.retrieval.vector_store import StoredChunk, VectorStore

faiss = pytest.importorskip("faiss")


def _chunk(chunk_id: str, text: str) -> StoredChunk:
    return StoredChunk(
        chunk_id=chunk_id,
        text=text,
        doc_name="doc.pdf",
        section="1 Intro",
        page_start=1,
        page_end=1,
    )


def test_query_returns_most_similar_first(tmp_path):
    store = VectorStore(persist_dir=str(tmp_path))
    collection = store.get_or_create_collection("c1", dimension=3)

    collection.add(
        embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        chunks=[_chunk("a", "a"), _chunk("b", "b"), _chunk("c", "c")],
    )

    matches = collection.query([1.0, 0.0, 0.0], top_k=2)
    assert matches[0].chunk.chunk_id == "a"
    assert matches[0].score == pytest.approx(1.0, abs=1e-5)


def test_collections_are_isolated(tmp_path):
    store = VectorStore(persist_dir=str(tmp_path))
    c1 = store.get_or_create_collection("doc1", dimension=3)
    c2 = store.get_or_create_collection("doc2", dimension=3)

    c1.add([[1.0, 0.0, 0.0]], [_chunk("a", "only in doc1")])

    assert c1.index.ntotal == 1
    assert c2.index.ntotal == 0


def test_persist_and_reload_roundtrip(tmp_path):
    store = VectorStore(persist_dir=str(tmp_path))
    collection = store.get_or_create_collection("c1", dimension=3)
    collection.add([[1.0, 0.0, 0.0]], [_chunk("a", "hello")])
    store.persist("c1")

    reloaded_store = VectorStore(persist_dir=str(tmp_path))
    reloaded = reloaded_store.get_or_create_collection("c1", dimension=3)

    assert reloaded.index.ntotal == 1
    assert reloaded.chunks[0].text == "hello"


def test_empty_collection_query_returns_no_matches(tmp_path):
    store = VectorStore(persist_dir=str(tmp_path))
    collection = store.get_or_create_collection("empty", dimension=3)
    assert collection.query([1.0, 0.0, 0.0]) == []


def test_reset_collection_overwrites_instead_of_duplicating(tmp_path):
    """Re-ingesting the same document must not silently double its chunks."""
    store = VectorStore(persist_dir=str(tmp_path))
    collection = store.get_or_create_collection("c1", dimension=3)
    collection.add([[1.0, 0.0, 0.0]], [_chunk("a", "first ingest")])
    store.persist("c1")

    collection = store.reset_collection("c1", dimension=3)
    collection.add([[1.0, 0.0, 0.0]], [_chunk("a", "second ingest")])
    store.persist("c1")

    reloaded_store = VectorStore(persist_dir=str(tmp_path))
    reloaded = reloaded_store.get_or_create_collection("c1", dimension=3)
    assert reloaded.index.ntotal == 1
    assert reloaded.chunks[0].text == "second ingest"
