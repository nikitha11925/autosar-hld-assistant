"""
FAISS-backed vector store.

Persists chunk embeddings locally (see data/chroma_db/, name kept from the
original ChromaDB plan) and provides similarity search. Enforces
project/document isolation: each ingested document (or document set) gets
its own collection — collections are never silently merged across projects
(see CLAUDE.md, design principle 4).

ChromaDB was the originally planned store, but its chroma-hnswlib
dependency requires a C++ compiler toolchain that is not available on this
dev machine (no prebuilt wheel for this Python/OS combination). FAISS is
listed in CLAUDE.md as an acceptable alternative local vector store, has
prebuilt wheels, and is wrapped behind this same interface — so swapping
back to ChromaDB later only touches this file.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import faiss
import numpy as np

DEFAULT_PERSIST_DIR = os.environ.get("FAISS_PERSIST_DIR", "./data/chroma_db")


@dataclass
class StoredChunk:
    """Metadata for one embedded chunk, stored alongside its vector."""

    chunk_id: str
    text: str
    doc_name: str
    section: str | None
    page_start: int
    page_end: int


@dataclass
class Match:
    """One retrieval result: the stored chunk plus its similarity score."""

    chunk: StoredChunk
    score: float  # cosine similarity, [-1, 1]; higher is more similar


@dataclass
class Collection:
    """One project/document's isolated vector index."""

    name: str
    dimension: int
    index: faiss.Index = field(repr=False)
    chunks: list[StoredChunk] = field(default_factory=list)

    def add(self, embeddings: list[list[float]], chunks: list[StoredChunk]) -> None:
        if len(embeddings) != len(chunks):
            raise ValueError("embeddings and chunks must be the same length")
        if not embeddings:
            return
        vectors = np.array(embeddings, dtype="float32")
        self.index.add(vectors)
        self.chunks.extend(chunks)

    def query(self, embedding: list[float], top_k: int = 5) -> list[Match]:
        if self.index.ntotal == 0:
            return []
        vector = np.array([embedding], dtype="float32")
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(vector, k)
        matches: list[Match] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            matches.append(Match(chunk=self.chunks[idx], score=float(score)))
        return matches


class VectorStore:
    """Manages one isolated FAISS collection per document/project."""

    def __init__(self, persist_dir: str = DEFAULT_PERSIST_DIR) -> None:
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self._collections: dict[str, Collection] = {}

    def _paths(self, collection_name: str) -> tuple[str, str]:
        safe_name = collection_name.replace("/", "_").replace("\\", "_")
        index_path = os.path.join(self.persist_dir, f"{safe_name}.faiss")
        meta_path = os.path.join(self.persist_dir, f"{safe_name}.meta.json")
        return index_path, meta_path

    def get_or_create_collection(self, collection_name: str, dimension: int) -> Collection:
        if collection_name in self._collections:
            return self._collections[collection_name]

        loaded = self._load_from_disk(collection_name, dimension)
        if loaded is not None:
            self._collections[collection_name] = loaded
            return loaded

        index = faiss.IndexFlatIP(dimension)  # inner product on normalized vectors == cosine similarity
        collection = Collection(name=collection_name, dimension=dimension, index=index)
        self._collections[collection_name] = collection
        return collection

    def reset_collection(self, collection_name: str, dimension: int) -> Collection:
        """Replace a collection with an empty one — used on (re-)ingest so that
        ingesting the same document twice overwrites rather than duplicates its
        chunks (see CLAUDE.md design principle 4: no silent cross-ingest merging)."""
        index = faiss.IndexFlatIP(dimension)
        collection = Collection(name=collection_name, dimension=dimension, index=index)
        self._collections[collection_name] = collection
        return collection

    def persist(self, collection_name: str) -> None:
        collection = self._collections.get(collection_name)
        if collection is None:
            return
        index_path, meta_path = self._paths(collection_name)
        faiss.write_index(collection.index, index_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "dimension": collection.dimension,
                    "chunks": [
                        {
                            "chunk_id": c.chunk_id,
                            "text": c.text,
                            "doc_name": c.doc_name,
                            "section": c.section,
                            "page_start": c.page_start,
                            "page_end": c.page_end,
                        }
                        for c in collection.chunks
                    ],
                },
                f,
            )

    def _load_from_disk(self, collection_name: str, dimension: int) -> Collection | None:
        index_path, meta_path = self._paths(collection_name)
        if not (os.path.exists(index_path) and os.path.exists(meta_path)):
            return None
        index = faiss.read_index(index_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta["dimension"] != dimension:
            return None
        chunks = [
            StoredChunk(
                chunk_id=c["chunk_id"],
                text=c["text"],
                doc_name=c["doc_name"],
                section=c["section"],
                page_start=c["page_start"],
                page_end=c["page_end"],
            )
            for c in meta["chunks"]
        ]
        return Collection(name=collection_name, dimension=dimension, index=index, chunks=chunks)

    def list_collections(self) -> list[str]:
        names = set(self._collections.keys())
        if os.path.isdir(self.persist_dir):
            for fname in os.listdir(self.persist_dir):
                if fname.endswith(".faiss"):
                    names.add(fname[: -len(".faiss")])
        return sorted(names)


_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store
