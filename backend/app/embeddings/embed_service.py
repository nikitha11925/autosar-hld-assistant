"""
Embedding service.

Wraps a sentence-transformers model (bge-small-en or e5-small class) behind
a small interface so the embedding model can be swapped without touching
callers. Used both at ingest time (embedding chunks for storage) and at
query time (embedding the user's question for similarity search).
"""

from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"


class EmbeddingService:
    """Thin wrapper around a sentence-transformers model."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    @property
    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed chunk texts for storage. bge models expect no special prefix for passages."""
        if not texts:
            return []
        vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a user question. bge models recommend an instruction prefix for queries."""
        prefixed = f"Represent this sentence for searching relevant passages: {text}"
        vector = self._model.encode([prefixed], normalize_embeddings=True, show_progress_bar=False)
        return vector[0].tolist()


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    """Process-wide singleton — loading the model is expensive, do it once."""
    return EmbeddingService()
