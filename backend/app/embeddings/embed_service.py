"""
Embedding service.

Wraps a sentence-transformers model (bge-small-en or e5-small class) behind
a small interface so the embedding model can be swapped without touching
callers. Used both at ingest time (embedding chunks for storage) and at
query time (embedding the user's question for similarity search).
"""
