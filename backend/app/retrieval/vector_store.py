"""
ChromaDB-backed vector store.

Persists chunk embeddings locally (see data/chroma_db/) and provides
similarity search. Enforces project/document isolation: each ingested
document (or document set) gets its own collection — collections are never
silently merged across projects (see CLAUDE.md, design principle 4).
"""
