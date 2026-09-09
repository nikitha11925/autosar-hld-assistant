"""
Heading-aware chunking.

Splits parsed document content into retrieval-sized chunks, preferring
section/heading boundaries over naive fixed-size windows. Falls back to
~500-token sliding windows only within sections that are themselves too
long to embed as a single chunk.

Every chunk emitted here must carry metadata: {doc_name, section, page_range}
so the RAG layer can produce a citation for any chunk it retrieves.
"""
