"""
POST /query

Accepts a natural-language question (+ project/collection identifier),
runs it through rag/qa_engine.py, logs the query + retrieved chunk IDs +
answer to SQLite, and returns the answer object: {answer, citations[],
grounded}.
"""
