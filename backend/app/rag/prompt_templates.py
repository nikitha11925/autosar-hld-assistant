"""
Prompt templates for the RAG Q&A engine.

Holds the strict grounding prompt: the LLM must answer only from retrieved
context and must not use outside knowledge. Kept separate from qa_engine.py
so prompt wording can be iterated on and evaluated independently of
orchestration logic.
"""
