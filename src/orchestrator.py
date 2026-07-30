"""
Orchestrator Agent.

WHY THIS FILE EXISTS:
Previously, the README/architecture diagram described an "Orchestrator
Agent" but no such thing existed in code — app.py just called functions
directly, one after another. This file is that missing piece: it decides
WHICH specialist agent(s) to call for a given user action, and stitches
their outputs into one grounded response. This is the smallest honest
version of "orchestration" — a coordinator function, not a hardcoded
sequence baked into the UI code.
"""

from src.embeddings import retrieve_top_k
from src.llm import summarizer_agent, risk_agent, qa_agent


def handle_question(client, question, chunks, chunk_embeddings, k=3):
    """
    Full pipeline for one user question:
    1. Retrieve the most semantically relevant chunks (Retrieval Engine)
    2. Ask the Summarizer Agent for a plain-language summary
    3. Ask the Risk Agent to flag risky clauses in that same context
    4. Ask the QA Agent to directly answer the user's question
    Returns a dict with all three pieces, grounded in the same evidence.
    """
    retrieved = retrieve_top_k(client, question, chunks, chunk_embeddings, k=k)
    retrieved_chunks = [chunk for chunk, _score in retrieved]

    summary = summarizer_agent(client, retrieved_chunks)
    risks = risk_agent(client, retrieved_chunks)
    answer = qa_agent(client, question, retrieved_chunks)

    return {
        "answer": answer,
        "summary": summary,
        "risks": risks,
        "sources": retrieved_chunks,
    }


def handle_quick_scan(client, chunks, chunk_embeddings, k=3):
    """
    For the sidebar 'quick risk scan' — run risk analysis over the
    top few chunks of the document without a specific user question.
    """
    top_chunks = chunks[:k] if chunks else []
    return risk_agent(client, top_chunks)
