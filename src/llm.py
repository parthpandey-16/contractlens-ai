"""
The actual "agents" — each one is a distinct, purposeful call to the
Gemini LLM with its own prompt and job. This is what makes the system
genuinely "agentic": each function has ONE clear responsibility, and the
orchestrator (see agents.py) decides which ones to call and in what order.

WHY THIS REPLACES THE OLD agents.py:
The old version never called any AI model. summarize_chunks() just took
the first few sentences of text, and risk_agent() just checked if literal
words like "indemnify" appeared. That works only for the exact phrases it
was hardcoded to look for, and understands nothing. These functions
instead send the retrieved contract text to Gemini and ask it to reason
about it, the way a real assistant would.
"""

from google import genai

CHAT_MODEL = "gemini-flash-latest"


def _ask_gemini(client: genai.Client, prompt: str) -> str:
    """Send one prompt to Gemini and return its text response."""
    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=prompt,
    )
    return response.text.strip()


def summarizer_agent(client: genai.Client, retrieved_chunks: list[str]) -> str:
    """Ask Gemini to summarize the retrieved contract excerpts in plain language."""
    if not retrieved_chunks:
        return "No relevant content found in the contract for this query."

    context = "\n\n---\n\n".join(retrieved_chunks)
    prompt = f"""You are a legal-document summarization assistant.
Summarize the following contract excerpts in 2-4 simple, plain-English
sentences a non-lawyer could understand. Do not invent details that are
not present in the text.

CONTRACT EXCERPTS:
{context}

SUMMARY:"""
    return _ask_gemini(client, prompt)


def risk_agent(client: genai.Client, retrieved_chunks: list[str]) -> str:
    """Ask Gemini to identify risky or one-sided clauses in the excerpts."""
    if not retrieved_chunks:
        return "No text available to analyze for risk."

    context = "\n\n---\n\n".join(retrieved_chunks)
    prompt = f"""You are a contract risk-analysis assistant.
Read the excerpts below and list any clauses that could be risky or
one-sided for the party signing the contract (e.g. automatic renewal,
unlimited liability, unilateral termination, broad indemnification).
For each risk found, give a one-line plain-English explanation.
If you find no such clauses in the given text, say so clearly instead
of inventing risks.

CONTRACT EXCERPTS:
{context}

RISK FINDINGS:"""
    return _ask_gemini(client, prompt)


def qa_agent(client: genai.Client, question: str, retrieved_chunks: list[str]) -> str:
    """Ask Gemini to answer the user's question, grounded only in the retrieved excerpts."""
    if not retrieved_chunks:
        return "I could not find relevant clauses in the uploaded document to answer this."

    context = "\n\n---\n\n".join(
        f"[Excerpt {i+1}] {chunk}" for i, chunk in enumerate(retrieved_chunks)
    )
    prompt = f"""You are a legal-document question-answering assistant.
Answer the question below using ONLY the contract excerpts provided.
If the excerpts do not contain enough information to answer, say so
honestly instead of guessing.

QUESTION: {question}

CONTRACT EXCERPTS:
{context}

ANSWER (grounded strictly in the excerpts above):"""
    return _ask_gemini(client, prompt)
