"""
Real semantic retrieval using Google Gemini's embedding model.

WHY THIS FILE EXISTS (read this first):
The old retrieval.py picked "relevant" chunks by counting how many literal
words overlapped between your question and each chunk. That fails as soon
as you phrase things differently than the contract does (e.g. asking about
"cancelling early" when the contract says "termination for convenience").

An embedding is a list of numbers (a vector) that represents the MEANING
of a piece of text. Two pieces of text that mean similar things end up
with vectors that point in a similar direction. We measure "similar
direction" using cosine similarity (a number from -1 to 1, higher = more
similar). This is the actual technique real RAG systems use.
"""

import numpy as np
from google import genai

EMBED_MODEL = "gemini-embedding-001"


def get_client(api_key: str) -> genai.Client:
    """Create a Gemini client using the given API key."""
    return genai.Client(api_key=api_key)


def embed_texts(client: genai.Client, texts: list[str]) -> np.ndarray:
    """
    Turn a list of text chunks into a list of embedding vectors.
    Returns a 2D numpy array of shape (num_texts, embedding_dim).

    Note: gemini-embedding-001 only accepts ONE text per request, so we
    call it once per chunk rather than sending the whole list at once.
    """
    if not texts:
        return np.array([])

    vectors = []
    for text in texts:
        result = client.models.embed_content(
            model=EMBED_MODEL,
            contents=text,
        )
        vectors.append(result.embeddings[0].values)
    return np.array(vectors)


def embed_query(client: genai.Client, query: str) -> np.ndarray:
    """Embed a single question. Returns a 1D numpy array."""
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=query,
    )
    return np.array(result.embeddings[0].values)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Standard cosine similarity between two vectors."""
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def retrieve_top_k(client: genai.Client, query: str, chunks: list[str],
                    chunk_embeddings: np.ndarray, k: int = 3):
    """
    Find the k chunks whose MEANING is closest to the question's meaning.
    Returns a list of (chunk_text, similarity_score) tuples, best first.
    """
    if not chunks or chunk_embeddings.size == 0:
        return []

    query_vec = embed_query(client, query)
    scores = [cosine_similarity(query_vec, chunk_vec) for chunk_vec in chunk_embeddings]

    ranked = sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)
    return ranked[:k]
