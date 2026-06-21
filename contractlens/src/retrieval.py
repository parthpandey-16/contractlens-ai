def retrieve_top_k(query, chunks, k=3):

    if not chunks:
        return []

    query_words = set(query.lower().split())

    scores = []

    for chunk in chunks:

        chunk_words = set(chunk.lower().split())

        score = len(query_words.intersection(chunk_words))

        scores.append((chunk, score))

    scores.sort(key=lambda x: x[1], reverse=True)

    return [(chunk, score) for chunk, score in scores[:k] if score > 0]