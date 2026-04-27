from langchain_chroma import Chroma


def retrieve_with_scores(
    vector_store: Chroma, query: str, k: int
) -> list[tuple[object, float]]:
    raw_results = vector_store.similarity_search_with_score(query=query, k=max(k * 3, k))
    unique_results: list[tuple[object, float]] = []
    seen_texts: set[str] = set()

    for doc, score in raw_results:
        normalized = " ".join(doc.page_content.split())
        if normalized in seen_texts:
            continue
        seen_texts.add(normalized)
        unique_results.append((doc, score))
        if len(unique_results) == k:
            break

    return unique_results
