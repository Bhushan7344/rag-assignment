from langchain_chroma import Chroma


def retrieve_with_scores(
    vector_store: Chroma, query: str, k: int
) -> list[tuple[object, float]]:
    return vector_store.similarity_search_with_score(query=query, k=k)
