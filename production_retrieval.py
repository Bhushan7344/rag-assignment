from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Iterable

from langchain_chroma import Chroma
from langchain_core.documents import Document
from sklearn.feature_extraction.text import TfidfVectorizer


WORD_RE = re.compile(r"[a-zA-Z0-9_]+")


@dataclass
class ScoredChunk:
    text: str
    metadata: dict
    vector_distance: float | None
    vector_similarity: float
    keyword_score: float
    hybrid_score: float
    rerank_score: float

    def to_dict(self) -> dict:
        return asdict(self)


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def tokenize(text: str) -> set[str]:
    return {token.lower() for token in WORD_RE.findall(text)}


def classify_section(text: str) -> str:
    lowered = text.lower()
    if any(key in lowered for key in ("conclusion", "recommendation", "summary")):
        return "conclusion"
    if any(key in lowered for key in ("objective", "goal", "purpose")):
        return "objective"
    if any(key in lowered for key in ("key point", "important point", "highlights")):
        return "key_points"
    return "general"


def enrich_chunk_metadata(chunks: Iterable[Document], source_path: str) -> list[Document]:
    doc_type = source_path.rsplit(".", maxsplit=1)[-1].lower() if "." in source_path else "unknown"
    enriched: list[Document] = []
    for chunk in chunks:
        metadata = dict(chunk.metadata)
        metadata["document_type"] = doc_type
        metadata["section"] = classify_section(chunk.page_content)
        metadata.setdefault("source_name", source_path.split("/")[-1])
        enriched.append(Document(page_content=chunk.page_content, metadata=metadata))
    return enriched


def _preferred_filters_for_query(query: str) -> dict:
    lowered = query.lower()
    section = None
    if any(word in lowered for word in ("objective", "goal", "purpose")):
        section = "objective"
    elif any(word in lowered for word in ("key point", "important point", "highlights")):
        section = "key_points"
    elif any(word in lowered for word in ("conclusion", "recommendation", "final")):
        section = "conclusion"
    return {"section": section}


def _normalize(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    lo = min(values.values())
    hi = max(values.values())
    if hi - lo < 1e-12:
        return {key: 1.0 for key in values}
    return {key: (value - lo) / (hi - lo) for key, value in values.items()}


def _apply_metadata_filter(
    chunks_by_key: dict[str, ScoredChunk], query: str
) -> tuple[dict[str, ScoredChunk], dict]:
    filters = _preferred_filters_for_query(query)
    wanted_section = filters.get("section")
    if not wanted_section:
        return chunks_by_key, {"applied": False, "reason": "no section intent"}

    filtered = {
        key: chunk
        for key, chunk in chunks_by_key.items()
        if chunk.metadata.get("section") == wanted_section
    }
    if filtered:
        return filtered, {
            "applied": True,
            "section": wanted_section,
            "remaining_candidates": len(filtered),
        }
    return chunks_by_key, {
        "applied": False,
        "section": wanted_section,
        "reason": "no matching chunks found",
    }


def retrieve_vector_baseline(
    vector_store: Chroma, query: str, k: int
) -> list[tuple[Document, float]]:
    return vector_store.similarity_search_with_score(query=query, k=k)


def retrieve_hybrid_with_rerank(
    vector_store: Chroma, query: str, k: int
) -> tuple[list[ScoredChunk], dict]:
    raw_results = vector_store.similarity_search_with_score(query=query, k=max(k * 6, 24))
    deduped: dict[str, ScoredChunk] = {}

    for doc, distance in raw_results:
        key = normalize_text(doc.page_content)
        if key in deduped:
            continue
        sim = 1.0 / (1.0 + float(distance))
        deduped[key] = ScoredChunk(
            text=doc.page_content,
            metadata=dict(doc.metadata),
            vector_distance=float(distance),
            vector_similarity=sim,
            keyword_score=0.0,
            hybrid_score=0.0,
            rerank_score=0.0,
        )

    if not deduped:
        return [], {"raw_candidates": 0}

    corpus = list(deduped.keys())
    tfidf = TfidfVectorizer(stop_words="english")
    matrix = tfidf.fit_transform(corpus)
    query_vec = tfidf.transform([query])
    keyword_scores_dense = (matrix @ query_vec.T).toarray().ravel()

    keyword_scores = {text: float(score) for text, score in zip(corpus, keyword_scores_dense)}
    normalized_keyword = _normalize(keyword_scores)

    vector_scores = {text: chunk.vector_similarity for text, chunk in deduped.items()}
    normalized_vector = _normalize(vector_scores)

    query_tokens = tokenize(query)
    for text, chunk in deduped.items():
        chunk.keyword_score = normalized_keyword.get(text, 0.0)
        vec = normalized_vector.get(text, 0.0)
        chunk.hybrid_score = (0.65 * vec) + (0.35 * chunk.keyword_score)

        chunk_tokens = tokenize(text)
        overlap = len(query_tokens & chunk_tokens) / max(len(query_tokens), 1)
        chunk.rerank_score = (0.75 * chunk.hybrid_score) + (0.25 * overlap)

    filtered_chunks, filter_debug = _apply_metadata_filter(deduped, query=query)
    ranked = sorted(filtered_chunks.values(), key=lambda item: item.rerank_score, reverse=True)
    return ranked[:k], {
        "raw_candidates": len(raw_results),
        "deduped_candidates": len(deduped),
        "metadata_filter": filter_debug,
    }
