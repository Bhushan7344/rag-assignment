import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

from chunking import ChunkingStrategy, split_documents
from retrieval import retrieve_with_scores
from vectordb import build_vector_store
from llm import generate_answer


@dataclass
class RetrievalItem:
    score: float
    text: str
    metadata: dict


def run_strategy(
    strategy: ChunkingStrategy,
    documents,
    embeddings,
    llm,
    chroma_dir: Path,
    top_k: int,
    queries: list[str],
) -> dict:
    chunks = split_documents(documents=documents, strategy=strategy, embedding_model=embeddings)
    store = build_vector_store(
        documents=chunks,
        embeddings=embeddings,
        persist_dir=chroma_dir,
        collection_name=f"rag_{strategy}",
    )

    results: dict[str, dict] = {}
    all_scores: list[float] = []

    for query in queries:
        retrieved = retrieve_with_scores(store, query=query, k=top_k)
        items: list[RetrievalItem] = []
        context_texts: list[str] = []

        for doc, score in retrieved:
            all_scores.append(score)
            context_texts.append(doc.page_content)
            items.append(
                RetrievalItem(
                    score=score,
                    text=doc.page_content,
                    metadata=doc.metadata,
                )
            )

        answer = generate_answer(llm=llm, query=query, contexts=context_texts)
        results[query] = {
            "retrieved_chunks": [asdict(item) for item in items],
            "answer": answer,
        }

    return {
        "strategy": strategy,
        "chunk_count": len(chunks),
        "avg_score": mean(all_scores) if all_scores else 0.0,
        "queries": results,
    }


def save_strategy_outputs(result: dict, output_dir: Path) -> None:
    strategy = result["strategy"]

    retrieved_path = output_dir / "retrieved_chunks" / f"{strategy}.json"
    answers_path = output_dir / "final_answers" / f"{strategy}.json"

    retrieved_payload = {
        query: data["retrieved_chunks"] for query, data in result["queries"].items()
    }
    answers_payload = {query: data["answer"] for query, data in result["queries"].items()}

    retrieved_path.write_text(json.dumps(retrieved_payload, indent=2), encoding="utf-8")
    answers_path.write_text(json.dumps(answers_payload, indent=2), encoding="utf-8")
