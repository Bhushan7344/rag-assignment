import json
from pathlib import Path

from compare import run_strategy, save_strategy_outputs
from config import get_settings
from embeddings import get_embedding_model
from ingestion import load_documents
from llm import get_llm
from visualization import plot_avg_scores, plot_chunk_counts, plot_query_strategy_heatmap


DEFAULT_QUERIES = [
    "What is the main objective of this document?",
    "List the key points discussed in the document.",
    "What conclusions or recommendations are provided?",
]


def write_summary_report(all_results: list[dict], out_file: Path) -> None:
    payload = {
        "strategies": [
            {
                "strategy": item["strategy"],
                "chunk_count": item["chunk_count"],
                "avg_score": item["avg_score"],
                "query_avg_scores": item["query_avg_scores"],
            }
            for item in all_results
        ],
        "queries": DEFAULT_QUERIES,
    }
    out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    settings = get_settings()
    documents = load_documents(settings.source_document)
    embeddings = get_embedding_model(
        model_name=settings.embedding_model,
        local_files_only=settings.embedding_local_files_only,
    )
    llm = get_llm(base_url=settings.ollama_base_url, model=settings.ollama_model)

    all_results: list[dict] = []
    for strategy in ["fixed", "overlap", "semantic"]:
        result = run_strategy(
            strategy=strategy,
            documents=documents,
            embeddings=embeddings,
            llm=llm,
            chroma_dir=settings.chroma_base_dir,
            top_k=settings.top_k,
            queries=DEFAULT_QUERIES,
        )
        save_strategy_outputs(result=result, output_dir=settings.output_dir)
        all_results.append(result)

    write_summary_report(
        all_results=all_results,
        out_file=settings.output_dir / "comparison_summary.json",
    )

    chunk_counts = {item["strategy"]: item["chunk_count"] for item in all_results}
    avg_scores = {item["strategy"]: item["avg_score"] for item in all_results}

    plot_chunk_counts(chunk_counts, settings.output_dir / "charts" / "chunk_counts.png")
    plot_avg_scores(avg_scores, settings.output_dir / "charts" / "avg_scores.png")
    per_query_scores = {
        item["strategy"]: item["query_avg_scores"] for item in all_results
    }
    plot_query_strategy_heatmap(
        per_query_scores, settings.output_dir / "charts" / "query_strategy_scores.png"
    )

    print("Completed RAG chunking comparison.")
    print(f"Outputs saved under: {settings.output_dir}")


if __name__ == "__main__":
    main()
