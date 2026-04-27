import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path(__file__).resolve().parent / ".mplconfig")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_chunk_counts(counts: dict[str, int], out_path: Path) -> None:
    labels = list(counts.keys())
    values = [counts[label] for label in labels]

    plt.figure(figsize=(8, 4.5))
    plt.bar(labels, values)
    plt.title("Chunk count by strategy")
    plt.xlabel("Strategy")
    plt.ylabel("Number of chunks")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_avg_scores(score_map: dict[str, float], out_path: Path) -> None:
    labels = list(score_map.keys())
    values = [score_map[label] for label in labels]

    plt.figure(figsize=(8, 4.5))
    plt.bar(labels, values)
    plt.title("Average retrieval relevance score")
    plt.xlabel("Strategy")
    plt.ylabel("Avg score")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_query_strategy_heatmap(
    query_score_map: dict[str, dict[str, float]], out_path: Path
) -> None:
    strategies = list(query_score_map.keys())
    if not strategies:
        return

    queries = list(next(iter(query_score_map.values())).keys())
    matrix = [[query_score_map[strategy].get(query, 0.0) for strategy in strategies] for query in queries]

    fig, ax = plt.subplots(figsize=(10, 5))
    heatmap = ax.imshow(matrix, aspect="auto")
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(strategies)
    ax.set_yticks(range(len(queries)))
    ax.set_yticklabels([q[:60] + ("..." if len(q) > 60 else "") for q in queries])
    ax.set_title("Avg retrieval score by query and strategy")
    fig.colorbar(heatmap, ax=ax, label="Score")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
