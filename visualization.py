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
