#!/usr/bin/env python3
"""Reproduce compact reference plots from versioned TSV figure data."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
OUT = FIGURES / "reproduced"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    protocol = pd.read_csv(FIGURES / "protocol_macro_comparison_figure_data.tsv", sep="	")
    pivot = protocol.pivot(index="candidate_size", columns="protocol", values="ndcg10")
    ax = pivot.plot(marker="o", figsize=(7, 4))
    ax.set_xlabel("Candidate size")
    ax.set_ylabel("nDCG@10")
    ax.set_title("Protocol comparison")
    ax.grid(True, alpha=0.25)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(OUT / "protocol_macro_comparison_reproduced.png", dpi=180)
    plt.close(fig)

    degree = pd.read_csv(FIGURES / "graph_gain_by_degree_figure_data.tsv", sep="	")
    summary = degree.groupby(["protocol", "method"], as_index=False)["ndcg10"].mean()
    ax = summary.pivot(index="protocol", columns="method", values="ndcg10").plot(kind="bar", figsize=(7, 4))
    ax.set_xlabel("Protocol")
    ax.set_ylabel("Mean nDCG@10 across degree bins")
    ax.set_title("Degree-stratified reference comparison")
    ax.grid(True, axis="y", alpha=0.25)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(OUT / "graph_gain_by_degree_reproduced.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
