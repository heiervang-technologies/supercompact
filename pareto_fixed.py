#!/usr/bin/env python3
"""Pareto plot: inference speed vs recall at multiple compression levels.

128K token conversation prefix, budgets at 20K and 2K.
Includes Claude Code /compact as a generative baseline.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RESULTS_FILE = Path("pareto_fixed_results.json")
OUTPUT_FILE = Path("pareto_fixed.png")

METHOD_STYLES = {
    "dedup":        {"color": "#e74c3c", "marker": "s",  "label": "Dedup (suffix automaton)"},
    "llama-embed":  {"color": "#3498db", "marker": "D",  "label": "Llama-embed (Qwen3-Embed-0.6B)"},
    "llama-rerank": {"color": "#2ecc71", "marker": "^",  "label": "Llama-rerank (Qwen3-Rerank-0.6B)"},
    "claude-code":  {"color": "#f39c12", "marker": "*",  "label": "Claude Code /compact"},
}


def plot_pareto(results: list[dict], output_path: Path):
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="#c9d1d9", labelsize=11)
    ax.xaxis.label.set_color("#c9d1d9")
    ax.yaxis.label.set_color("#c9d1d9")
    ax.title.set_color("#e6edf3")
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.grid(True, alpha=0.15, color="#484f58")

    # Group by method, connect budget points with lines
    methods_seen = {}
    for r in results:
        method = r["method"]
        style = METHOD_STYLES.get(method, {"color": "#888", "marker": "o", "label": method})

        # Only add label once per method
        label = style["label"] if method not in methods_seen else None
        methods_seen[method] = True

        marker_size = 250 if method == "claude-code" else 150
        ax.scatter(
            r["speed_s"], r["recall"],
            c=style["color"], marker=style["marker"],
            s=marker_size, label=label, zorder=5,
            edgecolors="white", linewidths=0.8,
        )

        # Annotate with budget/kept info
        budget = r.get("budget", "?")
        kept_k = r["kept_tokens"] / 1000
        if isinstance(budget, int):
            budget_label = f"{budget // 1000}K" if budget >= 1000 else f"{budget}"
        else:
            budget_label = str(budget)

        ax.annotate(
            f'budget={budget_label}\n{kept_k:.1f}K kept',
            (r["speed_s"], r["recall"]),
            textcoords="offset points",
            xytext=(14, -4),
            fontsize=8,
            color=style["color"],
            alpha=0.9,
        )

    # Connect same-method points across budgets
    for method in METHOD_STYLES:
        pts = [r for r in results if r["method"] == method]
        if len(pts) >= 2:
            pts_sorted = sorted(pts, key=lambda p: p["speed_s"])
            style = METHOD_STYLES[method]
            ax.plot(
                [p["speed_s"] for p in pts_sorted],
                [p["recall"] for p in pts_sorted],
                color=style["color"], alpha=0.3, linewidth=1.5,
                linestyle="--", zorder=3,
            )

    # Draw a horizontal line showing the extractive floor at 2K budget
    floor_results = [r for r in results if r.get("budget") == 2000]
    if floor_results:
        floor_recall = floor_results[0]["recall"]
        ax.axhline(
            y=floor_recall, color="#484f58", linewidth=1,
            linestyle=":", alpha=0.6, zorder=2,
        )
        ax.annotate(
            f"extractive floor ({floor_results[0]['kept_tokens']//1000}K mandatory turns)",
            xy=(0.02, floor_recall),
            xycoords=("axes fraction", "data"),
            fontsize=8, color="#8b949e",
            va="bottom",
            xytext=(0, 4),
            textcoords="offset points",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Inference Speed (seconds, log scale)  →  lower is better", fontsize=12)
    ax.set_ylabel("Recall (information preservation)  →  higher is better", fontsize=12)

    total_tok = results[0]["total_tokens"] if results else 0
    ax.set_title(
        f"Compaction Methods: Speed vs Recall\n"
        f"128K token prefix  |  budgets: 2K and 20K",
        fontsize=14, fontweight="bold",
    )

    ax.legend(
        loc="upper left", fontsize=10,
        facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9",
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Plot saved to {output_path}")
    plt.close()


def main():
    results = json.loads(RESULTS_FILE.read_text())
    print(f"Loaded {len(results)} results from {RESULTS_FILE}")
    plot_pareto(results, OUTPUT_FILE)


if __name__ == "__main__":
    main()
