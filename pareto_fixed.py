#!/usr/bin/env python3
"""Pareto plot: inference speed vs quality at multiple compression levels.

Supports two Y-axis modes:
  --metric recall    (legacy TF-IDF recall)
  --metric composite (LLM-as-Judge composite score)

128K token conversation prefix, budget at 3K.
Includes Claude Code /compact as a generative baseline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RESULTS_FILE = Path("pareto_fixed_results.json")
LLM_RESULTS_FILE = Path("llm_eval_results.json")
OUTPUT_FILE = Path("pareto_fixed.png")

METHOD_STYLES = {
    "dedup":        {"color": "#e74c3c", "marker": "s",  "label": "Dedup (suffix automaton)"},
    "llama-embed":  {"color": "#3498db", "marker": "D",  "label": "Llama-embed (Qwen3-Embed-0.6B)"},
    "llama-rerank": {"color": "#2ecc71", "marker": "^",  "label": "Llama-rerank (Qwen3-Rerank-0.6B)"},
    "claude-code":  {"color": "#f39c12", "marker": "*",  "label": "Claude Code /compact"},
}


def plot_pareto(results: list[dict], output_path: Path, metric: str = "recall"):
    y_key = metric  # "recall" or "composite"
    y_label = {
        "recall": "Recall (TF-IDF vocabulary overlap)  \u2192  higher is better",
        "composite": "Composite Score (LLM-as-Judge)  \u2192  higher is better",
    }.get(metric, metric)

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

    # Filter to results that have the metric
    valid = [r for r in results if y_key in r]
    if not valid:
        print(f"No results with metric '{y_key}' found")
        return

    # Group by method, connect budget points with lines
    methods_seen = {}
    for r in valid:
        method = r["method"]
        style = METHOD_STYLES.get(method, {"color": "#888", "marker": "o", "label": method})

        label = style["label"] if method not in methods_seen else None
        methods_seen[method] = True

        marker_size = 250 if method == "claude-code" else 150
        ax.scatter(
            r["speed_s"], r[y_key],
            c=style["color"], marker=style["marker"],
            s=marker_size, label=label, zorder=5,
            edgecolors="white", linewidths=0.8,
        )

        budget = r.get("budget", "?")
        if isinstance(budget, int):
            budget_label = f"{budget // 1000}K" if budget >= 1000 else f"{budget}"
        else:
            budget_label = str(budget)

        # Annotation depends on available fields
        if "kept_tokens" in r:
            kept_k = r["kept_tokens"] / 1000
            note = f'budget={budget_label}\n{kept_k:.1f}K kept'
        else:
            note = f'budget={budget_label}'

        if "model_label" in r:
            note += f'\n{r["model_label"]}'

        ax.annotate(
            note,
            (r["speed_s"], r[y_key]),
            textcoords="offset points",
            xytext=(14, -4),
            fontsize=8,
            color=style["color"],
            alpha=0.9,
        )

    # Connect same-method points across budgets
    for method in METHOD_STYLES:
        pts = [r for r in valid if r["method"] == method]
        if len(pts) >= 2:
            pts_sorted = sorted(pts, key=lambda p: p["speed_s"])
            style = METHOD_STYLES[method]
            ax.plot(
                [p["speed_s"] for p in pts_sorted],
                [p[y_key] for p in pts_sorted],
                color=style["color"], alpha=0.3, linewidth=1.5,
                linestyle="--", zorder=3,
            )

    ax.set_xscale("log")
    ax.set_xlabel("Inference Speed (seconds, log scale)  \u2192  lower is better", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)

    metric_display = "LLM Composite" if metric == "composite" else "TF-IDF Recall"
    ax.set_title(
        f"Compaction Methods: Speed vs {metric_display}\n"
        f"128K token prefix  |  budget: 3K",
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
    parser = argparse.ArgumentParser(description="Generate Pareto frontier plot")
    parser.add_argument("--metric", choices=["recall", "composite"], default="recall",
                        help="Y-axis metric (default: recall)")
    parser.add_argument("--results", type=Path, default=None,
                        help="Results JSON file (auto-detected from metric if not given)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output PNG path")
    args = parser.parse_args()

    if args.results:
        results_file = args.results
    elif args.metric == "composite":
        results_file = LLM_RESULTS_FILE
    else:
        results_file = RESULTS_FILE

    output_file = args.output or Path(f"pareto_{args.metric}.png")

    results = json.loads(results_file.read_text())
    print(f"Loaded {len(results)} results from {results_file}")
    plot_pareto(results, output_file, metric=args.metric)


if __name__ == "__main__":
    main()
