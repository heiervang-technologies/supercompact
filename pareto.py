#!/usr/bin/env python3
"""Pareto frontier plot: inference speed vs retrieval recall for compaction methods.

Runs fitness evaluation across multiple budget levels for each method,
then plots the Pareto frontier.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lib.parser import parse_jsonl
from lib.fitness import evaluate, FitnessResult


CONVERSATON_FILE = Path("/home/me/.claude/projects/-home-me/52d71008-9b91-4cc0-8d8f-4d62c2fa068b.jsonl")
BUDGETS = [40_000, 60_000, 80_000, 100_000, 120_000, 160_000, 200_000]

METHODS = {
    "dedup": {},
    "llama-embed": {"embed_url": "http://localhost:8082"},
    "llama-rerank": {"rerank_url": "http://localhost:8181"},
}

METHOD_STYLES = {
    "dedup":        {"color": "#e74c3c", "marker": "s", "label": "Dedup (suffix automaton)"},
    "llama-embed":  {"color": "#3498db", "marker": "D", "label": "Llama-embed (Qwen3-Embed-0.6B)"},
    "llama-rerank": {"color": "#2ecc71", "marker": "^", "label": "Llama-rerank (Qwen3-Rerank-0.6B)"},
    "embed":        {"color": "#9b59b6", "marker": "o", "label": "PyTorch embed (Qwen3-Embed-0.6B)"},
}


def pareto_frontier(points: list[tuple[float, float]]) -> list[int]:
    """Return indices of Pareto-optimal points (minimize x, maximize y)."""
    sorted_idx = sorted(range(len(points)), key=lambda i: points[i][0])
    frontier = []
    best_y = -float("inf")
    for i in sorted_idx:
        if points[i][1] > best_y:
            frontier.append(i)
            best_y = points[i][1]
    return frontier


def run_evaluations(turns: list, output_path: Path) -> list[dict]:
    """Run all methods across all budgets, caching results to JSON."""
    if output_path.exists():
        print(f"Loading cached results from {output_path}")
        with open(output_path) as f:
            return json.load(f)

    results = []
    for method, kwargs in METHODS.items():
        for budget in BUDGETS:
            print(f"  {method} @ budget={budget:,}...", end=" ", flush=True)
            try:
                fr = evaluate(
                    turns=turns,
                    method=method,
                    budget=budget,
                    split_ratio=0.70,
                    **kwargs,
                )
                row = {
                    "method": method,
                    "budget": budget,
                    "recall": fr.recall,
                    "speed_s": fr.speed_s,
                    "compression": fr.compression,
                    "f1": fr.f1,
                    "kept_tokens": fr.kept_tokens,
                    "total_tokens": fr.total_tokens,
                    "scored_count": fr.scored_count,
                    "kept_scored": fr.kept_scored,
                }
                results.append(row)
                print(f"recall={fr.recall:.4f}  speed={fr.speed_s:.2f}s")
            except Exception as e:
                print(f"ERROR: {e}")
                continue

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} results to {output_path}")
    return results


def plot_pareto(results: list[dict], output_path: Path):
    """Generate the Pareto frontier plot."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("#0d1117")

    for ax in axes:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#c9d1d9")
        ax.xaxis.label.set_color("#c9d1d9")
        ax.yaxis.label.set_color("#c9d1d9")
        ax.title.set_color("#e6edf3")
        for spine in ax.spines.values():
            spine.set_color("#30363d")
        ax.grid(True, alpha=0.15, color="#484f58")

    # --- Left plot: Speed vs Recall ---
    ax1 = axes[0]
    all_points = []

    for method, style in METHOD_STYLES.items():
        pts = [r for r in results if r["method"] == method]
        if not pts:
            continue
        speeds = [p["speed_s"] for p in pts]
        recalls = [p["recall"] for p in pts]
        budgets = [p["budget"] for p in pts]

        ax1.scatter(speeds, recalls, c=style["color"], marker=style["marker"],
                    s=100, label=style["label"], zorder=5, edgecolors="white",
                    linewidths=0.5)

        # Connect points for same method
        order = np.argsort(budgets)
        ax1.plot([speeds[i] for i in order], [recalls[i] for i in order],
                 color=style["color"], alpha=0.3, linewidth=1.5, zorder=3)

        # Label budget on each point
        for s, r, b in zip(speeds, recalls, budgets):
            all_points.append((s, r, method, b))
            ax1.annotate(f"{b // 1000}k", (s, r), textcoords="offset points",
                         xytext=(6, 6), fontsize=7, color=style["color"], alpha=0.7)

    # Compute and draw Pareto frontier
    coords = [(p[0], p[1]) for p in all_points]
    frontier_idx = pareto_frontier(coords)
    if len(frontier_idx) >= 2:
        frontier_pts = sorted([(coords[i][0], coords[i][1]) for i in frontier_idx])
        fx, fy = zip(*frontier_pts)
        ax1.plot(fx, fy, color="#f0883e", linewidth=2.5, linestyle="--",
                 alpha=0.8, zorder=4, label="Pareto frontier")
        ax1.fill_between(fx, fy, max(fy) * 1.05, alpha=0.05, color="#f0883e")

    ax1.set_xlabel("Inference Speed (seconds) →  lower is better", fontsize=11)
    ax1.set_ylabel("Recall (information preservation) →  higher is better", fontsize=11)
    ax1.set_title("Pareto Frontier: Speed vs Retrieval Recall", fontsize=13, fontweight="bold")
    ax1.legend(loc="lower right", fontsize=9, facecolor="#161b22",
               edgecolor="#30363d", labelcolor="#c9d1d9")

    # --- Right plot: Compression vs Recall (efficiency frontier) ---
    ax2 = axes[1]

    for method, style in METHOD_STYLES.items():
        pts = [r for r in results if r["method"] == method]
        if not pts:
            continue
        comps = [p["compression"] for p in pts]
        recalls = [p["recall"] for p in pts]
        budgets = [p["budget"] for p in pts]

        ax2.scatter(comps, recalls, c=style["color"], marker=style["marker"],
                    s=100, label=style["label"], zorder=5, edgecolors="white",
                    linewidths=0.5)

        order = np.argsort(budgets)
        ax2.plot([comps[i] for i in order], [recalls[i] for i in order],
                 color=style["color"], alpha=0.3, linewidth=1.5, zorder=3)

        for c, r, b in zip(comps, recalls, budgets):
            ax2.annotate(f"{b // 1000}k", (c, r), textcoords="offset points",
                         xytext=(6, 6), fontsize=7, color=style["color"], alpha=0.7)

    # Diagonal reference: recall = compression (random baseline)
    ax2.plot([0, 1], [0, 1], color="#484f58", linewidth=1, linestyle=":",
             alpha=0.5, label="Random baseline (recall ≈ compression)")

    ax2.set_xlabel("Compression ratio (kept / total tokens) →  lower is more compressed", fontsize=11)
    ax2.set_ylabel("Recall (information preservation) →  higher is better", fontsize=11)
    ax2.set_title("Efficiency Frontier: Compression vs Recall", fontsize=13, fontweight="bold")
    ax2.legend(loc="lower right", fontsize=9, facecolor="#161b22",
               edgecolor="#30363d", labelcolor="#c9d1d9")

    fig.suptitle("Supercompact: Compaction Method Tradeoffs\n682k token Claude conversation, 70/30 prefix/suffix split",
                 fontsize=14, fontweight="bold", color="#e6edf3", y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Plot saved to {output_path}")
    plt.close()


def main():
    print(f"Parsing {CONVERSATON_FILE.name}...")
    turns = parse_jsonl(CONVERSATON_FILE)
    print(f"  {len(turns)} turns")

    cache_path = Path("pareto_results.json")
    results = run_evaluations(turns, cache_path)

    plot_path = Path("pareto_frontier.png")
    plot_pareto(results, plot_path)


if __name__ == "__main__":
    main()
