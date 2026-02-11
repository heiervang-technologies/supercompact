#!/usr/bin/env python3
"""Pareto plot: inference speed vs quality at multiple compression levels.

Supports two Y-axis modes:
  --metric recall    (legacy TF-IDF recall from pareto_fixed_results.json)
  --metric composite (LLM-as-Judge composite from llm_eval_merged.json)

128K token conversation prefix.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULTS_FILE = Path("pareto_fixed_results.json")
LLM_RESULTS_FILE = Path("llm_eval_merged.json")

METHOD_STYLES = {
    "dedup":        {"color": "#e74c3c", "marker": "s",  "label": "Dedup (suffix automaton)"},
    "llama-embed":  {"color": "#3498db", "marker": "D",  "label": "Llama-embed (Qwen3-Embed-0.6B)"},
    "llama-rerank": {"color": "#2ecc71", "marker": "^",  "label": "Llama-rerank (Qwen3-Rerank-0.6B)"},
    "claude-code":  {"color": "#f39c12", "marker": "*",  "label": "Claude Code /compact"},
}

MODEL_KEY_STYLES = {
    "capable": {"alpha": 1.0, "size_mult": 1.0, "suffix": ""},
    "cheap":   {"alpha": 0.5, "size_mult": 0.6, "suffix": " (cheap)"},
}


def plot_pareto(results: list[dict], output_path: Path, metric: str = "recall"):
    y_key = metric
    y_label = {
        "recall": "Recall (TF-IDF vocabulary overlap)  →  higher is better",
        "composite": "Composite Score (LLM-as-Judge)  →  higher is better",
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

    # Filter to results that have the metric with a non-null value
    valid = [r for r in results if r.get(y_key) is not None]
    if not valid:
        print(f"No results with metric '{y_key}' found")
        return

    # For composite: separate capable vs cheap model; for recall: no model distinction
    methods_seen_capable = {}
    methods_seen_cheap = {}

    for r in valid:
        method = r["method"]
        style = METHOD_STYLES.get(method, {"color": "#888", "marker": "o", "label": method})
        model_key = r.get("model_key", "capable")

        # Skip recall_only entries (no composite score)
        if model_key == "recall_only":
            continue

        ms = MODEL_KEY_STYLES.get(model_key, MODEL_KEY_STYLES["capable"])
        is_cheap = model_key == "cheap"
        seen_map = methods_seen_cheap if is_cheap else methods_seen_capable

        label = None
        if method not in seen_map:
            if is_cheap:
                label = f"{style['label']} (cheap model)"
            else:
                label = style["label"]
            seen_map[method] = True

        base_size = 250 if method == "claude-code" else 150
        marker_size = base_size * ms["size_mult"]

        ax.scatter(
            r["speed_s"], r[y_key],
            c=style["color"], marker=style["marker"],
            s=marker_size, label=label, zorder=5,
            edgecolors="white", linewidths=0.8,
            alpha=ms["alpha"],
        )

        budget = r.get("budget", "?")
        if isinstance(budget, int):
            budget_label = f"{budget // 1000}K" if budget >= 1000 else f"{budget}"
        else:
            budget_label = str(budget)

        kept_k = r["kept_tokens"] / 1000 if "kept_tokens" in r else None
        model_label = r.get("model_label", "")

        parts = [f"budget={budget_label}"]
        if kept_k:
            parts.append(f"{kept_k:.1f}K kept")
        if model_label and metric == "composite":
            parts.append(model_label)
        note = "\n".join(parts)

        # Offset annotations for cheap model to avoid overlap
        offset_y = -18 if is_cheap else -4

        ax.annotate(
            note,
            (r["speed_s"], r[y_key]),
            textcoords="offset points",
            xytext=(14, offset_y),
            fontsize=7 if is_cheap else 8,
            color=style["color"],
            alpha=ms["alpha"] * 0.9,
        )

    # Connect same-method + same-model_key points across budgets
    for model_key in ["capable", "cheap"]:
        for method in METHOD_STYLES:
            pts = [r for r in valid
                   if r["method"] == method
                   and r.get("model_key", "capable") == model_key
                   and r.get(y_key) is not None]
            if len(pts) >= 2:
                pts_sorted = sorted(pts, key=lambda p: p["speed_s"])
                style = METHOD_STYLES[method]
                ms = MODEL_KEY_STYLES.get(model_key, MODEL_KEY_STYLES["capable"])
                ax.plot(
                    [p["speed_s"] for p in pts_sorted],
                    [p[y_key] for p in pts_sorted],
                    color=style["color"], alpha=0.2 * ms["alpha"],
                    linewidth=1.5, linestyle="--", zorder=3,
                )

    ax.set_xscale("log")
    ax.set_xlabel("Compaction Speed (seconds, log scale)  →  lower is better", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)

    metric_display = "LLM Composite" if metric == "composite" else "TF-IDF Recall"

    # Determine budget range from data
    budgets = sorted({r["budget"] for r in valid if r.get(y_key) is not None})
    budget_str = ", ".join(f"{b//1000}K" if b >= 1000 else str(b) for b in budgets)

    ax.set_title(
        f"Compaction Methods: Speed vs {metric_display}\n"
        f"82K token prefix  |  budgets: {budget_str}",
        fontsize=14, fontweight="bold",
    )

    ax.legend(
        loc="best", fontsize=9,
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
