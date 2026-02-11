#!/usr/bin/env python3
"""Side-by-side Pareto: TF-IDF Recall vs LLM Composite.

Left panel: recall (all methods, 2K + 20K budgets)
Right panel: composite (available data: dedup + llama-embed, 3K + 8K)
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHOD_STYLES = {
    "dedup":        {"color": "#e74c3c", "marker": "s",  "label": "Dedup"},
    "llama-embed":  {"color": "#3498db", "marker": "D",  "label": "Llama-embed"},
    "llama-rerank": {"color": "#2ecc71", "marker": "^",  "label": "Llama-rerank"},
    "claude-code":  {"color": "#f39c12", "marker": "*",  "label": "Claude Code /compact"},
}


def plot_panel(ax, results, y_key, y_label, title, show_legend=False):
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="#c9d1d9", labelsize=10)
    ax.xaxis.label.set_color("#c9d1d9")
    ax.yaxis.label.set_color("#c9d1d9")
    ax.title.set_color("#e6edf3")
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.grid(True, alpha=0.15, color="#484f58")

    valid = [r for r in results if r.get(y_key) is not None]
    methods_seen = {}

    for r in valid:
        method = r["method"]
        model_key = r.get("model_key", "capable")
        if model_key not in ("capable", None):
            continue  # skip cheap model for cleaner plot

        style = METHOD_STYLES.get(method, {"color": "#888", "marker": "o", "label": method})
        label = style["label"] if method not in methods_seen else None
        methods_seen[method] = True

        size = 300 if method == "claude-code" else 180
        ax.scatter(
            r["speed_s"], r[y_key],
            c=style["color"], marker=style["marker"],
            s=size, label=label, zorder=5,
            edgecolors="white", linewidths=0.8,
        )

        budget = r.get("budget", "?")
        budget_label = f"{budget // 1000}K" if isinstance(budget, int) and budget >= 1000 else str(budget)

        parts = [f"{budget_label}"]
        if "kept_tokens" in r:
            parts.append(f"{r['kept_tokens'] / 1000:.1f}K kept")
        note = "\n".join(parts)

        ax.annotate(
            note, (r["speed_s"], r[y_key]),
            textcoords="offset points", xytext=(12, -4),
            fontsize=7, color=style["color"], alpha=0.85,
        )

    # Connect same-method points
    for method, style in METHOD_STYLES.items():
        pts = [r for r in valid if r["method"] == method and r.get("model_key", "capable") in ("capable", None)]
        if len(pts) >= 2:
            pts.sort(key=lambda p: p["speed_s"])
            ax.plot(
                [p["speed_s"] for p in pts],
                [p[y_key] for p in pts],
                color=style["color"], alpha=0.25, linewidth=1.5,
                linestyle="--", zorder=3,
            )

    ax.set_xscale("log")
    ax.set_xlabel("Compaction Speed (s, log)  →  lower is better", fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", color="#e6edf3")

    if show_legend:
        ax.legend(
            loc="lower left", fontsize=8,
            facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9",
        )


def main():
    recall_data = json.loads(Path("pareto_fixed_results.json").read_text())
    composite_data = json.loads(Path("llm_eval_merged.json").read_text())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    fig.patch.set_facecolor("#0d1117")

    plot_panel(ax1, recall_data,
               "recall",
               "TF-IDF Recall  →  higher is better",
               "TF-IDF Recall (all methods)\n2K + 20K budgets, 128K prefix",
               show_legend=True)

    plot_panel(ax2, composite_data,
               "composite",
               "LLM Composite  →  higher is better",
               "LLM-as-Judge Composite (Opus-4.5)\n3K + 8K budgets, 82K prefix")

    fig.suptitle(
        "Pareto Frontiers: Speed vs Quality — Two Metrics Compared",
        fontsize=15, fontweight="bold", color="#e6edf3", y=1.02,
    )

    plt.tight_layout()
    out = Path("pareto_dual.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved to {out}")
    plt.close()


if __name__ == "__main__":
    main()
