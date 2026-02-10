#!/usr/bin/env python3
"""Pareto plot for v2 evaluation metrics: Entity Preservation.

Plots compaction speed (x) vs weighted entity coverage (y) for each
method and budget level. Uses the same dark theme as pareto_dual.py.
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


def plot_entity_coverage(ax, results, show_legend=True):
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="#c9d1d9", labelsize=10)
    ax.xaxis.label.set_color("#c9d1d9")
    ax.yaxis.label.set_color("#c9d1d9")
    ax.title.set_color("#e6edf3")
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.grid(True, alpha=0.15, color="#484f58")

    methods_seen = {}

    for r in results:
        method = r["method"]
        style = METHOD_STYLES.get(method, {"color": "#888", "marker": "o", "label": method})
        label = style["label"] if method not in methods_seen else None
        methods_seen[method] = True

        y_val = r["weighted_coverage"]
        size = 300 if method == "claude-code" else 180

        ax.scatter(
            r["speed_s"], y_val,
            c=style["color"], marker=style["marker"],
            s=size, label=label, zorder=5,
            edgecolors="white", linewidths=0.8,
        )

        budget = r.get("budget", "?")
        budget_label = f"{budget // 1000}K" if isinstance(budget, int) and budget >= 1000 else str(budget)
        kept = r.get("kept_tokens", 0)
        kept_label = f"{kept / 1000:.0f}K kept" if kept >= 1000 else str(kept)

        ax.annotate(
            f"{budget_label}\n{kept_label}",
            (r["speed_s"], y_val),
            textcoords="offset points", xytext=(12, -4),
            fontsize=7, color=style["color"], alpha=0.85,
        )

    # Connect same-method points
    for method, style in METHOD_STYLES.items():
        pts = [r for r in results if r["method"] == method]
        if len(pts) >= 2:
            pts.sort(key=lambda p: p["speed_s"])
            ax.plot(
                [p["speed_s"] for p in pts],
                [p["weighted_coverage"] for p in pts],
                color=style["color"], alpha=0.25, linewidth=1.5,
                linestyle="--", zorder=3,
            )

    ax.set_xlabel("Compaction Speed (s)  \u2192  lower is better", fontsize=11)
    ax.set_ylabel("Entity Coverage (weighted)  \u2192  higher is better", fontsize=11)
    ax.set_title(
        "Entity Preservation: Speed vs Coverage\n"
        "Dedup method, 40K\u2013200K budgets, 682K token conversation",
        fontsize=13, fontweight="bold", color="#e6edf3",
    )

    if show_legend:
        ax.legend(
            loc="lower right", fontsize=9,
            facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9",
        )


def plot_type_breakdown(ax, results):
    """Bar chart of entity coverage by type for the largest budget."""
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="#c9d1d9", labelsize=9)
    ax.xaxis.label.set_color("#c9d1d9")
    ax.yaxis.label.set_color("#c9d1d9")
    ax.title.set_color("#e6edf3")
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.grid(True, alpha=0.15, color="#484f58", axis="y")

    # Use the largest budget result
    largest = max(results, key=lambda r: r.get("budget", 0))
    tc = largest.get("type_coverage", {})

    if not tc:
        return

    types = sorted(tc.keys(), key=lambda t: tc[t]["coverage"], reverse=True)
    coverages = [tc[t]["coverage"] for t in types]
    counts = [f"{tc[t]['covered']}/{tc[t]['total']}" for t in types]

    colors = ["#e74c3c" if tc[t]["weight"] >= 0.8 else
              "#f39c12" if tc[t]["weight"] >= 0.6 else
              "#3498db" for t in types]

    bars = ax.barh(range(len(types)), coverages, color=colors, alpha=0.8, edgecolor="#30363d")

    ax.set_yticks(range(len(types)))
    ax.set_yticklabels([f"{t}  ({counts[i]})" for i, t in enumerate(types)],
                       fontsize=9, color="#c9d1d9")
    ax.set_xlabel("Coverage  \u2192  higher is better", fontsize=10)
    ax.set_title(
        f"Entity Type Breakdown (budget={largest['budget'] // 1000}K)",
        fontsize=12, fontweight="bold", color="#e6edf3",
    )
    ax.set_xlim(0, max(coverages) * 1.3 if coverages else 1)
    ax.invert_yaxis()

    # Add value labels
    for bar, cov in zip(bars, coverages):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{cov:.1%}", va="center", fontsize=8, color="#c9d1d9")


def main():
    # Load all v2 results
    results = json.loads(Path("eval_v2_all_results.json").read_text())

    # Deduplicate — only keep unique (method, kept_tokens) combinations
    seen = set()
    unique = []
    for r in results:
        key = (r["method"], r["kept_tokens"])
        if key not in seen:
            seen.add(key)
            unique.append(r)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    fig.patch.set_facecolor("#0d1117")

    plot_entity_coverage(ax1, unique, show_legend=True)
    plot_type_breakdown(ax2, unique)

    plt.tight_layout()
    out = Path("pareto_v2.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved to {out}")
    plt.close()


if __name__ == "__main__":
    main()
