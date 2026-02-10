#!/usr/bin/env python3
"""Standalone Pareto plot script (uses lib.pareto).

Usage:
    uv run pareto_v2.py                          # reads eval_v2_all_results.json
    uv run compact.py plot eval_results.json      # CLI subcommand (preferred)
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lib.pareto import plot_entity_coverage, plot_type_breakdown


def main():
    results = json.loads(Path("eval_v2_all_results.json").read_text())

    # Deduplicate
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
