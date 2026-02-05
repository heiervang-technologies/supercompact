"""Rich terminal output and JSON export for evaluation results."""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from .probes import DIMENSIONS
from .aggregate import AggregateResult


console = Console()


def print_results(results: list[AggregateResult]) -> None:
    """Print a rich table of evaluation results to the terminal."""
    if not results:
        console.print("[yellow]No results to display.[/yellow]")
        return

    # Group by method+budget for comparison
    table = Table(
        title="\nLLM-as-Judge Evaluation",
        show_header=True,
        header_style="bold",
    )

    table.add_column("Metric", style="bold")
    for r in results:
        col_name = f"{r.method}\n({r.model_label}, {r.budget:,} budget)"
        table.add_column(col_name, justify="right")

    # Dimension rows
    for dim_name, dim_weight in DIMENSIONS.items():
        label = f"{dim_name} (w={dim_weight:.2f})"
        values = []
        for r in results:
            dm = r.dimension_map
            d = dm.get(dim_name)
            if d and d.probe_count > 0:
                values.append(f"{d.mean_score:.3f}  ({d.probe_count}p)")
            else:
                values.append("—")
        table.add_row(label, *values)

    # Separator
    table.add_row("─" * 20, *["─" * 12 for _ in results])

    # Composite
    table.add_row(
        "[bold]Composite[/bold]",
        *[f"[bold]{r.composite:.3f}[/bold]" for r in results],
    )

    # NDCG
    table.add_row(
        "NDCG (difficulty-weighted)",
        *[f"{r.ndcg:.3f}" for r in results],
    )

    console.print(table)

    # Print model comparison insight if we have both cheap and capable for same method
    methods = {r.method for r in results}
    for method in methods:
        method_results = [r for r in results if r.method == method]
        if len(method_results) >= 2:
            by_key = {r.model_key: r for r in method_results}
            if "cheap" in by_key and "capable" in by_key:
                gap = by_key["capable"].composite - by_key["cheap"].composite
                if gap > 0.1:
                    console.print(
                        f"\n  [yellow]{method}[/yellow]: {gap:.3f} gap between "
                        f"{by_key['capable'].model_label} and {by_key['cheap'].model_label} — "
                        f"context lacks explicit information, capable model compensates."
                    )
                elif gap < 0.02:
                    console.print(
                        f"\n  [green]{method}[/green]: Both models score similarly — "
                        f"context quality is high, model capability doesn't matter."
                    )


def export_json(results: list[AggregateResult], output_path: Path) -> None:
    """Export results as JSON for downstream use (e.g., Pareto plotting)."""
    data = []
    for r in results:
        entry = {
            "method": r.method,
            "budget": r.budget,
            "model_key": r.model_key,
            "model_label": r.model_label,
            "composite": r.composite,
            "ndcg": r.ndcg,
            "dimensions": {
                d.dimension: {
                    "score": d.mean_score,
                    "weight": d.weight,
                    "probe_count": d.probe_count,
                    "raw_scores": d.raw_scores,
                }
                for d in r.dimensions
            },
        }
        data.append(entry)
    output_path.write_text(json.dumps(data, indent=2))
    console.print(f"Results exported to {output_path}")
