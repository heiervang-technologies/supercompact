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
            "speed_s": r.speed_s,
            "kept_tokens": r.kept_tokens,
            "total_tokens": r.total_tokens,
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


def export_trace(
    method: str,
    budget: int,
    probe_set,
    answers: list,
    trace_dir: Path,
) -> Path:
    """Export full trace: probes, prompts, answers, scores, judge reasoning.

    Writes a JSON file per method/budget with every detail needed to audit.
    """
    from .judge import _ANSWER_SYSTEM, _JUDGE_SYSTEM

    trace_dir.mkdir(parents=True, exist_ok=True)
    out_path = trace_dir / f"trace_{method}_{budget}.json"

    probe_map = {p.id: p for p in probe_set.probes}
    entries = []

    for a in answers:
        probe = probe_map.get(a.probe_id)
        if not probe:
            continue

        entries.append({
            "probe_id": a.probe_id,
            "dimension": probe.dimension,
            "tier": probe.tier,
            "difficulty": probe.difficulty,
            "model_key": a.model_key,
            "model_label": a.model_label,
            "question": probe.question,
            "gold_answer": probe.gold_answer,
            "evidence_turns": probe.evidence_turns,
            "answer_system_prompt": _ANSWER_SYSTEM,
            "answer_user_prompt_template": "<context>\\n{compacted_context}\\n</context>\\n\\nQuestion: {question}",
            "generated_answer": a.answer,
            "judge_system_prompt": _JUDGE_SYSTEM,
            "judge_user_prompt": (
                f"Question: {probe.question}\n\n"
                f"Gold answer: {probe.gold_answer}\n\n"
                f"Candidate answer: {a.answer}"
            ),
            "score": a.score,
            "judge_reasoning": a.judge_reasoning,
        })

    data = {
        "method": method,
        "budget": budget,
        "probe_count": len(probe_set.probes),
        "answer_count": len(entries),
        "entries": entries,
    }

    out_path.write_text(json.dumps(data, indent=2))
    console.print(f"Trace exported to {out_path}")
    return out_path
