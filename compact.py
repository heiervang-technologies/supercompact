#!/usr/bin/env python3
"""Conversation compaction for Claude Code JSONL histories.

Subcommands:
  compact   — Score and select turns to fit a token budget (default)
  evaluate  — Entity preservation evaluation across methods/budgets
  plot      — Generate Pareto plots from evaluation results

Methods:
  dedup        — Suffix automaton dedup, unique content ratio (no model, instant)
  eitf         — Entity-frequency inverse turn frequency (no model, instant)
  setcover     — Greedy entity coverage with forward-reference weighting (instant)
  embed        — Qwen3-Embedding-0.6B cosine similarity (needs GPU/CPU + torch)
  llama-embed  — Qwen3-Embedding-0.6B via llama.cpp server (HTTP, no torch)
  llama-rerank — Qwen3-Reranker-0.6B via llama.cpp server (HTTP, no torch)

Usage:
    uv run compact.py JSONL_FILE --method eitf [--budget 80000]
    uv run compact.py evaluate JSONL_FILE --method all --budget 100000
    uv run compact.py plot eval_results.json -o pareto.png
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table

from lib.parser import parse_jsonl, extract_text
from lib.tokenizer import turn_tokens, estimate_tokens
from lib.types import ScoredTurn, build_query, random_scores
from lib.selector import select_turns
from lib.formatter import print_stats, write_compacted_jsonl, write_scores_csv
from lib.scorer_base import SCORERS, LOCAL_METHODS, ALL_METHODS, get_scorer

console = Console()


# ---------------------------------------------------------------------------
# Compact subcommand (default)
# ---------------------------------------------------------------------------

def cmd_compact(args: argparse.Namespace) -> int:
    """Score and select turns to fit within a token budget."""
    turns = _parse_file(args.jsonl_file)
    if turns is None:
        return 1

    user_turns = [t for t in turns if t.kind == "user"]
    system_turns = [t for t in turns if t.kind == "system"]
    console.print(f"  {len(turns)} turns total: {len(user_turns)} user, {len(system_turns)} system")

    t_start = time.monotonic()

    # Token estimation
    console.print("Estimating tokens...")
    token_counts: dict[int, int] = {}
    for turn in turns:
        token_counts[turn.index] = turn_tokens(turn)

    total_tokens = sum(token_counts.values())
    console.print(f"  {total_tokens:,} tokens total")

    if total_tokens <= args.budget:
        console.print(f"[green]Already within budget ({total_tokens:,} <= {args.budget:,}), nothing to compact.[/green]")
        return 0

    # Identify long system turns that need scoring
    long_system = [t for t in system_turns if token_counts.get(t.index, 0) > args.short_threshold]
    short_system = [t for t in system_turns if token_counts.get(t.index, 0) <= args.short_threshold]

    console.print(f"  {len(short_system)} short system turns (always kept)")
    console.print(f"  {len(long_system)} long system turns (to be scored)")

    if not long_system:
        console.print("[yellow]No long system turns to score.[/yellow]")
        return 0

    # Score
    if args.dry_run:
        console.print("[yellow]Dry run: using random scores[/yellow]")
        scored = random_scores(long_system, token_counts)
    else:
        scorer = get_scorer(args.method)
        console.print(f"Scoring with {scorer.name}...")
        scored = scorer.score(
            turns, long_system, token_counts,
            min_repeat_len=args.min_repeat_len,
            budget=args.budget,
            short_threshold=args.short_threshold,
            device=args.device,
            batch_size=args.batch_size,
            embed_url=args.embed_url,
            rerank_url=args.rerank_url,
        )

    # Select
    result = select_turns(
        turns=turns,
        scored=scored,
        token_counts=token_counts,
        budget=args.budget,
        short_threshold=args.short_threshold,
    )

    t_elapsed = time.monotonic() - t_start

    # Output
    print_stats(result, verbose=args.verbose)
    console.print(f"\nMethod: {args.method} | Wall time: {t_elapsed:.1f}s")

    if args.output:
        write_compacted_jsonl(result, args.output)

    if args.scores_file:
        kept_indices = {t.index for t in result.kept_turns}
        write_scores_csv(scored, kept_indices, args.scores_file)

    return 0


# ---------------------------------------------------------------------------
# Evaluate subcommand
# ---------------------------------------------------------------------------

def cmd_evaluate(args: argparse.Namespace) -> int:
    """Run entity preservation evaluation across methods and budgets."""
    from lib.eval.entity_coverage import (
        extract_entities, compute_coverage, EntityCoverageResult, ENTITY_TYPES,
    )
    from lib.eval.evidence_coverage import compute_evidence_coverage
    from lib.eval.cache import conv_hash, load_probes

    turns = _parse_file(args.jsonl_file)
    if turns is None:
        return 1

    methods = _resolve_methods(args.method)

    # Split conversation
    split_idx = int(len(turns) * args.split_ratio)
    while split_idx < len(turns) and turns[split_idx].kind != "user":
        split_idx += 1

    prefix_turns = turns[:split_idx]
    suffix_turns = turns[split_idx:]

    if not prefix_turns or not suffix_turns:
        console.print("[red]Split produced empty prefix or suffix.[/red]")
        return 1

    console.print(f"Split: {len(prefix_turns)} prefix / {len(suffix_turns)} suffix turns")

    # Load cached probes (optional — evidence coverage needs them)
    key = conv_hash(args.jsonl_file, args.split_ratio)
    probe_set = load_probes(args.probe_cache, key)
    if probe_set is not None:
        console.print(f"Loaded {len(probe_set.probes)} cached probes")
    else:
        console.print("[dim]No cached probes found — skipping evidence coverage[/dim]")

    # Extract suffix entities once
    suffix_texts = [extract_text(t) for t in suffix_turns if t.kind == "system"]
    suffix_entities = extract_entities("\n".join(suffix_texts))
    console.print(f"Extracted {suffix_entities.total_count} entities from suffix")

    evidence_results = []
    entity_results: list[EntityCoverageResult] = []

    for method in methods:
        console.print(f"\n[bold]{'='*50}[/bold]")
        console.print(f"[bold]Evaluating: {method} (budget={args.budget:,})[/bold]")

        try:
            result, compact_speed_s, kept_tokens, total_prefix_tokens = _compact_prefix(
                method, prefix_turns, args,
            )
        except Exception as e:
            console.print(f"[red]  Compaction error: {e}[/red]")
            import traceback
            traceback.print_exc()
            continue

        console.print(
            f"  Compacted to {kept_tokens:,} tokens "
            f"({len(result.kept_turns)} turns) in {compact_speed_s:.1f}s"
        )

        kept_indices = {t.index for t in result.kept_turns}

        # Evidence turn coverage (only if probes available)
        if probe_set is not None:
            ev_result = compute_evidence_coverage(
                probe_set=probe_set,
                kept_turn_indices=kept_indices,
                method=method,
                budget=args.budget,
            )
            ev_result.speed_s = compact_speed_s
            ev_result.kept_tokens = kept_tokens
            ev_result.total_tokens = total_prefix_tokens
            evidence_results.append(ev_result)

        # Entity coverage
        kept_texts = [extract_text(t) for t in result.kept_turns]
        kept_entities = extract_entities("\n".join(kept_texts))
        cov, wcov, type_breakdown = compute_coverage(suffix_entities, kept_entities)

        entity_results.append(EntityCoverageResult(
            method=method,
            budget=args.budget,
            speed_s=compact_speed_s,
            coverage=cov,
            weighted_coverage=wcov,
            type_coverage=type_breakdown,
            total_tokens=total_prefix_tokens,
            kept_tokens=kept_tokens,
            compression=kept_tokens / total_prefix_tokens if total_prefix_tokens > 0 else 0,
            suffix_entity_count=suffix_entities.total_count,
            prefix_entity_count=kept_entities.total_count,
            covered_count=len(suffix_entities.all_entities() & kept_entities.all_entities()),
        ))

    if not evidence_results and not entity_results:
        console.print("[red]No results.[/red]")
        return 1

    _print_evidence_table(evidence_results)
    _print_entity_table(entity_results)

    if args.verbose and evidence_results:
        _print_dropped_evidence(evidence_results)

    if args.eval_output:
        _export_eval_json(evidence_results, entity_results, args.eval_output)

    return 0


# ---------------------------------------------------------------------------
# Plot subcommand
# ---------------------------------------------------------------------------

def cmd_plot(args: argparse.Namespace) -> int:
    """Generate Pareto plots from evaluation result JSON files."""
    from lib.pareto import plot_entity_coverage, plot_type_breakdown

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Load results from all input files
    results = []
    for f in args.result_files:
        data = json.loads(f.read_text())
        if isinstance(data, list):
            results.extend(data)
        else:
            results.append(data)

    if not results:
        console.print("[red]No results found in input files.[/red]")
        return 1

    # Deduplicate by (method, kept_tokens)
    seen = set()
    unique = []
    for r in results:
        key = (r["method"], r.get("kept_tokens", 0))
        if key not in seen:
            seen.add(key)
            unique.append(r)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    fig.patch.set_facecolor("#0d1117")

    plot_entity_coverage(ax1, unique, show_legend=True)
    plot_type_breakdown(ax2, unique)

    plt.tight_layout()
    output = args.output or Path("pareto_v2.png")
    fig.savefig(output, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    console.print(f"Saved plot to {output}")
    plt.close()

    return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_file(path: Path) -> list | None:
    """Parse JSONL and print basic stats. Returns None on error."""
    if not path.exists():
        console.print(f"[red]Error: {path} not found[/red]")
        return None
    console.print(f"Parsing {path.name}...")
    return parse_jsonl(path)


def _resolve_methods(method_arg: str) -> list[str]:
    """Resolve 'all' to the list of methods, or return a single method."""
    if method_arg == "all":
        return ALL_METHODS
    return [method_arg]


def _compact_prefix(method, prefix_turns, args):
    """Run a compaction method on prefix turns. Returns (result, speed_s, kept_tokens, total_tokens)."""
    import time as _time

    prefix_copy = list(prefix_turns)
    for i, t in enumerate(prefix_copy):
        t.index = i

    token_counts: dict[int, int] = {}
    for t in prefix_copy:
        token_counts[t.index] = turn_tokens(t)

    total_prefix_tokens = sum(token_counts.values())

    # Claude-code LLM summarization is a special path
    if method == "claude-code":
        return _compact_claude_code(prefix_copy, token_counts, total_prefix_tokens, args)

    # Standard score-and-select
    prefix_system = [t for t in prefix_copy if t.kind == "system"]
    prefix_long = [t for t in prefix_system if token_counts.get(t.index, 0) > args.short_threshold]

    scorer = get_scorer(method)
    t_start = _time.monotonic()
    scored = scorer.score(
        prefix_copy, prefix_long, token_counts,
        min_repeat_len=args.min_repeat_len,
        budget=args.budget,
        short_threshold=args.short_threshold,
        device=args.device,
        batch_size=args.batch_size,
        embed_url=args.embed_url,
        rerank_url=args.rerank_url,
    )

    result = select_turns(
        turns=prefix_copy,
        scored=scored,
        token_counts=token_counts,
        budget=args.budget,
        short_threshold=args.short_threshold,
    )

    compact_speed_s = _time.monotonic() - t_start
    kept_tokens = sum(token_counts.get(t.index, 0) for t in result.kept_turns)

    return result, compact_speed_s, kept_tokens, total_prefix_tokens


def _compact_claude_code(prefix_copy, token_counts, total_prefix_tokens, args):
    """Run claude-code LLM summarization and wrap result for entity evaluation."""
    import time as _time
    from lib.llm_compact import llm_compact, make_synthetic_turn
    from lib.eval.entity_coverage import extract_entities, compute_coverage
    from lib.selector import SelectionResult

    console.print("  Calling Claude via OpenRouter for summarization...")
    t_start = _time.monotonic()
    summary = llm_compact(prefix_copy, args.budget)
    compact_speed_s = _time.monotonic() - t_start

    synthetic_turn = make_synthetic_turn(summary)
    kept_tokens = estimate_tokens(summary)
    console.print(f"  Summary: {kept_tokens:,} tokens in {compact_speed_s:.1f}s")

    # Wrap in a SelectionResult for compatibility
    result = SelectionResult(kept_turns=[synthetic_turn])

    return result, compact_speed_s, kept_tokens, total_prefix_tokens


# ---------------------------------------------------------------------------
# Result display helpers
# ---------------------------------------------------------------------------

def _print_evidence_table(evidence_results):
    """Print evidence turn coverage table."""
    if not evidence_results:
        console.print("\n[dim]Evidence coverage: skipped (no probes)[/dim]")
        return

    from lib.eval.probes import DIMENSIONS

    table = Table(title="\nEvidence Turn Coverage", show_header=True, header_style="bold")
    table.add_column("Metric", style="bold")
    for r in evidence_results:
        table.add_column(f"{r.method}\n({r.budget:,} budget)", justify="right")

    for dim_name, dim_weight in DIMENSIONS.items():
        label = f"{dim_name} (w={dim_weight:.2f})"
        values = []
        for r in evidence_results:
            d = r.dimension_map.get(dim_name)
            if d and d.probe_count > 0:
                values.append(f"{d.mean_coverage:.3f}  ({d.probe_count}p)")
            else:
                values.append("—")
        table.add_row(label, *values)

    table.add_row("─" * 24, *["─" * 14 for _ in evidence_results])
    table.add_row("[bold]Composite[/bold]", *[f"[bold]{r.composite:.3f}[/bold]" for r in evidence_results])
    table.add_row("NDCG (difficulty-weighted)", *[f"{r.ndcg:.3f}" for r in evidence_results])
    table.add_row("Speed (s)", *[f"{r.speed_s:.2f}" for r in evidence_results])
    table.add_row("Tokens (kept / total)", *[f"{r.kept_tokens:,} / {r.total_tokens:,}" for r in evidence_results])

    console.print(table)


def _print_entity_table(entity_results):
    """Print entity preservation table."""
    if not entity_results:
        return

    from lib.eval.entity_coverage import ENTITY_TYPES

    table = Table(title="\nEntity Preservation", show_header=True, header_style="bold")
    table.add_column("Metric", style="bold")
    for r in entity_results:
        table.add_column(f"{r.method}\n({r.budget:,} budget)", justify="right")

    all_types = sorted(
        {t for r in entity_results for t in r.type_coverage},
        key=lambda t: ENTITY_TYPES.get(t, 0),
        reverse=True,
    )
    for etype in all_types:
        label = f"{etype} (w={ENTITY_TYPES.get(etype, 0):.1f})"
        values = []
        for r in entity_results:
            tc = r.type_coverage.get(etype)
            if tc:
                values.append(f"{tc['coverage']:.3f}  ({tc['covered']}/{tc['total']})")
            else:
                values.append("—")
        table.add_row(label, *values)

    table.add_row("─" * 24, *["─" * 18 for _ in entity_results])
    table.add_row("[bold]Coverage (unweighted)[/bold]", *[f"{r.coverage:.3f}" for r in entity_results])
    table.add_row("[bold]Coverage (weighted)[/bold]", *[f"[bold]{r.weighted_coverage:.3f}[/bold]" for r in entity_results])
    table.add_row("Suffix entities", *[f"{r.suffix_entity_count}" for r in entity_results])
    table.add_row("Covered entities", *[f"{r.covered_count}" for r in entity_results])

    console.print(table)


def _print_dropped_evidence(evidence_results):
    """Print dropped evidence details."""
    for r in evidence_results:
        dropped = [p for p in r.probe_details if p.coverage < 1.0]
        if dropped:
            console.print(f"\n  [yellow]{r.method}[/yellow] — dropped evidence:")
            for p in dropped:
                console.print(
                    f"    {p.probe_id} ({p.dimension}, {p.difficulty}): "
                    f"coverage={p.coverage:.2f}  "
                    f"kept={p.kept_evidence}  dropped={p.dropped_evidence}"
                )


def _export_eval_json(evidence_results, entity_results, output_path):
    """Export evaluation results as JSON."""
    data = []
    if evidence_results:
        for i, ev in enumerate(evidence_results):
            entry = ev.to_dict()
            if i < len(entity_results):
                entry["entity_coverage"] = entity_results[i].to_dict()
            data.append(entry)
    else:
        for ent in entity_results:
            data.append(ent.to_dict())
    output_path.write_text(json.dumps(data, indent=2))
    console.print(f"\nResults exported to {output_path}")


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared between compact and evaluate."""
    parser.add_argument("jsonl_file", type=Path, help="Path to the JSONL conversation file")
    parser.add_argument("--method", choices=list(SCORERS.keys()) + ["claude-code", "all"],
                        default="eitf", help="Scoring method (default: eitf)")
    parser.add_argument("--budget", type=int, default=80_000, help="Target token budget (default: 80000)")
    parser.add_argument("--short-threshold", type=int, default=300,
                        help="System turns <= this many tokens are always kept (default: 300)")
    parser.add_argument("--device", type=str, default="cpu", help="PyTorch device for embed method")
    parser.add_argument("--batch-size", type=int, default=16, help="Embedding batch size")
    parser.add_argument("--min-repeat-len", type=int, default=64,
                        help="Min repeated substring length for dedup")
    parser.add_argument("--embed-url", type=str, default="http://localhost:8080",
                        help="llama.cpp embedding server URL")
    parser.add_argument("--rerank-url", type=str, default="http://localhost:8181",
                        help="llama.cpp reranker server URL")
    parser.add_argument("--verbose", action="store_true", help="Show detailed breakdown")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        description="Supercompact: conversation compaction for Claude Code JSONL histories.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- compact (default when no subcommand given) ---
    compact_p = subparsers.add_parser("compact", help="Compact a conversation (default)")
    _add_common_args(compact_p)
    compact_p.add_argument("--output", type=Path, help="Write compacted JSONL to this file")
    compact_p.add_argument("--scores-file", type=Path, help="Write scores CSV to this file")
    compact_p.add_argument("--dry-run", action="store_true", help="Use random scores")

    # --- evaluate ---
    eval_p = subparsers.add_parser("evaluate", help="Run entity preservation evaluation")
    _add_common_args(eval_p)
    eval_p.add_argument("--split-ratio", type=float, default=0.70, help="Prefix/suffix split ratio")
    eval_p.add_argument("--probe-cache", type=Path, default=Path("eval_cache"),
                        help="Directory for cached probe sets")
    eval_p.add_argument("--eval-output", type=Path, help="Export results as JSON")

    # --- plot ---
    plot_p = subparsers.add_parser("plot", help="Generate Pareto plots from eval results")
    plot_p.add_argument("result_files", type=Path, nargs="+", help="JSON result files to plot")
    plot_p.add_argument("-o", "--output", type=Path, help="Output image path (default: pareto_v2.png)")

    return parser


def main() -> int:
    # If the first arg looks like a file (not a subcommand), prepend 'compact'
    subcommands = {"compact", "evaluate", "plot"}
    argv = sys.argv[1:]
    if argv and argv[0] not in subcommands and not argv[0].startswith("-"):
        argv = ["compact"] + argv

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "compact":
        return cmd_compact(args)
    elif args.command == "evaluate":
        return cmd_evaluate(args)
    elif args.command == "plot":
        return cmd_plot(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
