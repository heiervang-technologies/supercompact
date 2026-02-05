#!/usr/bin/env python3
"""Conversation compaction for Claude Code JSONL histories.

Methods:
  embed        — Qwen3-Embedding-0.6B cosine similarity (needs GPU/CPU + torch)
  dedup        — Suffix automaton dedup, unique content ratio (no model, instant)
  llama-embed  — Qwen3-Embedding-0.6B via llama.cpp server (HTTP, no torch)
  llama-rerank — Qwen3-Reranker-0.6B via llama.cpp server (HTTP, no torch)

Usage:
    uv run compact.py JSONL_FILE --method embed [--budget 80000] [--device cuda:0]
    uv run compact.py JSONL_FILE --method dedup [--budget 80000] [--min-repeat-len 64]
    uv run compact.py JSONL_FILE --method llama-embed --embed-url http://localhost:8080
    uv run compact.py JSONL_FILE --method llama-rerank --rerank-url http://localhost:8181
    uv run compact.py JSONL_FILE --evaluate [--budget 80000]       # fitness benchmark
    uv run compact.py JSONL_FILE --evaluate-llm --method all --budget 3000  # LLM-as-Judge
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table

from lib.parser import parse_jsonl, extract_text
from lib.tokenizer import turn_tokens
from lib.types import ScoredTurn, build_query, random_scores
from lib.selector import select_turns
from lib.formatter import print_stats, write_compacted_jsonl, write_scores_csv

console = Console()


def _run_evaluate(args: argparse.Namespace, turns: list) -> int:
    """Run fitness evaluation: benchmark method(s) on a split conversation."""
    from lib.fitness import evaluate, FitnessResult

    all_methods = ["dedup", "embed", "llama-embed", "llama-rerank"]
    methods = [args.method] if args.method != "all" else all_methods
    results: list[FitnessResult] = []

    for method in methods:
        console.print(f"\n[bold]Evaluating: {method}[/bold]")
        try:
            fr = evaluate(
                turns=turns,
                method=method,
                budget=args.budget,
                split_ratio=args.split_ratio,
                short_threshold=args.short_threshold,
                min_repeat_len=args.min_repeat_len,
                device=args.device,
                batch_size=args.batch_size,
                embed_url=args.embed_url,
                rerank_url=args.rerank_url,
            )
            results.append(fr)
        except Exception as e:
            console.print(f"[red]  Error: {e}[/red]")
            continue

    if not results:
        console.print("[red]No results.[/red]")
        return 1

    # Display results
    table = Table(title="\nFitness Evaluation", show_header=True)
    table.add_column("Metric", style="bold")
    for r in results:
        table.add_column(r.method, justify="right")

    rows = [
        ("Recall", [f"{r.recall:.4f}" for r in results]),
        ("Speed (s)", [f"{r.speed_s:.2f}" for r in results]),
        ("Compression", [f"{r.compression:.3f}" for r in results]),
        ("F1 (recall vs compression)", [f"{r.f1:.4f}" for r in results]),
        ("Budget", [f"{r.budget:,}" for r in results]),
        ("Prefix tokens", [f"{r.total_tokens:,}" for r in results]),
        ("Kept tokens", [f"{r.kept_tokens:,}" for r in results]),
        ("Prefix / suffix turns", [f"{r.prefix_turns} / {r.suffix_turns}" for r in results]),
        ("Suffix vocab size", [f"{r.suffix_vocab_size:,}" for r in results]),
        ("Scored turns", [f"{r.scored_count}" for r in results]),
        ("Kept / dropped scored", [f"{r.kept_scored} / {r.dropped_scored}" for r in results]),
    ]

    for label, values in rows:
        table.add_row(label, *values)

    console.print(table)

    # Highlight winner
    if len(results) == 2:
        a, b = results
        recall_winner = a.method if a.recall >= b.recall else b.method
        speed_winner = a.method if a.speed_s <= b.speed_s else b.method
        f1_winner = a.method if a.f1 >= b.f1 else b.method

        console.print(f"\n  Recall winner:  {recall_winner}")
        console.print(f"  Speed winner:   {speed_winner}")
        console.print(f"  F1 winner:      {f1_winner}")

    return 0


def _run_evaluate_llm(args: argparse.Namespace, turns: list) -> int:
    """Run LLM-as-Judge evaluation: probe generation, answer gen, scoring."""
    from lib.eval.probes import generate_probes, ProbeSet
    from lib.eval.cache import conv_hash, load_probes, save_probes, DEFAULT_CACHE_DIR
    from lib.eval.judge import generate_answers, score_answers, ANSWER_MODELS
    from lib.eval.aggregate import aggregate
    from lib.eval.report import print_results, export_json

    cache_dir = args.probe_cache
    all_methods = ["dedup", "llama-embed", "llama-rerank"]
    methods = [args.method] if args.method != "all" else all_methods

    # --- 1. Split conversation ---
    split_idx = int(len(turns) * args.split_ratio)
    while split_idx < len(turns) and turns[split_idx].kind != "user":
        split_idx += 1

    prefix_turns = turns[:split_idx]
    suffix_turns = turns[split_idx:]

    if not prefix_turns or not suffix_turns:
        console.print("[red]Split produced empty prefix or suffix.[/red]")
        return 1

    console.print(f"Split: {len(prefix_turns)} prefix / {len(suffix_turns)} suffix turns")

    # --- 2. Get or generate probes ---
    key = conv_hash(args.jsonl_file, args.split_ratio)
    probe_set = None

    if not args.regenerate_probes:
        probe_set = load_probes(cache_dir, key)
        if probe_set:
            console.print(f"Loaded {len(probe_set.probes)} cached probes")

    if probe_set is None:
        console.print("Generating probes via Claude API...")
        probe_set = generate_probes(
            prefix_turns=prefix_turns,
            suffix_turns=suffix_turns,
            split_idx=split_idx,
            conv_hash=key,
            split_ratio=args.split_ratio,
            model=args.judge_model,
        )
        path = save_probes(cache_dir, probe_set)
        console.print(f"Generated {len(probe_set.probes)} probes, cached to {path}")

    # --- 3. For each method: compact, generate answers, score ---
    all_results = []

    for method in methods:
        console.print(f"\n[bold]{'='*50}[/bold]")
        console.print(f"[bold]Evaluating: {method} (budget={args.budget:,})[/bold]")

        # Run compaction to get the compacted context
        try:
            # Re-index prefix turns for the fitness evaluator
            prefix_copy = [t for t in prefix_turns]
            for i, t in enumerate(prefix_copy):
                t.index = i

            token_counts: dict[int, int] = {}
            for t in prefix_copy:
                token_counts[t.index] = turn_tokens(t)

            prefix_system = [t for t in prefix_copy if t.kind == "system"]
            prefix_long = [t for t in prefix_system
                           if token_counts.get(t.index, 0) > args.short_threshold]
            prefix_user = [t for t in prefix_copy if t.kind == "user"]

            # Score turns using the compaction method
            scored: list[ScoredTurn]

            if method == "dedup":
                from lib.dedup import dedup_scores
                scored = dedup_scores(prefix_copy, prefix_long, token_counts,
                                      min_repeat_len=args.min_repeat_len)
            elif method == "llama-embed":
                from lib.llama_embed import LlamaEmbedScorer
                scorer = LlamaEmbedScorer(base_url=args.embed_url)
                query = build_query(prefix_user)
                scored = scorer.score_turns(prefix_long, query, token_counts,
                                            batch_size=args.batch_size)
            elif method == "llama-rerank":
                from lib.llama_rerank import LlamaRerankScorer
                scorer = LlamaRerankScorer(base_url=args.rerank_url)
                query = build_query(prefix_user)
                scored = scorer.score_turns(prefix_long, query, token_counts)
            else:
                console.print(f"[red]Unsupported method for LLM eval: {method}[/red]")
                continue

            result = select_turns(
                turns=prefix_copy,
                scored=scored,
                token_counts=token_counts,
                budget=args.budget,
                short_threshold=args.short_threshold,
            )

            # Build compacted context string
            compacted_text = "\n\n".join(extract_text(t) for t in result.kept_turns)
            kept_tokens = sum(token_counts.get(t.index, 0) for t in result.kept_turns)
            console.print(f"  Compacted to {kept_tokens:,} tokens ({len(result.kept_turns)} turns)")

        except Exception as e:
            console.print(f"[red]  Compaction error: {e}[/red]")
            import traceback
            traceback.print_exc()
            continue

        # Generate answers
        console.print("  Generating answers...")
        answers = generate_answers(compacted_text, probe_set)

        # Score answers
        console.print("  Scoring answers...")
        score_answers(answers, probe_set, judge_model=args.judge_model)

        # Aggregate
        method_results = aggregate(answers, probe_set, method, args.budget)
        all_results.extend(method_results)

    if not all_results:
        console.print("[red]No results.[/red]")
        return 1

    # --- 4. Display and export ---
    print_results(all_results)

    if args.eval_output:
        export_json(all_results, args.eval_output)

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compact Claude Code conversation histories."
    )
    parser.add_argument("jsonl_file", type=Path, help="Path to the JSONL conversation file")
    parser.add_argument("--method", choices=["embed", "dedup", "llama-embed", "llama-rerank", "all"], default="embed", help="Scoring method (default: embed, 'all' for evaluate mode)")
    parser.add_argument("--budget", type=int, default=3_000, help="Target token budget (default: 3000)")
    parser.add_argument("--short-threshold", type=int, default=300, help="System turns <= this many tokens are always kept (default: 300)")
    parser.add_argument("--device", type=str, default="cpu", help="PyTorch device for embed method (default: cpu)")
    parser.add_argument("--output", type=Path, default=None, help="Write compacted JSONL to this file")
    parser.add_argument("--scores-file", type=Path, default=None, help="Write scores CSV to this file")
    parser.add_argument("--batch-size", type=int, default=16, help="Embedding batch size (default: 16)")
    parser.add_argument("--min-repeat-len", type=int, default=64, help="Min repeated substring length for dedup (default: 64)")
    parser.add_argument("--embed-url", type=str, default="http://localhost:8080", help="llama.cpp embedding server URL (default: http://localhost:8080)")
    parser.add_argument("--rerank-url", type=str, default="http://localhost:8181", help="llama.cpp reranker server URL (default: http://localhost:8181)")
    parser.add_argument("--evaluate", action="store_true", help="Run fitness evaluation (split conversation, measure recall)")
    parser.add_argument("--evaluate-llm", action="store_true", help="Run LLM-as-Judge evaluation (comprehension probes)")
    parser.add_argument("--split-ratio", type=float, default=0.70, help="Prefix/suffix split for evaluation (default: 0.70)")
    parser.add_argument("--judge-model", type=str, default="anthropic/claude-opus-4-5", help="Model for probe generation and judging (default: anthropic/claude-opus-4-5)")
    parser.add_argument("--probe-cache", type=Path, default=Path("eval_cache"), help="Directory for cached probe sets (default: eval_cache)")
    parser.add_argument("--regenerate-probes", action="store_true", help="Force regeneration of probes even if cached")
    parser.add_argument("--eval-output", type=Path, default=None, help="Export LLM eval results as JSON")
    parser.add_argument("--dry-run", action="store_true", help="Skip model loading, use random scores")
    parser.add_argument("--verbose", action="store_true", help="Show detailed score breakdown")

    args = parser.parse_args()

    if not args.jsonl_file.exists():
        console.print(f"[red]Error: {args.jsonl_file} not found[/red]")
        return 1

    # Parse
    console.print(f"Parsing {args.jsonl_file.name}...")
    turns = parse_jsonl(args.jsonl_file)

    user_turns = [t for t in turns if t.kind == "user"]
    system_turns = [t for t in turns if t.kind == "system"]

    console.print(f"  {len(turns)} turns total: {len(user_turns)} user, {len(system_turns)} system")

    # --- Evaluation modes ---
    if args.evaluate_llm:
        return _run_evaluate_llm(args, turns)
    if args.evaluate:
        return _run_evaluate(args, turns)

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
    long_system = [
        t for t in system_turns
        if token_counts.get(t.index, 0) > args.short_threshold
    ]
    short_system = [
        t for t in system_turns
        if token_counts.get(t.index, 0) <= args.short_threshold
    ]

    console.print(f"  {len(short_system)} short system turns (always kept)")
    console.print(f"  {len(long_system)} long system turns (to be scored)")

    if not long_system:
        console.print("[yellow]No long system turns to score.[/yellow]")
        return 0

    # Score
    scored: list[ScoredTurn]

    if args.dry_run:
        console.print("[yellow]Dry run: using random scores[/yellow]")
        scored = random_scores(long_system, token_counts)

    elif args.method == "dedup":
        console.print(f"Dedup scoring (min_repeat_len={args.min_repeat_len})...")
        from lib.dedup import dedup_scores
        scored = dedup_scores(turns, long_system, token_counts, min_repeat_len=args.min_repeat_len)

    elif args.method == "embed":
        console.print(f"Loading embedding model on {args.device}...")
        from lib.scorer import Scorer
        scorer = Scorer(device=args.device)

        query = build_query(user_turns)
        if args.verbose:
            console.print(f"  Query ({len(query)} chars): {query[:200]}...")

        console.print(f"Scoring {len(long_system)} turns (batch_size={args.batch_size})...")
        scored = scorer.score_turns(long_system, query, token_counts, batch_size=args.batch_size)

    elif args.method == "llama-embed":
        console.print(f"Connecting to llama.cpp embed server at {args.embed_url}...")
        from lib.llama_embed import LlamaEmbedScorer
        scorer = LlamaEmbedScorer(base_url=args.embed_url)

        query = build_query(user_turns)
        if args.verbose:
            console.print(f"  Query ({len(query)} chars): {query[:200]}...")

        console.print(f"Scoring {len(long_system)} turns (batch_size={args.batch_size})...")
        scored = scorer.score_turns(long_system, query, token_counts, batch_size=args.batch_size)

    elif args.method == "llama-rerank":
        console.print(f"Connecting to llama.cpp rerank server at {args.rerank_url}...")
        from lib.llama_rerank import LlamaRerankScorer
        scorer = LlamaRerankScorer(base_url=args.rerank_url)

        query = build_query(user_turns)
        if args.verbose:
            console.print(f"  Query ({len(query)} chars): {query[:200]}...")

        console.print(f"Reranking {len(long_system)} turns...")
        scored = scorer.score_turns(long_system, query, token_counts)

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


if __name__ == "__main__":
    sys.exit(main())
