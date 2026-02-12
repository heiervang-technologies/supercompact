#!/usr/bin/env python3
"""Compact a Codex CLI session using supercompact's EITF scoring.

This is the Python entry point called by the codex-compact shell wrapper.
It adapts supercompact's pipeline for Codex's rollout JSONL format.

Usage (from supercompact repo root):
    uv run python plugins/codex-cli/compact_codex.py SESSION.jsonl [OPTIONS]

Options match compact.py: --method, --budget, --output, --verbose, etc.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure supercompact lib is importable
_this_dir = Path(__file__).resolve().parent
_repo_root = _this_dir.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
if str(_this_dir) not in sys.path:
    sys.path.insert(0, str(_this_dir))

from rich.console import Console

from codex_parser import parse_codex_jsonl, extract_codex_text, find_latest_codex_session
from lib.tokenizer import estimate_tokens
from lib.types import ScoredTurn, build_query
from lib.selector import select_turns, SelectionResult
from lib.formatter import print_stats
from lib.scorer_base import get_scorer, LOCAL_METHODS
import lib.parser as parser_mod

console = Console()

# All modules that import extract_text from lib.parser
_EXTRACT_TEXT_MODULES = [
    'lib.parser', 'lib.formatter', 'lib.types', 'lib.tokenizer',
    'lib.eitf', 'lib.dedup', 'lib.setcover', 'lib.scorer',
    'lib.llama_embed', 'lib.llama_rerank', 'lib.fitness',
    'lib.llm_compact', 'lib.eval.probes', 'lib.eval.entity_coverage',
]

_orig_refs: dict[str, object] = {}


def _patch_extract_text():
    """Monkey-patch extract_text in all modules that import it.

    The supercompact pipeline imports extract_text in many modules via
    'from .parser import extract_text'. Each gets its own local reference,
    so we must patch every module individually.
    """
    import importlib
    for mod_name in _EXTRACT_TEXT_MODULES:
        try:
            mod = importlib.import_module(mod_name)
            if hasattr(mod, 'extract_text'):
                _orig_refs[mod_name] = mod.extract_text
                mod.extract_text = extract_codex_text
        except ImportError:
            pass


def _unpatch_extract_text():
    """Restore all original extract_text references."""
    import importlib
    for mod_name, orig in _orig_refs.items():
        try:
            mod = importlib.import_module(mod_name)
            mod.extract_text = orig
        except ImportError:
            pass
    _orig_refs.clear()


def compact_session(
    session_path: Path,
    method: str = "eitf",
    budget: int = 80_000,
    short_threshold: int = 300,
    verbose: bool = False,
    output: Path | None = None,
) -> int:
    """Run compaction on a Codex session file.

    Returns 0 on success, 1 on error.
    """
    if not session_path.exists():
        console.print(f"[red]Error: {session_path} not found[/red]")
        return 1

    # Parse
    console.print(f"Parsing {session_path.name}...")
    turns = parse_codex_jsonl(session_path)

    user_turns = [t for t in turns if t.kind == "user"]
    system_turns = [t for t in turns if t.kind == "system"]
    console.print(f"  {len(turns)} turns total: {len(user_turns)} user, {len(system_turns)} system")

    if not turns:
        console.print("[yellow]No turns found in session file.[/yellow]")
        return 0

    # Patch extract_text for Codex format
    _patch_extract_text()

    try:
        return _run_compaction(
            turns, user_turns, system_turns,
            method=method,
            budget=budget,
            short_threshold=short_threshold,
            verbose=verbose,
            output=output,
            session_path=session_path,
        )
    finally:
        _unpatch_extract_text()


def _run_compaction(
    turns, user_turns, system_turns,
    method, budget, short_threshold, verbose, output, session_path,
) -> int:
    """Core compaction logic after parsing."""
    # Token estimation
    console.print("Estimating tokens...")
    token_counts: dict[int, int] = {}
    for turn in turns:
        text = extract_codex_text(turn)
        token_counts[turn.index] = estimate_tokens(text) if text.strip() else 0

    total_tokens = sum(token_counts.values())
    console.print(f"  {total_tokens:,} tokens total")

    if total_tokens <= budget:
        console.print(
            f"[green]Already within budget ({total_tokens:,} <= {budget:,}), "
            f"nothing to compact.[/green]"
        )
        return 0

    # Identify long/short system turns
    long_system = [
        t for t in system_turns
        if token_counts.get(t.index, 0) > short_threshold
    ]
    short_system = [
        t for t in system_turns
        if token_counts.get(t.index, 0) <= short_threshold
    ]

    console.print(f"  {len(short_system)} short system turns (always kept)")
    console.print(f"  {len(long_system)} long system turns (to be scored)")

    if not long_system:
        console.print("[yellow]No long system turns to score.[/yellow]")
        return 0

    t_start = time.monotonic()

    # Score
    scorer = get_scorer(method)
    console.print(f"Scoring with {scorer.name}...")
    scored = scorer.score(
        turns, long_system, token_counts,
        min_repeat_len=64,
        budget=budget,
        short_threshold=short_threshold,
    )

    # Select turns within budget
    result = select_turns(
        turns=turns,
        scored=scored,
        token_counts=token_counts,
        budget=budget,
        short_threshold=short_threshold,
    )

    t_elapsed = time.monotonic() - t_start

    # Display stats
    print_stats(result, verbose=verbose)
    console.print(f"\nMethod: {method} | Wall time: {t_elapsed:.1f}s")

    # Write output
    if output:
        _write_codex_jsonl(result, output, session_path)

    return 0


def _write_codex_jsonl(
    result: SelectionResult, output_path: Path, original_path: Path
) -> None:
    """Write compacted turns back to a JSONL file in Codex rollout format.

    Preserves the session_meta header from the original file and writes
    the kept turns in their original Codex format.
    """
    # Read the original session_meta line (always the first line)
    session_meta = None
    with open(original_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if record.get("type") == "session_meta":
                    session_meta = line
                    break
            except json.JSONDecodeError:
                continue

    with open(output_path, "w") as f:
        # Write session_meta header if we found one
        if session_meta:
            f.write(session_meta + "\n")

        # Write kept turns
        for turn in result.kept_turns:
            for record in turn.lines:
                f.write(json.dumps(record) + "\n")

    console.print(f"\nWrote compacted JSONL to {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compact a Codex CLI session with supercompact EITF scoring."
    )
    parser.add_argument(
        "session_file", nargs="?", type=Path, default=None,
        help="Path to Codex rollout JSONL file (default: latest session)"
    )
    parser.add_argument(
        "--method", choices=LOCAL_METHODS, default="eitf",
        help="Scoring method (default: eitf)"
    )
    parser.add_argument(
        "--budget", type=int, default=80_000,
        help="Target token budget (default: 80000)"
    )
    parser.add_argument(
        "--short-threshold", type=int, default=300,
        help="System turns <= this many tokens are always kept (default: 300)"
    )
    parser.add_argument(
        "--output", "-o", type=Path,
        help="Write compacted output to this file"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show detailed breakdown"
    )

    args = parser.parse_args()

    # Find session file
    session_path = args.session_file
    if session_path is None:
        session_path = find_latest_codex_session()
        if session_path is None:
            console.print("[red]Error: No Codex session files found[/red]")
            console.print("Run Codex CLI first, or specify a session file.")
            return 1
        console.print(f"Using latest session: {session_path}")

    return compact_session(
        session_path=session_path,
        method=args.method,
        budget=args.budget,
        short_threshold=args.short_threshold,
        verbose=args.verbose,
        output=args.output,
    )


if __name__ == "__main__":
    sys.exit(main())
