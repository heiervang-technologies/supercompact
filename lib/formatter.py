"""Output formatting and stats display."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from .parser import Turn, extract_text
from .types import ScoredTurn
from .selector import SelectionResult


console = Console()


def print_stats(result: SelectionResult, verbose: bool = False) -> None:
    """Print selection statistics to the console."""
    total_kept = (
        result.user_tokens + result.short_system_tokens + result.scored_kept_tokens
    )

    # Summary table
    table = Table(title="Turn Budget Allocation", show_header=True)
    table.add_column("Category", style="bold")
    table.add_column("Turns", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Status", style="dim")

    user_count = sum(1 for t in result.kept_turns if t.kind == "user")
    short_count = sum(
        1
        for t in result.kept_turns
        if t.kind == "system" and t not in [s.turn for s in result.kept_scored]
    )

    table.add_row(
        "User turns",
        str(user_count),
        f"{result.user_tokens:,}",
        "always kept",
    )
    table.add_row(
        "Short system turns",
        str(short_count),
        f"{result.short_system_tokens:,}",
        "always kept",
    )
    table.add_row(
        "Scored system (kept)",
        str(len(result.kept_scored)),
        f"{result.scored_kept_tokens:,}",
        "selected by score",
    )
    table.add_row(
        "Scored system (dropped)",
        str(len(result.dropped_turns)),
        f"{result.scored_dropped_tokens:,}",
        "below cutoff",
    )
    table.add_section()
    table.add_row(
        "Total kept",
        str(len(result.kept_turns)),
        f"{total_kept:,}",
        f"budget: {result.budget:,}",
    )
    table.add_row(
        "Total input",
        "",
        f"{result.total_input_tokens:,}",
        "",
    )

    console.print()
    console.print(table)

    # Compression ratio
    if result.total_input_tokens > 0:
        ratio = total_kept / result.total_input_tokens
        reduction = 1 - ratio
        console.print(
            f"\nCompression: {result.total_input_tokens:,} -> {total_kept:,} tokens "
            f"({ratio:.1%} kept, {reduction:.1%} reduction)"
        )

    if verbose and result.kept_scored:
        _print_score_details(result)


def _print_score_details(result: SelectionResult) -> None:
    """Print detailed score information for kept and dropped turns."""
    # Kept scored turns
    table = Table(title="\nKept Scored Turns (by score)", show_header=True)
    table.add_column("Index", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Preview")

    for st in sorted(result.kept_scored, key=lambda s: s.score, reverse=True):
        preview = extract_text(st.turn)[:80].replace("\n", " ")
        table.add_row(
            str(st.turn.index),
            f"{st.score:.3f}",
            f"{st.tokens:,}",
            preview,
        )

    console.print(table)

    # Dropped turns
    if result.dropped_turns:
        table = Table(title="\nDropped Turns (by score)", show_header=True)
        table.add_column("Index", justify="right")
        table.add_column("Score", justify="right")
        table.add_column("Tokens", justify="right")
        table.add_column("Preview")

        for st in sorted(result.dropped_turns, key=lambda s: s.score, reverse=True)[:20]:
            preview = extract_text(st.turn)[:80].replace("\n", " ")
            table.add_row(
                str(st.turn.index),
                f"{st.score:.3f}",
                f"{st.tokens:,}",
                preview,
            )

        if len(result.dropped_turns) > 20:
            table.add_row("...", "", "", f"({len(result.dropped_turns) - 20} more)")

        console.print(table)


def write_summary_text(result: SelectionResult, output_path: Path) -> None:
    """Write kept turns as formatted text suitable for Claude's compaction summary.

    Produces a narrative-style summary that preserves the conversation flow,
    including user requests, assistant actions, tool calls, and key outputs.
    """
    parts: list[str] = []

    for turn in result.kept_turns:
        role = "User" if turn.kind == "user" else "Assistant"
        text = extract_text(turn).strip()
        if not text:
            continue

        # Truncate very long turns but keep enough for comprehension
        if len(text) > 4000:
            text = text[:4000] + "\n[... truncated]"

        parts.append(f"[{role} (turn {turn.index})]:\n{text}")

    summary = "\n\n---\n\n".join(parts)
    output_path.write_text(summary)
    console.print(f"\nWrote summary text to {output_path}")


def write_compacted_jsonl(result: SelectionResult, output_path: Path) -> None:
    """Write kept turns back to a JSONL file."""
    with open(output_path, "w") as f:
        for turn in result.kept_turns:
            for record in turn.lines:
                f.write(json.dumps(record) + "\n")
    console.print(f"\nWrote compacted JSONL to {output_path}")


def write_scores_csv(
    scored: list[ScoredTurn],
    kept_indices: set[int],
    output_path: Path,
) -> None:
    """Write scores to a CSV file for analysis."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["turn_index", "score", "tokens", "kept", "preview"])
        for st in sorted(scored, key=lambda s: s.turn.index):
            preview = extract_text(st.turn)[:120].replace("\n", " ")
            writer.writerow([
                st.turn.index,
                f"{st.score:.4f}",
                st.tokens,
                st.turn.index in kept_indices,
                preview,
            ])
    console.print(f"Wrote scores CSV to {output_path}")
