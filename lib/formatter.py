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
    
    # Fix the parentUuid chain so claude-code can load the transcript
    # without breaking at missing messages.
    last_uuid = None
    for turn in result.kept_turns:
        # Find the first message in this turn to link to the previous turn
        first_msg_idx = -1
        for i, record in enumerate(turn.lines):
            if isinstance(record, dict) and record.get("type") in ("user", "assistant", "system", "attachment"):
                first_msg_idx = i
                break
                
        if first_msg_idx >= 0 and last_uuid is not None:
            # We ONLY rewrite parentUuid if it already exists, to avoid touching root nodes
            if "parentUuid" in turn.lines[first_msg_idx]:
                turn.lines[first_msg_idx]["parentUuid"] = last_uuid
                
        # Find the last message in this turn to be the parent for the next turn
        for record in reversed(turn.lines):
            if isinstance(record, dict) and "uuid" in record and record.get("type") in ("user", "assistant", "system", "attachment"):
                last_uuid = record["uuid"]
                
                # FIX: If this is the absolute last assistant message in the kept turns,
                # we must clear its stale API usage data. Claude Code uses this usage object
                # to render the top-level token count. If we don't clear it, Claude Code will
                # still show the massive 150k+ token count from before compaction.
                if record.get("type") == "assistant" and "message" in record and turn is result.kept_turns[-1]:
                    if "usage" in record["message"]:
                        record["message"]["usage"] = {
                            "input_tokens": result.scored_kept_tokens,
                            "cache_creation_input_tokens": 0,
                            "cache_read_input_tokens": 0,
                            "output_tokens": 0
                        }
                break

    with open(output_path, "w") as f:
        for turn in result.kept_turns:
            for record in turn.lines:
                f.write(json.dumps(record) + "\n")
    console.print(f"\nWrote compacted JSONL to {output_path}")


def append_archive_jsonl(result: SelectionResult, input_path: Path) -> None:
    """Append fully dropped turns to a persistent archive JSONL file.
    
    This acts as a searchable history database of all discarded context.
    """
    if not result.dropped_turns:
        return
        
    # e.g., <uuid>.jsonl -> <uuid>.archive.jsonl
    archive_path = input_path.with_name(input_path.name.replace(".jsonl", ".archive.jsonl"))
    
    with open(archive_path, "a") as f:
        for st in sorted(result.dropped_turns, key=lambda s: s.turn.index):
            for record in st.turn.lines:
                f.write(json.dumps(record) + "\n")
    console.print(f"Appended {len(result.dropped_turns)} dropped turns to archive database: {archive_path}")


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
