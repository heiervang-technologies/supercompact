"""JSONL parsing for Codex CLI rollout/session files.

Codex rollout format (one JSON object per line):
  {"timestamp": "...", "type": "<variant>", "payload": {...}}

All record data is nested under the "payload" key.

Variants:
  session_meta    — Session metadata (first line, always skipped)
  response_item   — Model response items (messages, function_call, function_call_output, reasoning)
  compacted       — Compaction summary (replaces history)
  turn_context    — Turn metadata (cwd, model, policies, user_instructions)
  event_msg       — UI events (skipped)

This module converts Codex rollout entries into the Turn format expected by
supercompact's scoring/selection pipeline.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

# Lazy-resolve Turn class for compatibility with supercompact
_Turn = None


def _get_turn_class():
    """Lazily resolve the Turn class to avoid import issues."""
    global _Turn
    if _Turn is not None:
        return _Turn

    try:
        from lib.parser import Turn
        _Turn = Turn
        return _Turn
    except ImportError:
        pass

    @dataclass
    class Turn:
        kind: str  # "user" or "system"
        lines: list[dict] = field(default_factory=list)
        index: int = 0

        def append(self, record: dict) -> None:
            self.lines.append(record)

    _Turn = Turn
    return _Turn


# Codex record types to skip (non-conversation)
SKIP_TYPES = frozenset({
    "session_meta",
    "event_msg",
})


def _get_payload(record: dict) -> dict:
    """Extract the payload from a Codex rollout record.

    All real Codex records nest data under "payload". Falls back to
    the record itself for forward compatibility.
    """
    return record.get("payload", record)


def _is_user_turn_context(record: dict) -> bool:
    """Check if a turn_context record represents a user turn boundary."""
    return record.get("type") == "turn_context"


def _is_user_message(record: dict) -> bool:
    """Check if a response_item is a user message.

    Codex response_items with payload.role == 'user' are user messages.
    Developer messages (role='developer') are NOT user messages.
    """
    if record.get("type") != "response_item":
        return False
    payload = _get_payload(record)
    return payload.get("role") == "user"


def _is_compacted(record: dict) -> bool:
    """Check if this is a compaction summary record."""
    return record.get("type") == "compacted"


def parse_codex_jsonl(path: Path) -> list:
    """Parse a Codex rollout JSONL file into alternating user/system turns.

    Returns a list of Turn objects compatible with supercompact's pipeline.
    User turns contain turn_context + user message records.
    System turns contain response_item records (assistant messages, function calls, etc.).
    """
    Turn = _get_turn_class()
    turns: list = []
    current_system = None
    turn_index = 0

    with open(path, "r") as f:
        for line_str in f:
            line_str = line_str.strip()
            if not line_str:
                continue
            try:
                record = json.loads(line_str)
            except json.JSONDecodeError:
                continue

            record_type = record.get("type", "")

            # Skip non-conversation records
            if record_type in SKIP_TYPES:
                continue

            # Compacted entries are treated as system turns (summaries)
            if _is_compacted(record):
                if current_system and current_system.lines:
                    turns.append(current_system)
                    current_system = None

                system_turn = Turn(kind="system", index=turn_index)
                system_turn.append(record)
                turns.append(system_turn)
                turn_index += 1
                continue

            # turn_context marks a new user turn
            if _is_user_turn_context(record):
                if current_system and current_system.lines:
                    turns.append(current_system)
                    current_system = None

                user_turn = Turn(kind="user", index=turn_index)
                user_turn.append(record)
                turns.append(user_turn)
                turn_index += 1

                current_system = Turn(kind="system", index=turn_index)
                turn_index += 1
                continue

            # User messages within response_items
            if _is_user_message(record):
                if current_system and current_system.lines:
                    turns.append(current_system)
                    current_system = None

                if turns and turns[-1].kind == "user":
                    turns[-1].append(record)
                else:
                    user_turn = Turn(kind="user", index=turn_index)
                    user_turn.append(record)
                    turns.append(user_turn)
                    turn_index += 1

                current_system = Turn(kind="system", index=turn_index)
                turn_index += 1
                continue

            # All other response_items are system turn content
            if record_type == "response_item":
                if current_system is None:
                    current_system = Turn(kind="system", index=turn_index)
                    turn_index += 1
                current_system.append(record)
                continue

    # Flush trailing system turn
    if current_system and current_system.lines:
        turns.append(current_system)

    # Re-index turns sequentially
    for i, turn in enumerate(turns):
        turn.index = i

    return turns


def extract_codex_text(turn) -> str:
    """Extract human-readable text from a Codex turn for scoring/display.

    Handles Codex's payload-wrapped format:
    - Messages: payload.content[].text
    - Function calls: payload.name + payload.arguments
    - Function outputs: payload.output
    - Reasoning: payload.content[].text (reasoning_text type)
    - Turn context: payload.user_instructions
    - Compacted: payload.message
    """
    parts: list[str] = []

    for record in turn.lines:
        rtype = record.get("type", "")
        payload = _get_payload(record)

        # Compacted summary
        if rtype == "compacted":
            msg = payload.get("message", "")
            if msg:
                parts.append(msg)
            continue

        # Turn context (user turn metadata)
        if rtype == "turn_context":
            user_inst = payload.get("user_instructions", "")
            if user_inst:
                parts.append(user_inst)
            continue

        # Response items — the main content
        if rtype == "response_item":
            payload_type = payload.get("type", "")

            # Message content (user, assistant, developer)
            if payload_type == "message":
                content = payload.get("content", [])
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            btype = block.get("type", "")
                            if btype in ("text", "output_text", "input_text"):
                                text = block.get("text", "")
                                if text:
                                    parts.append(text)
                            elif btype == "refusal":
                                refusal = block.get("refusal", "")
                                if refusal:
                                    parts.append(refusal)
                        elif isinstance(block, str):
                            parts.append(block)

            # Function calls
            elif payload_type == "function_call":
                name = payload.get("name", "")
                arguments = payload.get("arguments", "")
                parts.append(f"[function_call: {name}]")
                if isinstance(arguments, str) and arguments:
                    if len(arguments) > 500:
                        arguments = arguments[:500] + "..."
                    parts.append(arguments)

            # Function call output
            elif payload_type == "function_call_output":
                output = payload.get("output", "")
                if isinstance(output, str):
                    if len(output) > 1000:
                        output = output[:1000] + "..."
                    parts.append(output)

            # Reasoning (chain-of-thought)
            elif payload_type == "reasoning":
                content = payload.get("content", [])
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            btype = block.get("type", "")
                            if btype == "reasoning_text":
                                text = block.get("text", "")
                                if text:
                                    parts.append(text)
                # Also check summary
                summary = payload.get("summary", [])
                if isinstance(summary, list):
                    for item in summary:
                        if isinstance(item, dict):
                            text = item.get("text", "")
                            if text:
                                parts.append(text)

            # Direct text field (fallback)
            else:
                text = payload.get("text", "")
                if text and text not in parts:
                    parts.append(text)

    return "\n".join(parts)


def _codex_home() -> Path:
    """Return the Codex home directory, respecting CODEX_HOME env var."""
    return Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex")))


def find_latest_codex_session() -> Path | None:
    """Find the most recently modified Codex session JSONL file.

    Codex sessions are stored at:
      $CODEX_HOME/sessions/YYYY/MM/DD/rollout-*.jsonl
    (defaults to ~/.codex/sessions/)
    """
    sessions_dir = _codex_home() / "sessions"
    if not sessions_dir.exists():
        return None

    jsonl_files = sorted(
        sessions_dir.rglob("rollout-*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    return jsonl_files[0] if jsonl_files else None
