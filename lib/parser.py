"""JSONL parsing and turn grouping for Claude Code conversation histories."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


SKIP_TYPES = frozenset({
    "progress",
    "file-history-snapshot",
    "system",
    "summary",
    "queue-operation",
})


@dataclass
class Turn:
    """A logical turn: either a user message or a system response block."""

    kind: str  # "user" or "system"
    lines: list[dict] = field(default_factory=list)
    index: int = 0  # position in the turn sequence

    def append(self, record: dict) -> None:
        self.lines.append(record)


def _is_user_message(record: dict) -> bool:
    """True if this JSONL record is a genuine user message (not a tool result)."""
    if record.get("type") != "user":
        return False
    # Tool results injected by the system have sourceToolAssistantUUID
    if record.get("sourceToolAssistantUUID"):
        return False
    msg = record.get("message", {})
    content = msg.get("content")
    # String content is always a user message
    if isinstance(content, str):
        return True
    # List content: user messages have text blocks, not tool_result blocks
    if isinstance(content, list):
        types = {
            block.get("type") for block in content if isinstance(block, dict)
        }
        return "tool_result" not in types
    return False


def parse_jsonl(path: Path) -> list[Turn]:
    """Parse a JSONL file into alternating user/system turns.

    Returns a list of Turn objects. User turns contain the original user
    message lines. System turns contain everything between two user messages
    (assistant responses, thinking blocks, tool calls, tool results).
    """
    turns: list[Turn] = []
    current_system: Turn | None = None
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

            if _is_user_message(record):
                # Flush any accumulated system turn
                if current_system and current_system.lines:
                    turns.append(current_system)
                    current_system = None

                # Create a user turn
                user_turn = Turn(kind="user", index=turn_index)
                user_turn.append(record)
                turns.append(user_turn)
                turn_index += 1

                # Start a new system turn accumulator
                current_system = Turn(kind="system", index=turn_index)
                turn_index += 1
            else:
                # Accumulate into the current system turn
                if current_system is None:
                    current_system = Turn(kind="system", index=turn_index)
                    turn_index += 1
                current_system.append(record)

    # Flush trailing system turn
    if current_system and current_system.lines:
        turns.append(current_system)

    # Re-index turns sequentially
    for i, turn in enumerate(turns):
        turn.index = i

    return turns


def extract_text(turn: Turn) -> str:
    """Extract human-readable text from a turn for scoring/display.

    Concatenates message content strings, thinking text, tool_use names/inputs,
    and tool_result content into a single string.
    """
    parts: list[str] = []

    for record in turn.lines:
        msg = record.get("message", {})
        content = msg.get("content")

        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type", "")
                if btype == "text":
                    parts.append(block.get("text", ""))
                elif btype == "thinking":
                    parts.append(block.get("thinking", ""))
                elif btype == "tool_use":
                    name = block.get("name", "")
                    inp = block.get("input", {})
                    parts.append(f"[tool_use: {name}]")
                    if isinstance(inp, dict):
                        for k, v in inp.items():
                            v_str = str(v)
                            if len(v_str) > 500:
                                v_str = v_str[:500] + "..."
                            parts.append(f"  {k}: {v_str}")
                    elif isinstance(inp, str):
                        parts.append(inp[:1000])
                elif btype == "tool_result":
                    result_content = block.get("content", "")
                    if isinstance(result_content, str):
                        parts.append(result_content)
                    elif isinstance(result_content, list):
                        for sub in result_content:
                            if isinstance(sub, dict) and sub.get("type") == "text":
                                parts.append(sub.get("text", ""))

    return "\n".join(parts)
