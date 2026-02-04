"""Shared types and helpers for the compaction pipeline."""

from __future__ import annotations

import random
from dataclasses import dataclass

from .parser import Turn, extract_text


@dataclass
class ScoredTurn:
    """A turn with its relevance score."""

    turn: Turn
    score: float
    tokens: int


def build_query(user_turns: list[Turn], max_chars: int = 4000) -> str:
    """Build a query from the last 2-3 user messages."""
    recent = user_turns[-3:] if len(user_turns) >= 3 else user_turns
    parts = [extract_text(t) for t in recent]
    query = "\n---\n".join(parts)
    if len(query) > max_chars:
        query = query[-max_chars:]
    return query


def random_scores(
    system_turns: list[Turn],
    token_counts: dict[int, int],
) -> list[ScoredTurn]:
    """Generate random scores for dry-run testing."""
    return [
        ScoredTurn(
            turn=turn,
            score=random.random(),
            tokens=token_counts.get(turn.index, 0),
        )
        for turn in system_turns
    ]
