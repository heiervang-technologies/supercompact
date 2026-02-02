"""Budget-constrained turn selection.

Three tiers:
1. All user turns — always kept
2. Short system turns (<=threshold tokens) — always kept
3. Long system turns — scored by reranker, greedily selected by adjusted score
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .parser import Turn
from .scorer import ScoredTurn


@dataclass
class SelectionResult:
    """Result of budget-constrained turn selection."""

    kept_turns: list[Turn] = field(default_factory=list)
    dropped_turns: list[ScoredTurn] = field(default_factory=list)
    kept_scored: list[ScoredTurn] = field(default_factory=list)

    # Token accounting
    user_tokens: int = 0
    short_system_tokens: int = 0
    scored_kept_tokens: int = 0
    scored_dropped_tokens: int = 0
    total_input_tokens: int = 0
    budget: int = 0


def select_turns(
    turns: list[Turn],
    scored: list[ScoredTurn],
    token_counts: dict[int, int],
    budget: int = 80_000,
    short_threshold: int = 300,
) -> SelectionResult:
    """Select turns to keep within a token budget.

    Args:
        turns: All turns (user + system) in order.
        scored: ScoredTurn objects for long system turns only.
        token_counts: Map of turn.index -> token count for all turns.
        budget: Target token budget.
        short_threshold: System turns at or below this token count are always kept.
    """
    result = SelectionResult(budget=budget)
    total_turns = len(turns)

    # Separate turns into categories
    user_turns: list[Turn] = []
    short_system: list[Turn] = []
    scored_map: dict[int, ScoredTurn] = {st.turn.index: st for st in scored}

    for turn in turns:
        tc = token_counts.get(turn.index, 0)
        result.total_input_tokens += tc

        if turn.kind == "user":
            user_turns.append(turn)
            result.user_tokens += tc
        elif tc <= short_threshold:
            short_system.append(turn)
            result.short_system_tokens += tc

    # Always keep user turns and short system turns
    used_tokens = result.user_tokens + result.short_system_tokens
    kept_indices: set[int] = set()

    for t in user_turns:
        kept_indices.add(t.index)
    for t in short_system:
        kept_indices.add(t.index)

    # Most recent system turn is always kept
    last_system = None
    for turn in reversed(turns):
        if turn.kind == "system":
            last_system = turn
            break

    if last_system and last_system.index not in kept_indices:
        tc = token_counts.get(last_system.index, 0)
        kept_indices.add(last_system.index)
        used_tokens += tc
        # Track it in scored_kept if it was scored
        if last_system.index in scored_map:
            result.kept_scored.append(scored_map[last_system.index])

    # Apply recency bonus and sort long system turns by adjusted score
    adjusted: list[tuple[float, ScoredTurn]] = []
    for st in scored:
        if st.turn.index in kept_indices:
            continue  # already kept (e.g., last system turn)
        recency = st.turn.index / total_turns if total_turns > 0 else 0
        adj_score = st.score + 0.15 * recency
        adjusted.append((adj_score, st))

    adjusted.sort(key=lambda x: x[0], reverse=True)

    # Greedily select until budget is filled
    remaining = budget - used_tokens

    for adj_score, st in adjusted:
        if st.tokens <= remaining:
            kept_indices.add(st.turn.index)
            result.kept_scored.append(st)
            result.scored_kept_tokens += st.tokens
            remaining -= st.tokens
        else:
            result.dropped_turns.append(st)
            result.scored_dropped_tokens += st.tokens

    # Build final kept_turns in original order
    result.kept_turns = sorted(
        [t for t in turns if t.index in kept_indices],
        key=lambda t: t.index,
    )

    return result
