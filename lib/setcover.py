"""Greedy Set-Cover scorer for entity preservation.

Instead of scoring each turn independently (like EITF), this method
directly optimizes for entity coverage breadth using greedy set-cover:

1. Extract entities from all turns (reuses extract_entities)
2. Weight entities by proximity to the suffix boundary and rarity
3. Greedily select turns that maximize marginal weighted entity
   coverage, incorporating recency so the selector doesn't distort order

This avoids EITF's redundancy problem: EITF might select 5 turns that
all mention the same file path, wasting budget. Set-cover ensures each
selected turn contributes new entity coverage.

No ML model needed. Sub-second on any hardware.
"""

from __future__ import annotations

import math
from collections import Counter

from .eval.entity_coverage import ENTITY_TYPES, extract_entities
from .parser import Turn, extract_text
from .types import ScoredTurn


def setcover_scores(
    turns: list[Turn],
    system_turns: list[Turn],
    token_counts: dict[int, int],
    budget: int = 80_000,
    short_threshold: int = 300,
) -> list[ScoredTurn]:
    """Score system turns via greedy set-cover for entity coverage.

    Runs a greedy set-cover with recency-aware scoring. Each turn's
    marginal entity value includes a recency term so turns near the
    suffix boundary (which tend to share entities with the suffix)
    are preferred.

    Returns ScoredTurn objects with scores in [0, 1].
    """
    N = len(turns)
    total_turns = N

    # 1. Extract entities from ALL turns
    print("  [setcover] Extracting entities from all turns...", flush=True)
    turn_entity_sets: dict[int, set[tuple[str, str]]] = {}
    entity_turn_count: Counter[tuple[str, str]] = Counter()

    for turn in turns:
        text = extract_text(turn)
        es = extract_entities(text)
        pairs = es.all_entities()
        turn_entity_sets[turn.index] = pairs
        seen: set[tuple[str, str]] = set()
        for pair in pairs:
            if pair not in seen:
                entity_turn_count[pair] += 1
                seen.add(pair)

    total_entities = sum(len(v) for v in turn_entity_sets.values())
    unique_entities = len(entity_turn_count)
    print(f"  [setcover] Entities: {total_entities:,} occurrences, {unique_entities:,} unique", flush=True)

    # 2. Build entity importance weights.
    #    - Type weight from ENTITY_TYPES
    #    - ITF (inverse turn frequency) for rarity
    #    - Boundary proximity: entities appearing in last 30% of turns
    #      get progressively higher weight (proxy for suffix relevance)
    boundary_start = int(N * 0.70)
    entity_max_position: dict[tuple[str, str], int] = {}
    for turn in turns:
        for pair in turn_entity_sets.get(turn.index, set()):
            cur = entity_max_position.get(pair, -1)
            if turn.index > cur:
                entity_max_position[pair] = turn.index

    entity_weight: dict[tuple[str, str], float] = {}
    for pair, count in entity_turn_count.items():
        etype, _ = pair
        type_w = ENTITY_TYPES.get(etype, 0.3)
        itf = math.log(N / count)
        # Proximity bonus: scale by how late the entity appears
        max_pos = entity_max_position.get(pair, 0)
        if max_pos >= boundary_start:
            # Linear ramp from 1x at boundary_start to 4x at end
            frac = (max_pos - boundary_start) / max(N - boundary_start, 1)
            proximity_mult = 1.0 + 3.0 * frac
        else:
            proximity_mult = 1.0
        entity_weight[pair] = type_w * itf * proximity_mult

    # 3. Pre-compute entities from always-kept turns
    always_kept_entities: set[tuple[str, str]] = set()
    long_system_set = {t.index for t in system_turns}
    for turn in turns:
        if turn.index not in long_system_set:
            always_kept_entities |= turn_entity_sets.get(turn.index, set())

    # 4. Greedy set-cover with recency tiebreaker.
    #    For each candidate turn, compute:
    #      value = marginal_entity_weight / sqrt(tokens) + recency_bonus
    #    This mirrors what the selector does, so our ordering won't be
    #    distorted by the selector's 0.15 recency adjustment.
    candidates = {t.index: t for t in system_turns}
    covered = set(always_kept_entities)
    selected_order: list[int] = []

    print(f"  [setcover] Running greedy set-cover over {len(candidates)} candidates...", flush=True)

    while candidates:
        best_idx = -1
        best_score = -1.0

        for idx in candidates:
            turn_entities = turn_entity_sets.get(idx, set())
            new_entities = turn_entities - covered
            if not new_entities:
                continue

            tokens = token_counts.get(idx, 1)
            marginal_weight = sum(entity_weight.get(e, 0.0) for e in new_entities)
            efficiency = marginal_weight / math.sqrt(max(tokens, 1))

            # Recency bonus matching the selector's formula
            recency = idx / total_turns if total_turns > 0 else 0
            score = efficiency + 0.3 * recency  # slightly higher than selector's 0.15

            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx < 0:
            break

        selected_order.append(best_idx)
        covered |= turn_entity_sets.get(best_idx, set())
        del candidates[best_idx]

    print(f"  [setcover] Ordered {len(selected_order)} turns by marginal coverage", flush=True)

    # 5. Assign scores. Selected turns get high scores that ALREADY
    #    include recency, so we need to compensate for the selector's
    #    0.15 * recency bonus to maintain our ordering.
    n_selected = len(selected_order)
    selection_rank: dict[int, int] = {
        idx: rank for rank, idx in enumerate(selected_order)
    }

    results: list[ScoredTurn] = []
    for turn in system_turns:
        tokens = token_counts.get(turn.index, 0)

        if turn.index in selection_rank:
            rank = selection_rank[turn.index]
            # Base score: 1.0 declining to 0.1 over selection order
            if n_selected > 1:
                base_score = 1.0 - 0.9 * (rank / (n_selected - 1))
            else:
                base_score = 1.0
            # Subtract the selector's recency bonus so our ordering
            # is preserved after the selector adds it back
            recency = turn.index / total_turns if total_turns > 0 else 0
            score = max(base_score - 0.15 * recency, 0.01)
        else:
            score = 0.0

        results.append(ScoredTurn(
            turn=turn,
            score=score,
            tokens=tokens,
        ))

    return results
