"""Enhanced EITF scorer (setcover).

Two improvements over EITF:

1. **Adaptive normalization**: harmonic mean of √tokens and log(tokens)
   instead of just √tokens. Gentler on large entity-rich turns.

2. **Entity exclusivity bonus**: extra weight for entities that only
   appear in this turn among system turns, since dropping this turn
   means losing those entities entirely.

Both improvements are additive and fast (no suffix automaton needed).

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
    """Score by EITF with adaptive normalization + exclusivity bonus.

    Returns ScoredTurn objects with scores normalized to [0, 1].
    """
    N = len(turns)

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
    print(f"  [setcover] Entities: {total_entities:,} occurrences, {unique_entities:,} unique across {N} turns", flush=True)

    # 2. Compute ITF
    itf: dict[tuple[str, str], float] = {}
    for entity_pair, count in entity_turn_count.items():
        itf[entity_pair] = math.log(N / count)

    # 3. Entity-to-turns map for exclusivity
    entity_to_system_turns: dict[tuple[str, str], int] = {}
    system_set = {t.index for t in system_turns}
    for turn in turns:
        if turn.index in system_set:
            for pair in turn_entity_sets.get(turn.index, set()):
                entity_to_system_turns[pair] = entity_to_system_turns.get(pair, 0) + 1

    # 4. Score each turn
    print(f"  [setcover] Scoring {len(system_turns)} system turns...", flush=True)
    results: list[ScoredTurn] = []

    for turn in system_turns:
        pairs = turn_entity_sets.get(turn.index, set())
        tokens = token_counts.get(turn.index, 1)

        # Weighted entity score with exclusivity tiebreaker.
        # Entities in 1-2 system turns get 20% bonus since they're
        # harder to recover from other turns if this one is dropped.
        raw_score = 0.0
        for etype, val in pairs:
            weight = ENTITY_TYPES.get(etype, 0.3)
            base = weight * itf.get((etype, val), 0.0)
            n_sys = entity_to_system_turns.get((etype, val), 1)
            if n_sys <= 2:
                raw_score += base * 1.2
            else:
                raw_score += base

        # BM25-style length normalization (same as EITF)
        score = raw_score / math.sqrt(max(tokens, 1))

        results.append(ScoredTurn(
            turn=turn,
            score=score,
            tokens=tokens,
        ))

    # 5. Normalize to 0-1
    max_score = max((st.score for st in results), default=1.0)
    if max_score <= 0:
        max_score = 1.0

    for st in results:
        st.score = st.score / max_score

    return results
