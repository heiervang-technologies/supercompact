"""Entity-frequency Inverse Turn Frequency (EITF) scorer.

Adapts TF-IDF to entity space: extracts structured entities from each turn
using the same regex patterns as the evaluation metric, then scores turns by
weighted entity importance × rarity (inverse turn frequency).

Turns with many rare, high-weight entities score highest — they contain
irreplaceable information that would be lost if dropped.

No ML model needed. Sub-second on any hardware.
"""

from __future__ import annotations

import math
from collections import Counter

from .eval.entity_coverage import ENTITY_TYPES, extract_entities
from .parser import Turn, extract_text
from .types import ScoredTurn


def eitf_scores(
    turns: list[Turn],
    system_turns: list[Turn],
    token_counts: dict[int, int],
) -> list[ScoredTurn]:
    """Score system turns by Entity-frequency Inverse Turn Frequency.

    For each turn:
      score = Σ (entity_weight × ITF(entity)) / √tokens
      ITF(e) = log(N / turns_containing(e))

    Recency is handled by the selector's 0.15 recency bonus.

    Returns ScoredTurn objects with scores normalized to [0, 1].
    """
    N = len(turns)

    # 1. Extract entities from ALL turns
    print("  Extracting entities from all turns...", flush=True)
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
    print(f"  Entities: {total_entities:,} occurrences, {unique_entities:,} unique across {N} turns", flush=True)

    # 2. Compute ITF for each entity
    itf: dict[tuple[str, str], float] = {}
    for entity_pair, count in entity_turn_count.items():
        itf[entity_pair] = math.log(N / count)

    # 3. Score each long system turn
    print(f"  Scoring {len(system_turns)} system turns...", flush=True)
    results: list[ScoredTurn] = []

    for turn in system_turns:
        pairs = turn_entity_sets.get(turn.index, set())
        tokens = token_counts.get(turn.index, 1)

        score = 0.0
        for etype, val in pairs:
            weight = ENTITY_TYPES.get(etype, 0.3)
            score += weight * itf.get((etype, val), 0.0)

        # BM25-style length normalization
        score /= math.sqrt(max(tokens, 1))

        results.append(ScoredTurn(
            turn=turn,
            score=score,
            tokens=tokens,
        ))

    # 4. Normalize to 0-1
    max_score = max((st.score for st in results), default=1.0)
    if max_score <= 0:
        max_score = 1.0

    for st in results:
        st.score = st.score / max_score

    return results
