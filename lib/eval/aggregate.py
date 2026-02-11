"""Score aggregation: per-dimension means, weighted composite, NDCG."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .probes import DIMENSIONS, Probe, ProbeSet
from .judge import ProbeAnswer


DIFFICULTY_WEIGHTS = {"easy": 1.0, "medium": 2.0, "hard": 3.0}


@dataclass
class DimensionScore:
    dimension: str
    weight: float
    mean_score: float       # 0-1 (raw 0-3 normalized to 0-1)
    probe_count: int
    raw_scores: list[int] = field(default_factory=list)


@dataclass
class AggregateResult:
    method: str
    budget: int
    model_key: str
    model_label: str
    dimensions: list[DimensionScore] = field(default_factory=list)
    composite: float = 0.0     # weighted sum of dimension scores, 0-1
    ndcg: float = 0.0          # difficulty-weighted NDCG variant
    speed_s: float = 0.0       # compaction wall time in seconds
    kept_tokens: int = 0       # tokens in compacted output
    total_tokens: int = 0      # tokens in full prefix

    @property
    def dimension_map(self) -> dict[str, DimensionScore]:
        return {d.dimension: d for d in self.dimensions}


def _dcg(scores_with_weights: list[tuple[int, float]]) -> float:
    """Discounted cumulative gain. Items sorted by difficulty weight (hardest first)."""
    # Sort by weight descending so harder probes come first
    items = sorted(scores_with_weights, key=lambda x: x[1], reverse=True)
    dcg = 0.0
    for i, (score, weight) in enumerate(items):
        # Numerator: score * difficulty weight
        # Denominator: log2(position + 2) for 0-indexed
        dcg += (score * weight) / math.log2(i + 2)
    return dcg


def aggregate(
    answers: list[ProbeAnswer],
    probe_set: ProbeSet,
    method: str,
    budget: int,
) -> list[AggregateResult]:
    """Aggregate scores into per-dimension and composite metrics.

    Returns one AggregateResult per model_key found in answers.
    """
    probe_map = {p.id: p for p in probe_set.probes}

    # Group answers by model_key
    by_model: dict[str, list[ProbeAnswer]] = {}
    for a in answers:
        by_model.setdefault(a.model_key, []).append(a)

    results = []
    for model_key, model_answers in by_model.items():
        label = model_answers[0].model_label if model_answers else model_key

        # Group by dimension
        by_dim: dict[str, list[tuple[ProbeAnswer, Probe]]] = {}
        for a in model_answers:
            probe = probe_map.get(a.probe_id)
            if probe:
                by_dim.setdefault(probe.dimension, []).append((a, probe))

        dim_scores = []
        all_scored: list[tuple[int, float]] = []  # (score, difficulty_weight) for NDCG

        for dim_name, dim_weight in DIMENSIONS.items():
            items = by_dim.get(dim_name, [])
            if not items:
                dim_scores.append(DimensionScore(
                    dimension=dim_name,
                    weight=dim_weight,
                    mean_score=0.0,
                    probe_count=0,
                ))
                continue

            raw = [a.score for a, _ in items]
            mean_norm = (sum(raw) / len(raw)) / 3.0  # normalize 0-3 to 0-1

            dim_scores.append(DimensionScore(
                dimension=dim_name,
                weight=dim_weight,
                mean_score=mean_norm,
                probe_count=len(raw),
                raw_scores=raw,
            ))

            for a, probe in items:
                dw = DIFFICULTY_WEIGHTS.get(probe.difficulty, 1.0)
                all_scored.append((a.score, dw))

        # Weighted composite
        composite = sum(d.weight * d.mean_score for d in dim_scores)

        # NDCG: actual DCG / ideal DCG (where ideal = max score 3 for all)
        if all_scored:
            actual_dcg = _dcg(all_scored)
            ideal = [(3, w) for _, w in all_scored]
            ideal_dcg = _dcg(ideal)
            ndcg = actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0
        else:
            ndcg = 0.0

        results.append(AggregateResult(
            method=method,
            budget=budget,
            model_key=model_key,
            model_label=label,
            dimensions=dim_scores,
            composite=composite,
            ndcg=ndcg,
        ))

    return results
