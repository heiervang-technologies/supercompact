"""Evidence turn coverage: measures what fraction of probe evidence survived compaction.

Uses the existing cached probe set's evidence_turns fields. For each probe,
checks whether the cited evidence turns were preserved in the compacted output.

This is passage-level recall from IR: each probe is a "query" and its
evidence_turns are the "relevant documents." Zero LLM calls needed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .probes import DIMENSIONS, ProbeSet


DIFFICULTY_WEIGHTS = {"easy": 1.0, "medium": 2.0, "hard": 3.0}


@dataclass
class ProbeCoverage:
    """Coverage result for a single probe."""

    probe_id: str
    dimension: str
    difficulty: str
    evidence_turns: list[int]
    kept_evidence: list[int]
    dropped_evidence: list[int]
    coverage: float  # 0-1


@dataclass
class DimensionCoverage:
    """Aggregated coverage for one dimension."""

    dimension: str
    weight: float
    mean_coverage: float  # 0-1
    probe_count: int
    coverages: list[float] = field(default_factory=list)


@dataclass
class EvidenceCoverageResult:
    """Full evidence coverage result, compatible with reporting."""

    method: str
    budget: int
    dimensions: list[DimensionCoverage] = field(default_factory=list)
    composite: float = 0.0       # weighted sum of dimension coverages
    ndcg: float = 0.0            # difficulty-weighted NDCG
    probe_details: list[ProbeCoverage] = field(default_factory=list)
    speed_s: float = 0.0
    kept_tokens: int = 0
    total_tokens: int = 0

    @property
    def dimension_map(self) -> dict[str, DimensionCoverage]:
        return {d.dimension: d for d in self.dimensions}

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "budget": self.budget,
            "composite": self.composite,
            "ndcg": self.ndcg,
            "speed_s": self.speed_s,
            "kept_tokens": self.kept_tokens,
            "total_tokens": self.total_tokens,
            "dimensions": {
                d.dimension: {
                    "coverage": d.mean_coverage,
                    "weight": d.weight,
                    "probe_count": d.probe_count,
                    "coverages": d.coverages,
                }
                for d in self.dimensions
            },
            "probe_details": [
                {
                    "probe_id": p.probe_id,
                    "dimension": p.dimension,
                    "difficulty": p.difficulty,
                    "coverage": p.coverage,
                    "evidence_turns": p.evidence_turns,
                    "kept": p.kept_evidence,
                    "dropped": p.dropped_evidence,
                }
                for p in self.probe_details
            ],
        }


def _dcg(scores_with_weights: list[tuple[float, float]]) -> float:
    """Discounted cumulative gain, sorted by difficulty weight (hardest first)."""
    items = sorted(scores_with_weights, key=lambda x: x[1], reverse=True)
    dcg = 0.0
    for i, (score, weight) in enumerate(items):
        dcg += (score * weight) / math.log2(i + 2)
    return dcg


def compute_evidence_coverage(
    probe_set: ProbeSet,
    kept_turn_indices: set[int],
    method: str,
    budget: int,
) -> EvidenceCoverageResult:
    """Compute evidence turn coverage for a compaction result.

    Args:
        probe_set: Cached probes with evidence_turns fields.
        kept_turn_indices: Set of turn indices that survived compaction.
        method: Compaction method name.
        budget: Token budget used.

    Returns:
        EvidenceCoverageResult with per-probe, per-dimension, and composite scores.
    """
    # Score each probe
    probe_details: list[ProbeCoverage] = []
    by_dim: dict[str, list[ProbeCoverage]] = {}

    for probe in probe_set.probes:
        evidence = probe.evidence_turns
        if not evidence:
            # Probe has no evidence turns annotated — skip
            continue

        kept = [idx for idx in evidence if idx in kept_turn_indices]
        dropped = [idx for idx in evidence if idx not in kept_turn_indices]
        coverage = len(kept) / len(evidence)

        pc = ProbeCoverage(
            probe_id=probe.id,
            dimension=probe.dimension,
            difficulty=probe.difficulty,
            evidence_turns=evidence,
            kept_evidence=kept,
            dropped_evidence=dropped,
            coverage=coverage,
        )
        probe_details.append(pc)
        by_dim.setdefault(probe.dimension, []).append(pc)

    # Aggregate per dimension
    dim_scores: list[DimensionCoverage] = []
    all_scored: list[tuple[float, float]] = []  # (coverage, difficulty_weight) for NDCG

    for dim_name, dim_weight in DIMENSIONS.items():
        probes_in_dim = by_dim.get(dim_name, [])
        if not probes_in_dim:
            dim_scores.append(DimensionCoverage(
                dimension=dim_name,
                weight=dim_weight,
                mean_coverage=0.0,
                probe_count=0,
            ))
            continue

        coverages = [p.coverage for p in probes_in_dim]
        mean_cov = sum(coverages) / len(coverages)

        dim_scores.append(DimensionCoverage(
            dimension=dim_name,
            weight=dim_weight,
            mean_coverage=mean_cov,
            probe_count=len(coverages),
            coverages=coverages,
        ))

        for pc in probes_in_dim:
            dw = DIFFICULTY_WEIGHTS.get(pc.difficulty, 1.0)
            all_scored.append((pc.coverage, dw))

    # Weighted composite
    composite = sum(d.weight * d.mean_coverage for d in dim_scores)

    # NDCG: actual DCG / ideal DCG (ideal = coverage 1.0 for all)
    if all_scored:
        actual_dcg = _dcg(all_scored)
        ideal = [(1.0, w) for _, w in all_scored]
        ideal_dcg = _dcg(ideal)
        ndcg = actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0
    else:
        ndcg = 0.0

    return EvidenceCoverageResult(
        method=method,
        budget=budget,
        dimensions=dim_scores,
        composite=composite,
        ndcg=ndcg,
        probe_details=probe_details,
    )
