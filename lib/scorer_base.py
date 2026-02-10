"""Scorer protocol and registry for compaction methods.

Each scorer takes conversation turns and returns scored turns with relevance
scores in [0, 1]. The registry maps method names to factory functions that
create scorers from CLI args.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .parser import Turn
from .types import ScoredTurn


@runtime_checkable
class Scorer(Protocol):
    """Protocol for all compaction scorers."""

    name: str

    def score(
        self,
        turns: list[Turn],
        system_turns: list[Turn],
        token_counts: dict[int, int],
        **kwargs,
    ) -> list[ScoredTurn]: ...


# ---------------------------------------------------------------------------
# Concrete scorer wrappers
# ---------------------------------------------------------------------------

class DedupScorer:
    name = "dedup"

    def score(self, turns, system_turns, token_counts, **kwargs):
        from .dedup import dedup_scores
        return dedup_scores(
            turns, system_turns, token_counts,
            min_repeat_len=kwargs.get("min_repeat_len", 64),
        )


class EitfScorer:
    name = "eitf"

    def score(self, turns, system_turns, token_counts, **kwargs):
        from .eitf import eitf_scores
        return eitf_scores(turns, system_turns, token_counts)


class SetcoverScorer:
    name = "setcover"

    def score(self, turns, system_turns, token_counts, **kwargs):
        from .setcover import setcover_scores
        return setcover_scores(
            turns, system_turns, token_counts,
            budget=kwargs.get("budget", 80_000),
            short_threshold=kwargs.get("short_threshold", 300),
        )


class EmbedScorer:
    name = "embed"

    def score(self, turns, system_turns, token_counts, **kwargs):
        from .scorer import Scorer as PyTorchScorer
        from .types import build_query

        device = kwargs.get("device", "cpu")
        batch_size = kwargs.get("batch_size", 16)
        user_turns = [t for t in turns if t.kind == "user"]

        scorer = PyTorchScorer(device=device)
        query = build_query(user_turns)
        return scorer.score_turns(system_turns, query, token_counts, batch_size=batch_size)


class LlamaEmbedScorerWrapper:
    name = "llama-embed"

    def score(self, turns, system_turns, token_counts, **kwargs):
        from .llama_embed import LlamaEmbedScorer
        from .types import build_query

        embed_url = kwargs.get("embed_url", "http://localhost:8080")
        batch_size = kwargs.get("batch_size", 32)
        user_turns = [t for t in turns if t.kind == "user"]

        scorer = LlamaEmbedScorer(base_url=embed_url)
        query = build_query(user_turns)
        return scorer.score_turns(system_turns, query, token_counts, batch_size=batch_size)


class LlamaRerankScorerWrapper:
    name = "llama-rerank"

    def score(self, turns, system_turns, token_counts, **kwargs):
        from .llama_rerank import LlamaRerankScorer
        from .types import build_query

        rerank_url = kwargs.get("rerank_url", "http://localhost:8181")
        user_turns = [t for t in turns if t.kind == "user"]

        scorer = LlamaRerankScorer(base_url=rerank_url)
        query = build_query(user_turns)
        return scorer.score_turns(system_turns, query, token_counts)


# ---------------------------------------------------------------------------
# Registry: method name -> scorer instance
# ---------------------------------------------------------------------------

SCORERS: dict[str, Scorer] = {
    "dedup": DedupScorer(),
    "eitf": EitfScorer(),
    "setcover": SetcoverScorer(),
    "embed": EmbedScorer(),
    "llama-embed": LlamaEmbedScorerWrapper(),
    "llama-rerank": LlamaRerankScorerWrapper(),
}

# Methods that don't require external services (fast, local-only)
LOCAL_METHODS = ["dedup", "eitf", "setcover"]

# All standard score-and-select methods (excludes claude-code which uses LLM summarization)
ALL_METHODS = list(SCORERS.keys())


def get_scorer(method: str) -> Scorer:
    """Get a scorer by method name."""
    if method not in SCORERS:
        raise ValueError(
            f"Unknown method: {method}. Available: {', '.join(SCORERS.keys())}"
        )
    return SCORERS[method]
