"""Fitness function for evaluating compaction quality.

Measures how well a compaction method preserves information that the
conversation will need in the future.

Approach:
  1. Split the conversation at a configurable point (default 70%)
  2. Extract "information units" from the suffix (future) turns — words
     and n-grams that appear in future assistant responses
  3. For each prefix system turn, compute a "future relevance" score —
     how many suffix information units does it contain?
  4. After compaction of the prefix, recall = sum(relevance of kept turns)
     / sum(relevance of all prefix turns)

This directly answers: "Of the information the future conversation needs,
how much did we preserve?"
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from dataclasses import dataclass

from .parser import Turn, extract_text
from .tokenizer import turn_tokens
from .scorer import ScoredTurn
from .selector import select_turns, SelectionResult


# Words shorter than this are ignored (filters stopwords, articles, etc.)
MIN_WORD_LEN = 4

# Regex for extracting meaningful tokens
_WORD_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_./\-]{3,}")
_PATH_RE = re.compile(r"(?:[./~])?(?:/[\w.\-]+){2,}")


def _extract_vocab(text: str) -> Counter[str]:
    """Extract meaningful words from text, lowercased.

    Returns a Counter of word -> frequency.
    Includes file paths as single tokens and regular words.
    """
    vocab: Counter[str] = Counter()

    # File paths first (kept as-is, lowercased)
    for path_match in _PATH_RE.finditer(text):
        vocab[path_match.group().lower()] += 1

    # Regular words
    for word_match in _WORD_RE.finditer(text):
        w = word_match.group().lower()
        if len(w) >= MIN_WORD_LEN:
            vocab[w] += 1

    return vocab


def _idf(term: str, doc_vocabs: list[Counter[str]], total_docs: int) -> float:
    """Inverse document frequency of a term across document set."""
    df = sum(1 for v in doc_vocabs if term in v)
    if df == 0:
        return 0.0
    return math.log(1 + total_docs / df)


@dataclass
class FitnessResult:
    """Result of fitness evaluation for a single compaction method."""

    method: str
    recall: float           # 0-1, fraction of future-relevant info preserved
    speed_s: float          # wall-clock seconds for compaction
    compression: float      # kept_tokens / total_tokens
    budget: int
    total_tokens: int
    kept_tokens: int
    prefix_turns: int       # number of turns in prefix
    suffix_turns: int       # number of turns in suffix
    suffix_vocab_size: int  # unique information units in suffix
    scored_count: int       # number of system turns that were scored
    kept_scored: int        # number of scored turns that were kept
    dropped_scored: int     # number of scored turns that were dropped

    @property
    def f1(self) -> float:
        """F1-like score: harmonic mean of recall and compression efficiency.

        compression_eff = 1 - compression (how much we compressed).
        A method that keeps everything has recall=1 but compression_eff=0.
        A method that drops everything has compression_eff=1 but recall=0.
        """
        compression_eff = 1.0 - self.compression
        if self.recall + compression_eff == 0:
            return 0.0
        return 2 * self.recall * compression_eff / (self.recall + compression_eff)


def evaluate(
    turns: list[Turn],
    method: str,
    budget: int = 80_000,
    split_ratio: float = 0.70,
    short_threshold: int = 300,
    min_repeat_len: int = 64,
    device: str = "cpu",
    batch_size: int = 16,
) -> FitnessResult:
    """Run a compaction method and evaluate its fitness.

    Splits the conversation at split_ratio, compacts the prefix,
    and measures how well future-relevant information is preserved.
    """
    # --- 1. Split into prefix and suffix ---
    split_idx = int(len(turns) * split_ratio)
    # Snap to a user turn boundary (don't split mid-exchange)
    while split_idx < len(turns) and turns[split_idx].kind != "user":
        split_idx += 1

    prefix_turns = turns[:split_idx]
    suffix_turns = turns[split_idx:]

    if not prefix_turns or not suffix_turns:
        raise ValueError(
            f"Split at {split_ratio:.0%} ({split_idx}/{len(turns)}) "
            f"produced empty prefix or suffix"
        )

    # Re-index prefix turns (selector expects 0-based sequential indices)
    for i, t in enumerate(prefix_turns):
        t.index = i

    # --- 2. Extract suffix vocabulary (future information units) ---
    suffix_text_parts = [extract_text(t) for t in suffix_turns if t.kind == "system"]
    suffix_combined = "\n".join(suffix_text_parts)
    suffix_vocab = _extract_vocab(suffix_combined)

    if not suffix_vocab:
        raise ValueError("No meaningful vocabulary extracted from suffix")

    # --- 3. Token counts for prefix ---
    token_counts: dict[int, int] = {}
    for t in prefix_turns:
        token_counts[t.index] = turn_tokens(t)

    total_prefix_tokens = sum(token_counts.values())

    # --- 4. Compute per-turn relevance to suffix ---
    prefix_system = [t for t in prefix_turns if t.kind == "system"]
    prefix_long = [t for t in prefix_system if token_counts.get(t.index, 0) > short_threshold]

    # Build IDF over prefix system turns
    turn_vocabs = {t.index: _extract_vocab(extract_text(t)) for t in prefix_system}
    prefix_vocab_list = list(turn_vocabs.values())
    n_docs = len(prefix_vocab_list)

    # Relevance = TF-IDF weighted overlap with suffix vocab
    turn_relevance: dict[int, float] = {}
    for t in prefix_system:
        tv = turn_vocabs[t.index]
        score = 0.0
        for word, tf in tv.items():
            if word in suffix_vocab:
                idf_val = _idf(word, prefix_vocab_list, n_docs)
                score += tf * idf_val * suffix_vocab[word]
        turn_relevance[t.index] = score

    total_relevance = sum(turn_relevance.values())

    # --- 5. Run compaction ---
    t_start = time.monotonic()

    scored: list[ScoredTurn]

    if method == "dedup":
        from .dedup import dedup_scores
        scored = dedup_scores(prefix_turns, prefix_long, token_counts, min_repeat_len=min_repeat_len)
    elif method == "embed":
        from .scorer import Scorer, build_query
        user_turns = [t for t in prefix_turns if t.kind == "user"]
        scorer = Scorer(device=device)
        query = build_query(user_turns)
        scored = scorer.score_turns(prefix_long, query, token_counts, batch_size=batch_size)
    else:
        raise ValueError(f"Unknown method: {method}")

    result = select_turns(
        turns=prefix_turns,
        scored=scored,
        token_counts=token_counts,
        budget=budget,
        short_threshold=short_threshold,
    )

    t_elapsed = time.monotonic() - t_start

    # --- 6. Compute recall ---
    kept_indices = {t.index for t in result.kept_turns}
    kept_relevance = sum(
        turn_relevance.get(idx, 0.0)
        for idx in kept_indices
        if idx in turn_relevance
    )

    recall = kept_relevance / total_relevance if total_relevance > 0 else 1.0

    kept_tokens = sum(token_counts.get(t.index, 0) for t in result.kept_turns)

    return FitnessResult(
        method=method,
        recall=recall,
        speed_s=t_elapsed,
        compression=kept_tokens / total_prefix_tokens if total_prefix_tokens > 0 else 0,
        budget=budget,
        total_tokens=total_prefix_tokens,
        kept_tokens=kept_tokens,
        prefix_turns=len(prefix_turns),
        suffix_turns=len(suffix_turns),
        suffix_vocab_size=len(suffix_vocab),
        scored_count=len(scored),
        kept_scored=len(result.kept_scored),
        dropped_scored=len(result.dropped_turns),
    )
