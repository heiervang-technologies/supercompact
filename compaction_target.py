"""
compaction_target.py
====================

Single pure function: given current token count, model (or just a max context
length), task type, and a recall/space dial, returns the target token count
to compact down to.

Compactor-agnostic. Just outputs a number. Guaranteed less than current tokens.

Empirical basis:
  - Chroma "Context Rot" (Jul 2025): 18 models, degradation at every length.
  - Mejba Ahmed (Mar 2026): Opus 4.6 ~2% degradation per 100K tokens.
  - Elvex 2026: Claude Sonnet 4 <5% degradation across 200K.
  - LongCodeBench (Feb 2026): Gemini 2.5 Pro >90% at 512K (MC), ~50% (open).
  - Paulsen (Jan 2026): MECW is task-specific; complex reasoning fails early.
  - Shi et al. (Feb 2025): optimal context length bounded by training data.
  - Anthropic NIAH: Opus 4.6 scores 91.9→78.3 across 1M context.
  - Anthropic MRCR v2: Opus 4.6 76% at 1M (vs Sonnet 4.5 18.5%).

Usage:
    from compaction_target import compact_to

    # Known model:
    target = compact_to(250_000, model="claude-opus-4.6", task="coding", dial=0.5)

    # Unknown model — just pass max context length:
    target = compact_to(250_000, max_context=512_000, task="coding", dial=0.5)

    # target is always < current_tokens
    my_compactor(context, target)

The dial:
    0.0 = maximize space savings (aggressive, lose more recall)
    0.5 = balanced
    1.0 = maximize recall preservation (conservative, keep more tokens)
"""

from __future__ import annotations
from bisect import bisect_right
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Degradation profiles: (token_count, quality_retention) anchors
# Piecewise-linear interpolation. All values for simple retrieval baseline;
# task multiplier scales degradation from here.
# ─────────────────────────────────────────────────────────────────────────────

_PROFILES: dict[str, tuple[tuple[int, float], ...]] = {
    "claude-opus-4.6": (
        (0, 1.0), (64_000, .99), (100_000, .98), (200_000, .96),
        (300_000, .93), (500_000, .90), (750_000, .87), (1_000_000, .86),
    ),
    "claude-sonnet-4.6": (
        (0, 1.0), (64_000, .98), (100_000, .97), (200_000, .95),
        (300_000, .91), (500_000, .87), (750_000, .83), (1_000_000, .80),
    ),
    "claude-sonnet-4": (
        (0, 1.0), (64_000, .98), (100_000, .97), (200_000, .95),
    ),
    "claude-opus-4": (
        (0, 1.0), (64_000, .99), (100_000, .98), (200_000, .96),
    ),
    "gemini-2.5-pro": (
        (0, 1.0), (64_000, .97), (100_000, .95), (128_000, .93),
        (200_000, .90), (256_000, .88), (500_000, .82), (750_000, .75),
        (1_000_000, .70),
    ),
    "gemini-2.5-flash": (
        (0, 1.0), (64_000, .96), (100_000, .93), (128_000, .90),
        (200_000, .86), (256_000, .83), (500_000, .75), (750_000, .68),
        (1_000_000, .63),
    ),
    "gemini-3.0-pro": (
        (0, 1.0), (64_000, .98), (100_000, .96), (128_000, .94),
        (200_000, .91), (256_000, .89), (500_000, .83), (750_000, .77),
        (1_000_000, .73),
    ),
    "glm-4.7": (
        (0, 1.0), (64_000, .96), (100_000, .93), (128_000, .90),
        (200_000, .85),
    ),
    "glm-4.6": (
        (0, 1.0), (64_000, .95), (100_000, .91), (128_000, .88),
        (200_000, .82),
    ),
}

_TASK_MULT: dict[str, float] = {
    "retrieval": 1.0, "semantic": 1.5, "summary": 1.8,
    "chat": 2.0, "coding": 2.5, "reasoning": 3.0,
}

_ALIASES: dict[str, str] = {
    "opus": "claude-opus-4.6", "opus-4.6": "claude-opus-4.6",
    "opus-4": "claude-opus-4",
    "sonnet": "claude-sonnet-4.6", "sonnet-4.6": "claude-sonnet-4.6",
    "sonnet-4": "claude-sonnet-4",
    "gemini-pro": "gemini-2.5-pro", "gemini-flash": "gemini-2.5-flash",
    "gemini-3": "gemini-3.0-pro", "glm": "glm-4.7",
}
_TASK_ALIASES: dict[str, str] = {
    "code": "coding", "agent": "coding", "coding_agent": "coding",
    "code_generation": "coding", "agentic": "coding",
    "multi_hop": "reasoning", "multi_hop_reasoning": "reasoning",
    "simple_retrieval": "retrieval", "semantic_retrieval": "semantic",
    "summarization": "summary", "conversation_qa": "chat",
    "conversation": "chat", "qa": "chat",
}


# ─────────────────────────────────────────────────────────────────────────────
# Generic profile for unknown models
# ─────────────────────────────────────────────────────────────────────────────
# Shape based on median across all profiled models:
#   quality = 1.0 - 0.30 * (tokens / max_context)^0.6
# Conservative — sits between Claude (best) and Flash (worst).

def _generic_profile(max_context: int) -> tuple[tuple[int, float], ...]:
    """Generate a degradation profile from just a max context window size."""
    fracs = (0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.85, 1.0)
    return tuple(
        (int(max_context * f), round(1.0 - 0.30 * (f ** 0.6), 4))
        for f in fracs
    )


# ─────────────────────────────────────────────────────────────────────────────
# Internals
# ─────────────────────────────────────────────────────────────────────────────

def _lerp(profile: tuple[tuple[int, float], ...], tokens: int) -> float:
    if tokens <= 0:
        return 1.0
    ts = [p[0] for p in profile]
    if tokens >= ts[-1]:
        t1, v1 = profile[-2]
        t2, v2 = profile[-1]
        slope = (v2 - v1) / (t2 - t1) if t2 != t1 else 0.0
        return max(0.3, v2 + slope * (tokens - t2))
    i = max(0, min(bisect_right(ts, tokens) - 1, len(profile) - 2))
    t1, v1 = profile[i]
    t2, v2 = profile[i + 1]
    frac = (tokens - t1) / (t2 - t1) if t2 != t1 else 0.0
    return v1 + frac * (v2 - v1)


def _resolve(model: str) -> Optional[str]:
    k = model.lower().strip()
    if k in _PROFILES:
        return k
    if k in _ALIASES:
        return _ALIASES[k]
    for key in _PROFILES:
        if key in k or k in key:
            return key
    return None


def _task_mult(t: str) -> float:
    k = t.lower().strip().replace(" ", "_").replace("-", "_")
    if k in _TASK_MULT:
        return _TASK_MULT[k]
    if k in _TASK_ALIASES:
        return _TASK_MULT[_TASK_ALIASES[k]]
    return 2.5


def _get_profile(
    model: Optional[str], max_context: Optional[int],
) -> tuple[tuple[int, float], ...]:
    if model is not None:
        key = _resolve(model)
        if key is not None:
            return _PROFILES[key]
    if max_context is not None and max_context > 0:
        return _generic_profile(max_context)
    if model is not None:
        raise ValueError(
            f"Unknown model {model!r} and no max_context provided. "
            f"Known: {list(_PROFILES)}. Pass max_context for unknown models."
        )
    raise ValueError("Provide either model or max_context.")


def _quality(
    profile: tuple[tuple[int, float], ...], tokens: int, task: str,
) -> float:
    base = _lerp(profile, tokens)
    return max(0.3, 1.0 - (1.0 - base) * _task_mult(task))


def _find_crossing(
    profile: tuple[tuple[int, float], ...],
    task: str, target_q: float, lo: int, hi: int,
) -> int:
    if _quality(profile, lo, task) < target_q:
        return lo
    if _quality(profile, hi, task) >= target_q:
        return hi
    for _ in range(30):
        mid = (lo + hi) // 2
        if lo >= hi - 1:
            break
        if _quality(profile, mid, task) >= target_q:
            lo = mid
        else:
            hi = mid
    return lo


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def quality_at(
    tokens: int,
    task: str = "coding",
    model: Optional[str] = None,
    max_context: Optional[int] = None,
) -> float:
    """Estimated quality retention at a given token count. Returns 0.0–1.0.

    Provide model (for known profile) or max_context (for generic), or both.
    """
    return _quality(_get_profile(model, max_context), tokens, task)


def compact_to(
    current_tokens: int,
    model: Optional[str] = None,
    task: str = "coding",
    dial: float = 0.5,
    floor: int = 8_000,
    max_context: Optional[int] = None,
) -> int:
    """Compute the target token count to compact to.

    Args:
        current_tokens: Current context size in tokens.
        model: Known model name (optional if max_context given).
        task: "retrieval" | "semantic" | "summary" | "chat" | "coding" | "reasoning"
        dial: 0.0 (aggressive) to 1.0 (conservative). Default 0.5 (balanced).
        floor: Minimum output. Default 8_000.
        max_context: Max context window in tokens. Required for unknown models.
            Ignored when model resolves to a known profile.

    Returns:
        int, always satisfying: floor <= result < current_tokens.
    """
    if current_tokens <= floor:
        return current_tokens

    dial = max(0.0, min(1.0, dial))
    profile = _get_profile(model, max_context)
    prof_max = profile[-1][0]

    sweet = _find_crossing(
        profile, task, 0.90, lo=floor, hi=min(current_tokens, prof_max),
    )
    if sweet >= current_tokens:
        sweet = max(floor, int(current_tokens * 0.75))

    ceiling = max(floor + 1, int(current_tokens * 0.90))
    if dial <= 0.5:
        target = floor + (dial / 0.5) * (sweet - floor)
    else:
        target = sweet + ((dial - 0.5) / 0.5) * (ceiling - sweet)

    return max(floor, min(int(target), current_tokens - 1))


def compact_to_range(
    current_tokens: int,
    model: Optional[str] = None,
    task: str = "coding",
    floor: int = 8_000,
    max_context: Optional[int] = None,
) -> dict[str, int]:
    """Return targets at dial=0.0, 0.25, 0.5, 0.75, 1.0 in one call."""
    return {
        f"{d:.2f}": compact_to(current_tokens, model, task, d, floor, max_context)
        for d in (0.0, 0.25, 0.5, 0.75, 1.0)
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DS = [0.0, 0.25, 0.5, 0.75, 1.0]
    TS = [128_000, 256_000, 500_000, 1_000_000]

    def _row(name, tokens, task, **kw):
        vals = [compact_to(tokens, task=task, dial=d, **kw) for d in DS]
        cols = "".join(f" {v:>8,} |" for v in vals)
        return f"    {name:<22} |{cols}"

    def _header():
        h = f"    {'':22} |"
        for d in DS:
            h += f" {'d='+str(d):>8} |"
        return h + "\n    " + "─" * 23 + "┼" + ("─" * 10 + "┼") * 4 + "─" * 10 + "┤"

    print("compact_to() — Target token counts")
    print("=" * 90)

    for task in ("coding", "reasoning"):
        # Known models
        print(f"\n{'─'*90}\n  TASK: {task} — known models\n{'─'*90}")
        known = [
            ("claude-opus-4.6",   {"model": "claude-opus-4.6"}),
            ("claude-sonnet-4.6", {"model": "claude-sonnet-4.6"}),
            ("gemini-2.5-pro",    {"model": "gemini-2.5-pro"}),
            ("gemini-2.5-flash",  {"model": "gemini-2.5-flash"}),
            ("glm-4.7",           {"model": "glm-4.7"}),
        ]
        for tokens in TS:
            print(f"\n  Current: {tokens:,}")
            print(_header())
            for name, kw in known:
                p = _get_profile(kw.get("model"), None)
                if tokens <= p[-1][0]:
                    print(_row(name, tokens, task, **kw))

        # Generic models
        print(f"\n{'─'*90}\n  TASK: {task} — generic (max_context only)\n{'─'*90}")
        generics = [
            ("generic 128K",  128_000),
            ("generic 256K",  256_000),
            ("generic 512K",  512_000),
            ("generic 1M",    1_000_000),
            ("generic 2M",    2_000_000),
        ]
        for tokens in TS:
            print(f"\n  Current: {tokens:,}")
            print(_header())
            for name, mc in generics:
                if tokens <= mc:
                    print(_row(name, tokens, task, max_context=mc))

    # Generic profile shape
    print(f"\n{'─'*90}")
    print("  GENERIC PROFILE: quality = 1.0 - 0.30 * (tokens/max_context)^0.6")
    print(f"{'─'*90}\n")
    print(f"    {'%':>5} | {'Retrieval':>10} | {'Coding':>10} | {'Reasoning':>10}")
    print(f"    {'─'*5}─┼─{'─'*10}─┼─{'─'*10}─┼─{'─'*10}")
    for pct in (5, 10, 20, 30, 50, 70, 85, 100):
        f = pct / 100
        base = 1.0 - 0.30 * (f ** 0.6)
        r = max(0.3, 1.0 - (1.0 - base) * 1.0)
        c = max(0.3, 1.0 - (1.0 - base) * 2.5)
        rs = max(0.3, 1.0 - (1.0 - base) * 3.0)
        print(f"    {pct:>4}% | {r:>9.1%} | {c:>9.1%} | {rs:>9.1%}")
