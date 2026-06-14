"""Token counting using Qwen3 tokenizer with graceful fallback."""

from __future__ import annotations

import os
import sys

from .parser import Turn, extract_text

_tokenizer = None
_fallback = False


def _get_tokenizer():
    """Load Qwen3 tokenizer lazily. On any failure (missing model, perm
    error, offline, transformers absent) switch to a byte-heuristic
    fallback so EITF/dedup/setcover methods still work without a model."""
    global _tokenizer, _fallback
    if _tokenizer is not None or _fallback:
        return _tokenizer
    try:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
    except Exception as e:
        _fallback = True
        print(
            f"[tokenizer] WARN: Qwen tokenizer unavailable ({type(e).__name__}: {e}); "
            f"using byte-heuristic fallback (len/4).",
            file=sys.stderr,
        )
    return _tokenizer


def estimate_tokens(text: str) -> int:
    """Count tokens using the Qwen3 tokenizer, or byte-heuristic fallback."""
    tok = _get_tokenizer()
    if tok is None:
        # Rough heuristic: ~4 bytes/token for English; slightly under-counts
        # CJK but safe for budget enforcement.
        return max(1, len(text) // 4) if text else 0
    return len(tok.encode(text, add_special_tokens=False))


def turn_tokens(turn: Turn) -> int:
    """Count the tokens of an entire turn.

    Uses the untruncated extraction so tool_use payloads (which the API
    receives in full) are counted at their real size, not the scoring stub.
    """
    return estimate_tokens(extract_text(turn, truncate=False))
