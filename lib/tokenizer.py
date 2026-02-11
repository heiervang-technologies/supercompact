"""Token counting using Qwen3 tokenizer."""

from __future__ import annotations

from transformers import AutoTokenizer

from .parser import Turn, extract_text

_tokenizer = None


def _get_tokenizer() -> AutoTokenizer:
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
    return _tokenizer


def estimate_tokens(text: str) -> int:
    """Count tokens using the Qwen3 tokenizer."""
    return len(_get_tokenizer().encode(text, add_special_tokens=False))


def turn_tokens(turn: Turn) -> int:
    """Count the tokens of an entire turn."""
    return estimate_tokens(extract_text(turn))
