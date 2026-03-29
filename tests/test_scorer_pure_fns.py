"""Tests for lib/scorer.py — pure helper functions (no model loading).

Covers _format_instruct and _last_token_pool.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.scorer import _format_instruct, _last_token_pool


# ---------------------------------------------------------------------------
# _format_instruct
# ---------------------------------------------------------------------------

class TestFormatInstruct:
    def test_basic_format(self):
        result = _format_instruct("do task", "hello world")
        assert result == "Instruct: do task\nQuery: hello world"

    def test_instruction_included(self):
        result = _format_instruct("find errors", "some query text")
        assert "Instruct: find errors" in result

    def test_query_included(self):
        result = _format_instruct("find errors", "some query text")
        assert "Query: some query text" in result

    def test_newline_separates_parts(self):
        result = _format_instruct("inst", "qry")
        parts = result.split("\n")
        assert len(parts) == 2
        assert parts[0].startswith("Instruct:")
        assert parts[1].startswith("Query:")

    def test_empty_instruction(self):
        result = _format_instruct("", "query")
        assert result == "Instruct: \nQuery: query"

    def test_empty_query(self):
        result = _format_instruct("instruction", "")
        assert result == "Instruct: instruction\nQuery: "

    def test_multiline_query_preserved(self):
        query = "line1\nline2"
        result = _format_instruct("inst", query)
        # The query section starts after "Query: "
        assert result.endswith(f"\nQuery: {query}")

    def test_returns_string(self):
        result = _format_instruct("a", "b")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _last_token_pool — right-padding (variable-length sequences)
# ---------------------------------------------------------------------------

class TestLastTokenPoolRightPadding:
    """When attention_mask has right padding, pool at actual last non-pad position."""

    def _make_hidden(self, batch: int, seq: int, dim: int) -> torch.Tensor:
        return torch.zeros(batch, seq, dim)

    def test_single_seq_full_mask_returns_last_token(self):
        hidden = torch.zeros(1, 4, 3)
        hidden[0, 3, :] = torch.tensor([1.0, 2.0, 3.0])
        mask = torch.ones(1, 4, dtype=torch.long)  # no padding → right-pad variant
        result = _last_token_pool(hidden, mask)
        assert result.shape == (1, 3)
        assert torch.allclose(result[0], torch.tensor([1.0, 2.0, 3.0]))

    def test_right_padded_returns_last_real_token(self):
        hidden = torch.zeros(1, 5, 4)
        hidden[0, 2, :] = torch.tensor([1.0, 2.0, 3.0, 4.0])
        # seq len = 3, padded to 5
        mask = torch.tensor([[1, 1, 1, 0, 0]])
        result = _last_token_pool(hidden, mask)
        assert torch.allclose(result[0], torch.tensor([1.0, 2.0, 3.0, 4.0]))

    def test_batch_with_different_lengths(self):
        hidden = torch.zeros(2, 4, 3)
        # Seq 0: real length 2, last real token at index 1
        hidden[0, 1, :] = torch.tensor([10.0, 20.0, 30.0])
        # Seq 1: real length 3, last real token at index 2
        hidden[1, 2, :] = torch.tensor([40.0, 50.0, 60.0])
        mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
        result = _last_token_pool(hidden, mask)
        assert result.shape == (2, 3)
        assert torch.allclose(result[0], torch.tensor([10.0, 20.0, 30.0]))
        assert torch.allclose(result[1], torch.tensor([40.0, 50.0, 60.0]))

    def test_output_shape_matches_batch_and_dim(self):
        hidden = torch.randn(3, 6, 8)
        mask = torch.ones(3, 6, dtype=torch.long)
        result = _last_token_pool(hidden, mask)
        assert result.shape == (3, 8)

    def test_single_token_sequence(self):
        hidden = torch.zeros(1, 1, 4)
        hidden[0, 0, :] = torch.tensor([7.0, 8.0, 9.0, 10.0])
        mask = torch.tensor([[1]])
        result = _last_token_pool(hidden, mask)
        assert torch.allclose(result[0], torch.tensor([7.0, 8.0, 9.0, 10.0]))


# ---------------------------------------------------------------------------
# _last_token_pool — left-padding (Qwen3 default)
# ---------------------------------------------------------------------------

class TestLastTokenPoolLeftPadding:
    """When attention_mask has left padding (all last columns = 1), pool last index."""

    def _left_pad_mask(self, batch: int, seq: int, real_len: int) -> torch.Tensor:
        """Create a left-padded mask where real tokens are at the end."""
        mask = torch.zeros(batch, seq, dtype=torch.long)
        mask[:, seq - real_len :] = 1
        return mask

    def test_left_padded_returns_last_token(self):
        hidden = torch.zeros(1, 5, 3)
        hidden[0, 4, :] = torch.tensor([5.0, 6.0, 7.0])
        # Left-padded: real tokens at indices 2,3,4
        mask = torch.tensor([[0, 0, 1, 1, 1]])
        result = _last_token_pool(hidden, mask)
        assert torch.allclose(result[0], torch.tensor([5.0, 6.0, 7.0]))

    def test_batch_left_padded_all_return_index_minus1(self):
        batch, seq, dim = 3, 6, 4
        hidden = torch.randn(batch, seq, dim)
        # All sequences fully populated (mask all-ones is NOT left-padding per this fn)
        # Create left-padding: pad at start, real at end
        mask = torch.zeros(batch, seq, dtype=torch.long)
        mask[:, 3:] = 1  # last 3 tokens are real for all seqs
        result = _last_token_pool(hidden, mask)
        # Left-padding detected → returns last (index -1) hidden state
        assert result.shape == (batch, dim)
        assert torch.allclose(result, hidden[:, -1, :])

    def test_left_pad_output_shape(self):
        hidden = torch.randn(2, 5, 16)
        # Left-padded: last column all 1s → left_padding=True
        mask = torch.zeros(2, 5, dtype=torch.long)
        mask[:, -1] = 1
        result = _last_token_pool(hidden, mask)
        assert result.shape == (2, 16)
