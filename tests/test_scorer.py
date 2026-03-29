"""Tests for lib/scorer.py — pure helper functions."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.scorer import _last_token_pool, _format_instruct, MODEL_ID, ENCODE_MAX_LENGTH


# ---------------------------------------------------------------------------
# _format_instruct
# ---------------------------------------------------------------------------

class TestFormatInstruct:
    def test_contains_instruction(self):
        result = _format_instruct("Do something", "my text")
        assert "Do something" in result

    def test_contains_query(self):
        result = _format_instruct("Do something", "my text")
        assert "my text" in result

    def test_instruct_prefix(self):
        result = _format_instruct("Find things", "hello")
        assert result.startswith("Instruct:")

    def test_query_prefix(self):
        result = _format_instruct("Find things", "hello")
        assert "Query:" in result

    def test_instruction_before_query(self):
        result = _format_instruct("INSTR", "QTEXT")
        assert result.index("INSTR") < result.index("QTEXT")

    def test_empty_instruction(self):
        result = _format_instruct("", "hello")
        assert "hello" in result

    def test_empty_query(self):
        result = _format_instruct("Do it", "")
        assert "Do it" in result

    def test_returns_string(self):
        assert isinstance(_format_instruct("x", "y"), str)


# ---------------------------------------------------------------------------
# _last_token_pool
# ---------------------------------------------------------------------------

class TestLastTokenPool:
    def _hidden(self, batch, seq_len, hidden=4):
        """Create a simple last_hidden_states tensor."""
        return torch.arange(
            batch * seq_len * hidden, dtype=torch.float
        ).reshape(batch, seq_len, hidden)

    def test_right_padding_picks_last_real_token(self):
        # seq of length 3, real tokens at 0,1 — padding at 2
        hidden = self._hidden(1, 3)
        mask = torch.tensor([[1, 1, 0]], dtype=torch.long)
        result = _last_token_pool(hidden, mask)
        # attention_mask sums to 2 → sequence_length - 1 = 1 → hidden[0, 1]
        expected = hidden[0, 1]
        assert torch.allclose(result, expected.unsqueeze(0))

    def test_left_padding_picks_last_position(self):
        # left-padding: mask[-1] is 1 for all rows → sum == batch_size
        hidden = self._hidden(1, 3)
        mask = torch.tensor([[0, 1, 1]], dtype=torch.long)
        result = _last_token_pool(hidden, mask)
        # left-padded path returns hidden[:, -1]
        expected = hidden[0, 2]
        assert torch.allclose(result, expected.unsqueeze(0))

    def test_output_shape(self):
        B, S, H = 3, 5, 8
        hidden = torch.randn(B, S, H)
        mask = torch.ones(B, S, dtype=torch.long)
        result = _last_token_pool(hidden, mask)
        assert result.shape == (B, H)

    def test_no_padding_picks_last(self):
        hidden = self._hidden(1, 4)
        mask = torch.ones(1, 4, dtype=torch.long)
        result = _last_token_pool(hidden, mask)
        expected = hidden[0, 3]
        assert torch.allclose(result, expected.unsqueeze(0))

    def test_batch_of_two_right_padded(self):
        B, S, H = 2, 4, 4
        hidden = torch.arange(B * S * H, dtype=torch.float).reshape(B, S, H)
        # First row: 3 real tokens; second row: 2 real tokens
        mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.long)
        result = _last_token_pool(hidden, mask)
        assert torch.allclose(result[0], hidden[0, 2])
        assert torch.allclose(result[1], hidden[1, 1])


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

class TestModuleConstants:
    def test_model_id_is_string(self):
        assert isinstance(MODEL_ID, str)
        assert len(MODEL_ID) > 0

    def test_encode_max_length_positive(self):
        assert ENCODE_MAX_LENGTH > 0

    def test_encode_max_length_int(self):
        assert isinstance(ENCODE_MAX_LENGTH, int)
