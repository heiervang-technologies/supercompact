"""Tests for lib/llama_rerank.py — LlamaRerankScorer.

Covers: constants, __init__ (mocked health check), score_turns (mocked HTTP POST).
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.llama_rerank import LlamaRerankScorer, MAX_DOC_CHARS
from lib.parser import Turn
from lib.types import ScoredTurn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _turn(index: int, text: str) -> Turn:
    t = Turn(kind="system", index=index)
    t.append({"message": {"content": text}})
    return t


def _make_scorer(base_url: str = "http://localhost:8181") -> LlamaRerankScorer:
    """Create a scorer with mocked health check."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    with patch("lib.llama_rerank.httpx.get", return_value=mock_resp):
        return LlamaRerankScorer(base_url=base_url)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_max_doc_chars_is_int(self):
        assert isinstance(MAX_DOC_CHARS, int)

    def test_max_doc_chars_positive(self):
        assert MAX_DOC_CHARS > 0

    def test_max_doc_chars_reasonable_value(self):
        # Should be at least 500 and at most 16384
        assert 500 <= MAX_DOC_CHARS <= 16384


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

class TestLlamaRerankScorerInit:
    def test_health_check_called_on_init(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        with patch("lib.llama_rerank.httpx.get", return_value=mock_resp) as mock_get:
            LlamaRerankScorer(base_url="http://localhost:8181")
            mock_get.assert_called_once()

    def test_url_has_rerank_path(self):
        scorer = _make_scorer()
        assert scorer.url.endswith("/v1/rerank")

    def test_custom_base_url(self):
        scorer = _make_scorer("http://myserver:9090")
        assert "myserver:9090" in scorer.url

    def test_trailing_slash_stripped(self):
        scorer = _make_scorer("http://localhost:8181/")
        assert not scorer.url.endswith("//")

    def test_health_check_failure_propagates(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("Connection refused")
        with patch("lib.llama_rerank.httpx.get", return_value=mock_resp):
            with pytest.raises(Exception, match="Connection refused"):
                LlamaRerankScorer()


# ---------------------------------------------------------------------------
# score_turns
# ---------------------------------------------------------------------------

class TestScoreTurns:
    def _make_rerank_response(self, results: list[dict]) -> MagicMock:
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"results": results}
        return mock_resp

    def test_returns_list(self):
        scorer = _make_scorer()
        turns = [_turn(0, "hello world")]
        token_counts = {0: 5}
        results_data = [{"index": 0, "relevance_score": 0.8}]
        with patch("lib.llama_rerank.httpx.post", return_value=self._make_rerank_response(results_data)):
            result = scorer.score_turns(turns, "query", token_counts)
        assert isinstance(result, list)

    def test_returns_scored_turns(self):
        scorer = _make_scorer()
        turns = [_turn(0, "text")]
        results_data = [{"index": 0, "relevance_score": 0.7}]
        with patch("lib.llama_rerank.httpx.post", return_value=self._make_rerank_response(results_data)):
            result = scorer.score_turns(turns, "query", {0: 10})
        assert all(isinstance(s, ScoredTurn) for s in result)

    def test_one_turn_one_result(self):
        scorer = _make_scorer()
        turns = [_turn(0, "hello")]
        results_data = [{"index": 0, "relevance_score": 0.5}]
        with patch("lib.llama_rerank.httpx.post", return_value=self._make_rerank_response(results_data)):
            result = scorer.score_turns(turns, "query", {0: 5})
        assert len(result) == 1

    def test_score_set_from_response(self):
        scorer = _make_scorer()
        turns = [_turn(0, "relevant text")]
        results_data = [{"index": 0, "relevance_score": 0.95}]
        with patch("lib.llama_rerank.httpx.post", return_value=self._make_rerank_response(results_data)):
            result = scorer.score_turns(turns, "query", {0: 10})
        assert result[0].score == pytest.approx(0.95)

    def test_tokens_from_token_counts(self):
        scorer = _make_scorer()
        turns = [_turn(0, "text")]
        results_data = [{"index": 0, "relevance_score": 0.5}]
        with patch("lib.llama_rerank.httpx.post", return_value=self._make_rerank_response(results_data)):
            result = scorer.score_turns(turns, "query", {0: 42})
        assert result[0].tokens == 42

    def test_multiple_turns_all_scored(self):
        scorer = _make_scorer()
        turns = [_turn(i, f"text {i}") for i in range(3)]
        results_data = [
            {"index": 0, "relevance_score": 0.8},
            {"index": 1, "relevance_score": 0.5},
            {"index": 2, "relevance_score": 0.3},
        ]
        token_counts = {0: 5, 1: 8, 2: 6}
        with patch("lib.llama_rerank.httpx.post", return_value=self._make_rerank_response(results_data)):
            result = scorer.score_turns(turns, "query", token_counts)
        assert len(result) == 3

    def test_empty_turns_returns_empty(self):
        scorer = _make_scorer()
        results_data = []
        with patch("lib.llama_rerank.httpx.post", return_value=self._make_rerank_response(results_data)):
            result = scorer.score_turns([], "query", {})
        assert result == []

    def test_missing_index_gets_zero_score(self):
        scorer = _make_scorer()
        # Turn 0 exists but server response doesn't include it (unusual edge case)
        turns = [_turn(0, "text")]
        results_data = []  # No results returned
        with patch("lib.llama_rerank.httpx.post", return_value=self._make_rerank_response(results_data)):
            result = scorer.score_turns(turns, "query", {0: 5})
        assert result[0].score == 0.0

    def test_batching_uses_multiple_post_calls(self):
        scorer = _make_scorer()
        turns = [_turn(i, f"text {i}") for i in range(5)]
        results_per_batch = [{"index": j, "relevance_score": 0.5} for j in range(3)]

        call_count = 0
        def fake_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return self._make_rerank_response(results_per_batch[:min(3, 5 - (call_count - 1) * 3)])

        with patch("lib.llama_rerank.httpx.post", side_effect=fake_post):
            scorer.score_turns(turns, "query", {i: 5 for i in range(5)}, batch_size=3)

        assert call_count == 2  # 5 turns / batch_size=3 = 2 batches
