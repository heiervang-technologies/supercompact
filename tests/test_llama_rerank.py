"""Tests for lib.llama_rerank.LlamaRerankScorer."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.llama_rerank import LlamaRerankScorer, MAX_DOC_CHARS
from lib.parser import Turn
from lib.types import ScoredTurn


def _turn(index: int, text: str) -> Turn:
    t = Turn(kind="system", index=index)
    t.lines = [{"message": {"content": text}}]
    return t


def _make_scorer(mock_httpx_get=None):
    """Create a LlamaRerankScorer with mocked health-check GET."""
    get_response = MagicMock()
    get_response.raise_for_status = MagicMock()
    with patch("lib.llama_rerank.httpx.get", return_value=get_response):
        scorer = LlamaRerankScorer(base_url="http://localhost:8181")
    return scorer


def _mock_post_response(results: list[dict]):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"results": results}
    return resp


class TestLlamaRerankScorerInit:
    def test_health_check_called_on_init(self):
        get_resp = MagicMock()
        get_resp.raise_for_status = MagicMock()
        with patch("lib.llama_rerank.httpx.get", return_value=get_resp) as mock_get:
            LlamaRerankScorer(base_url="http://myserver:9090")
        mock_get.assert_called_once()
        assert "myserver:9090" in mock_get.call_args[0][0]

    def test_url_ends_with_v1_rerank(self):
        get_resp = MagicMock()
        get_resp.raise_for_status = MagicMock()
        with patch("lib.llama_rerank.httpx.get", return_value=get_resp):
            scorer = LlamaRerankScorer(base_url="http://localhost:8181")
        assert scorer.url.endswith("/v1/rerank")

    def test_trailing_slash_stripped_from_base_url(self):
        get_resp = MagicMock()
        get_resp.raise_for_status = MagicMock()
        with patch("lib.llama_rerank.httpx.get", return_value=get_resp):
            scorer = LlamaRerankScorer(base_url="http://localhost:8181/")
        assert "//" not in scorer.url.replace("http://", "")


class TestScoreTurns:
    def test_returns_list_of_scored_turns(self):
        scorer = _make_scorer()
        turns = [_turn(0, "hello"), _turn(1, "world")]
        token_counts = {0: 5, 1: 3}
        post_resp = _mock_post_response([
            {"index": 0, "relevance_score": 0.9},
            {"index": 1, "relevance_score": 0.3},
        ])
        with patch("lib.llama_rerank.httpx.post", return_value=post_resp):
            results = scorer.score_turns(turns, "query", token_counts)
        assert isinstance(results, list)
        assert all(isinstance(r, ScoredTurn) for r in results)

    def test_result_count_matches_input(self):
        scorer = _make_scorer()
        turns = [_turn(i, f"text {i}") for i in range(5)]
        token_counts = {i: 10 for i in range(5)}
        post_resp = _mock_post_response([
            {"index": i, "relevance_score": float(i) / 5}
            for i in range(5)
        ])
        with patch("lib.llama_rerank.httpx.post", return_value=post_resp):
            results = scorer.score_turns(turns, "query", token_counts)
        assert len(results) == 5

    def test_scores_assigned_correctly(self):
        scorer = _make_scorer()
        turns = [_turn(0, "relevant"), _turn(1, "not relevant")]
        token_counts = {0: 5, 1: 5}
        post_resp = _mock_post_response([
            {"index": 0, "relevance_score": 0.95},
            {"index": 1, "relevance_score": 0.1},
        ])
        with patch("lib.llama_rerank.httpx.post", return_value=post_resp):
            results = scorer.score_turns(turns, "query", token_counts)
        assert results[0].score == pytest.approx(0.95)
        assert results[1].score == pytest.approx(0.1)

    def test_token_counts_assigned(self):
        scorer = _make_scorer()
        turns = [_turn(0, "text")]
        token_counts = {0: 42}
        post_resp = _mock_post_response([{"index": 0, "relevance_score": 0.5}])
        with patch("lib.llama_rerank.httpx.post", return_value=post_resp):
            results = scorer.score_turns(turns, "query", token_counts)
        assert results[0].tokens == 42

    def test_missing_token_count_defaults_to_zero(self):
        scorer = _make_scorer()
        turns = [_turn(0, "text")]
        post_resp = _mock_post_response([{"index": 0, "relevance_score": 0.5}])
        with patch("lib.llama_rerank.httpx.post", return_value=post_resp):
            results = scorer.score_turns(turns, "query", token_counts={})
        assert results[0].tokens == 0

    def test_batching_makes_multiple_post_calls(self):
        scorer = _make_scorer()
        turns = [_turn(i, f"turn {i}") for i in range(5)]
        token_counts = {i: 1 for i in range(5)}
        # With batch_size=2, we need ceil(5/2)=3 batches
        call_count = 0

        def fake_post(url, json, timeout):
            nonlocal call_count
            docs = json["documents"]
            batch_offset = call_count * 2
            resp = _mock_post_response([
                {"index": j, "relevance_score": 0.5}
                for j in range(len(docs))
            ])
            call_count += 1
            return resp

        with patch("lib.llama_rerank.httpx.post", side_effect=fake_post):
            scorer.score_turns(turns, "q", token_counts, batch_size=2)
        assert call_count == 3

    def test_document_truncated_to_max_doc_chars(self):
        scorer = _make_scorer()
        long_text = "x" * (MAX_DOC_CHARS + 1000)
        turns = [_turn(0, long_text)]
        token_counts = {}
        captured_docs = []

        def fake_post(url, json, timeout):
            captured_docs.extend(json["documents"])
            return _mock_post_response([{"index": 0, "relevance_score": 0.5}])

        with patch("lib.llama_rerank.httpx.post", side_effect=fake_post):
            scorer.score_turns(turns, "q", token_counts)
        assert len(captured_docs[0]) <= MAX_DOC_CHARS

    def test_empty_turns_returns_empty_list(self):
        scorer = _make_scorer()
        with patch("lib.llama_rerank.httpx.post") as mock_post:
            results = scorer.score_turns([], "query", {})
        assert results == []
        mock_post.assert_not_called()

    def test_raises_on_http_error(self):
        scorer = _make_scorer()
        turns = [_turn(0, "text")]
        resp = MagicMock()
        resp.raise_for_status.side_effect = Exception("HTTP 500")
        with patch("lib.llama_rerank.httpx.post", return_value=resp):
            with pytest.raises(Exception, match="HTTP 500"):
                scorer.score_turns(turns, "query", {})
