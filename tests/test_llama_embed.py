"""Tests for lib/llama_embed.py and lib/llama_rerank.py — HTTP scorer clients."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.llama_embed import _instruct, LlamaEmbedScorer, QUERY_INSTRUCTION, DOC_INSTRUCTION, MAX_DOC_CHARS
from lib.llama_rerank import LlamaRerankScorer
from lib.parser import Turn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _turn(index: int = 0, text: str = "some content") -> Turn:
    t = Turn(kind="system", index=index)
    t.lines.append({
        "type": "assistant",
        "message": {"role": "assistant", "content": text},
    })
    return t


def _mock_health_ok() -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    return resp


def _mock_embed_response(embeddings: list[list[float]]) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {
        "data": [
            {"index": i, "embedding": emb}
            for i, emb in enumerate(embeddings)
        ]
    }
    return resp


# ---------------------------------------------------------------------------
# _instruct (pure function)
# ---------------------------------------------------------------------------

class TestInstruct:
    def test_contains_instruction(self):
        result = _instruct("Find documents", "my query")
        assert "Find documents" in result

    def test_contains_query(self):
        result = _instruct("instruction text", "my query")
        assert "my query" in result

    def test_format(self):
        result = _instruct("Do something", "hello")
        assert result == "Instruct: Do something\nQuery: hello"

    def test_empty_instruction(self):
        result = _instruct("", "query text")
        assert "query text" in result

    def test_empty_query(self):
        result = _instruct("instruction", "")
        assert "instruction" in result


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_query_instruction_non_empty(self):
        assert len(QUERY_INSTRUCTION) > 0

    def test_doc_instruction_non_empty(self):
        assert len(DOC_INSTRUCTION) > 0

    def test_max_doc_chars_positive(self):
        assert MAX_DOC_CHARS > 0


# ---------------------------------------------------------------------------
# LlamaEmbedScorer
# ---------------------------------------------------------------------------

class TestLlamaEmbedScorer:
    def _make_scorer(self) -> LlamaEmbedScorer:
        with patch("httpx.get", return_value=_mock_health_ok()):
            return LlamaEmbedScorer("http://localhost:8080")

    def test_init_calls_health_check(self):
        with patch("httpx.get", return_value=_mock_health_ok()) as mock_get:
            LlamaEmbedScorer("http://localhost:8080")
            mock_get.assert_called_once()
            assert "health" in mock_get.call_args[0][0]

    def test_init_strips_trailing_slash(self):
        with patch("httpx.get", return_value=_mock_health_ok()) as mock_get:
            LlamaEmbedScorer("http://localhost:8080/")
            assert "health" in mock_get.call_args[0][0]
            assert "//" not in mock_get.call_args[0][0].replace("http://", "")

    def test_score_turns_returns_list(self):
        scorer = self._make_scorer()
        turns = [_turn(0, "content")]
        dim = 4

        # One call for query + one for docs
        emb = [1.0, 0.0, 0.0, 0.0]
        responses = [
            _mock_embed_response([emb]),  # query
            _mock_embed_response([emb]),  # doc
        ]
        with patch("httpx.post", side_effect=responses):
            result = scorer.score_turns(turns, "query", {0: 100})
        assert isinstance(result, list)
        assert len(result) == 1

    def test_score_turns_returns_scored_turns(self):
        scorer = self._make_scorer()
        turns = [_turn(0, "content")]
        emb = [1.0, 0.0, 0.0, 0.0]
        responses = [
            _mock_embed_response([emb]),
            _mock_embed_response([emb]),
        ]
        with patch("httpx.post", side_effect=responses):
            result = scorer.score_turns(turns, "query", {0: 100})
        assert result[0].turn is turns[0]

    def test_score_in_0_1_range(self):
        scorer = self._make_scorer()
        turns = [_turn(0, "content")]
        emb = [1.0, 0.0, 0.0, 0.0]  # unit vector
        responses = [
            _mock_embed_response([emb]),
            _mock_embed_response([emb]),
        ]
        with patch("httpx.post", side_effect=responses):
            result = scorer.score_turns(turns, "query", {0: 100})
        assert 0.0 <= result[0].score <= 1.0

    def test_score_uses_token_counts(self):
        scorer = self._make_scorer()
        turns = [_turn(0, "content")]
        emb = [1.0, 0.0, 0.0, 0.0]
        responses = [
            _mock_embed_response([emb]),
            _mock_embed_response([emb]),
        ]
        with patch("httpx.post", side_effect=responses):
            result = scorer.score_turns(turns, "query", {0: 42})
        assert result[0].tokens == 42

    def test_orthogonal_vectors_mid_score(self):
        """Query and doc at 90° → cos=0 → score=0.5."""
        scorer = self._make_scorer()
        turns = [_turn(0)]
        query_emb = [1.0, 0.0]
        doc_emb = [0.0, 1.0]
        responses = [
            _mock_embed_response([query_emb]),
            _mock_embed_response([doc_emb]),
        ]
        with patch("httpx.post", side_effect=responses):
            result = scorer.score_turns(turns, "query", {0: 10})
        assert abs(result[0].score - 0.5) < 1e-5

    def test_identical_vectors_max_score(self):
        """Same direction → cos=1 → score=1.0."""
        scorer = self._make_scorer()
        turns = [_turn(0)]
        emb = [1.0, 0.0]
        responses = [
            _mock_embed_response([emb]),
            _mock_embed_response([emb]),
        ]
        with patch("httpx.post", side_effect=responses):
            result = scorer.score_turns(turns, "query", {0: 10})
        assert abs(result[0].score - 1.0) < 1e-5

    def test_multiple_turns_all_scored(self):
        scorer = self._make_scorer()
        turns = [_turn(i, f"content {i}") for i in range(3)]
        q_emb = [1.0, 0.0]
        doc_embs = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
        responses = [
            _mock_embed_response([q_emb]),
            _mock_embed_response(doc_embs),
        ]
        with patch("httpx.post", side_effect=responses):
            result = scorer.score_turns(turns, "query", {0: 10, 1: 20, 2: 30})
        assert len(result) == 3

    def test_embed_sorts_by_index(self):
        """_embed should sort by index, not rely on server order."""
        scorer = self._make_scorer()
        turns = [_turn(0)]
        q_emb = [1.0, 0.0]
        # Server returns in reverse order
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "data": [
                {"index": 1, "embedding": [0.0, 1.0]},
                {"index": 0, "embedding": [1.0, 0.0]},
            ]
        }
        responses_post = [
            _mock_embed_response([q_emb]),
            mock_resp,
        ]
        with patch("httpx.post", side_effect=responses_post):
            result = scorer.score_turns(turns, "query", {0: 10})
        # index 0 embedding [1,0] → cos(q=[1,0], d=[1,0]) = 1 → score 1.0
        assert abs(result[0].score - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# LlamaRerankScorer
# ---------------------------------------------------------------------------

class TestLlamaRerankScorer:
    def _make_scorer(self) -> LlamaRerankScorer:
        with patch("httpx.get", return_value=_mock_health_ok()):
            return LlamaRerankScorer("http://localhost:8181")

    def _mock_rerank_response(self, scores: list[float]) -> MagicMock:
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {
            "results": [
                {"index": i, "relevance_score": s}
                for i, s in enumerate(scores)
            ]
        }
        return resp

    def test_init_calls_health_check(self):
        with patch("httpx.get", return_value=_mock_health_ok()) as mock_get:
            LlamaRerankScorer("http://localhost:8181")
            assert "health" in mock_get.call_args[0][0]

    def test_score_turns_returns_list(self):
        scorer = self._make_scorer()
        turns = [_turn(0, "doc content")]
        with patch("httpx.post", return_value=self._mock_rerank_response([0.9])):
            result = scorer.score_turns(turns, "query", {0: 100})
        assert isinstance(result, list)
        assert len(result) == 1

    def test_score_preserved_from_server(self):
        scorer = self._make_scorer()
        turns = [_turn(0, "doc content")]
        with patch("httpx.post", return_value=self._mock_rerank_response([0.75])):
            result = scorer.score_turns(turns, "query", {0: 100})
        assert abs(result[0].score - 0.75) < 1e-9

    def test_score_uses_token_counts(self):
        scorer = self._make_scorer()
        turns = [_turn(0)]
        with patch("httpx.post", return_value=self._mock_rerank_response([0.5])):
            result = scorer.score_turns(turns, "query", {0: 55})
        assert result[0].tokens == 55

    def test_turn_preserved_in_result(self):
        scorer = self._make_scorer()
        turns = [_turn(0, "content")]
        with patch("httpx.post", return_value=self._mock_rerank_response([0.4])):
            result = scorer.score_turns(turns, "query", {0: 10})
        assert result[0].turn is turns[0]

    def test_multiple_turns(self):
        scorer = self._make_scorer()
        turns = [_turn(i) for i in range(3)]
        scores = [0.9, 0.5, 0.1]
        with patch("httpx.post", return_value=self._mock_rerank_response(scores)):
            result = scorer.score_turns(turns, "query", {0: 10, 1: 20, 2: 30})
        assert len(result) == 3
        for i, s in enumerate(scores):
            assert abs(result[i].score - s) < 1e-9

    def test_missing_index_score_defaults_zero(self):
        """If server omits an index, score should default to 0.0."""
        scorer = self._make_scorer()
        turns = [_turn(0), _turn(1)]
        # Server only returns index 0
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"results": [{"index": 0, "relevance_score": 0.8}]}
        with patch("httpx.post", return_value=resp):
            result = scorer.score_turns(turns, "query", {0: 10, 1: 20})
        assert abs(result[0].score - 0.8) < 1e-9
        assert abs(result[1].score - 0.0) < 1e-9

    def test_query_sent_in_request_body(self):
        scorer = self._make_scorer()
        turns = [_turn(0)]
        with patch("httpx.post", return_value=self._mock_rerank_response([0.5])) as mock_post:
            scorer.score_turns(turns, "my search query", {0: 10})
        body = mock_post.call_args[1]["json"]
        assert body["query"] == "my search query"

    def test_documents_sent_in_request_body(self):
        scorer = self._make_scorer()
        turns = [_turn(0, "document text")]
        with patch("httpx.post", return_value=self._mock_rerank_response([0.5])) as mock_post:
            scorer.score_turns(turns, "query", {0: 10})
        body = mock_post.call_args[1]["json"]
        assert "document text" in body["documents"][0]
