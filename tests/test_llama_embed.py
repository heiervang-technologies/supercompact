"""Tests for lib/llama_embed.py — constants, _instruct, LlamaEmbedScorer (mocked)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.llama_embed import (
    DOC_INSTRUCTION,
    MAX_DOC_CHARS,
    QUERY_INSTRUCTION,
    LlamaEmbedScorer,
    _instruct,
)
from lib.parser import Turn
from lib.types import ScoredTurn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _turn(kind: str, index: int, text: str) -> Turn:
    t = Turn(kind=kind, index=index)
    t.append({"message": {"content": text}})
    return t


def _mock_health():
    """Return a mock for httpx.get that simulates a healthy server."""
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    return resp


def _mock_embed_response(embeddings: list[list[float]]):
    """Return a mock httpx.post response for /v1/embeddings."""
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
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_query_instruction_is_str(self):
        assert isinstance(QUERY_INSTRUCTION, str)

    def test_query_instruction_nonempty(self):
        assert len(QUERY_INSTRUCTION) > 0

    def test_doc_instruction_is_str(self):
        assert isinstance(DOC_INSTRUCTION, str)

    def test_doc_instruction_nonempty(self):
        assert len(DOC_INSTRUCTION) > 0

    def test_max_doc_chars_is_int(self):
        assert isinstance(MAX_DOC_CHARS, int)

    def test_max_doc_chars_positive(self):
        assert MAX_DOC_CHARS > 0


# ---------------------------------------------------------------------------
# _instruct
# ---------------------------------------------------------------------------

class TestInstruct:
    def test_basic_format(self):
        result = _instruct("do task", "hello")
        assert result == "Instruct: do task\nQuery: hello"

    def test_instruction_in_result(self):
        result = _instruct("my instruction", "query text")
        assert "Instruct: my instruction" in result

    def test_query_in_result(self):
        result = _instruct("inst", "my query")
        assert "Query: my query" in result

    def test_newline_separator(self):
        result = _instruct("a", "b")
        lines = result.split("\n")
        assert len(lines) == 2

    def test_empty_inputs(self):
        result = _instruct("", "")
        assert result == "Instruct: \nQuery: "

    def test_returns_string(self):
        assert isinstance(_instruct("x", "y"), str)


# ---------------------------------------------------------------------------
# LlamaEmbedScorer — init with mocked health check
# ---------------------------------------------------------------------------

class TestLlamaEmbedScorerInit:
    def test_url_constructed_from_base(self):
        with patch("lib.llama_embed.httpx.get", return_value=_mock_health()):
            scorer = LlamaEmbedScorer("http://localhost:8080")
        assert scorer.url == "http://localhost:8080/v1/embeddings"

    def test_trailing_slash_stripped(self):
        with patch("lib.llama_embed.httpx.get", return_value=_mock_health()):
            scorer = LlamaEmbedScorer("http://localhost:8080/")
        assert scorer.url == "http://localhost:8080/v1/embeddings"

    def test_health_check_called(self):
        with patch("lib.llama_embed.httpx.get", return_value=_mock_health()) as mock_get:
            LlamaEmbedScorer("http://localhost:8080")
        mock_get.assert_called_once()
        call_url = mock_get.call_args[0][0]
        assert "health" in call_url

    def test_health_failure_propagates(self):
        bad_resp = MagicMock()
        bad_resp.raise_for_status.side_effect = Exception("Connection refused")
        with patch("lib.llama_embed.httpx.get", return_value=bad_resp):
            with pytest.raises(Exception, match="Connection refused"):
                LlamaEmbedScorer("http://localhost:8080")


# ---------------------------------------------------------------------------
# LlamaEmbedScorer.score_turns — mocked _embed
# ---------------------------------------------------------------------------

class TestLlamaEmbedScorerScoreTurns:
    def _make_scorer(self) -> LlamaEmbedScorer:
        with patch("lib.llama_embed.httpx.get", return_value=_mock_health()):
            return LlamaEmbedScorer("http://localhost:8080")

    def test_single_turn_returns_one_result(self):
        scorer = self._make_scorer()
        t = _turn("system", 0, "single content")
        token_counts = {0: 100}
        query_emb = np.array([[1.0, 0.0, 0.0, 0.0]])
        doc_embs = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

        call_count = [0]
        def fake_embed(texts):
            idx = call_count[0]
            call_count[0] += 1
            return query_emb if idx == 0 else doc_embs

        with patch.object(scorer, "_embed", side_effect=fake_embed):
            result = scorer.score_turns([t], "query", token_counts)
        assert len(result) == 1

    def test_returns_one_scored_turn_per_system_turn(self):
        scorer = self._make_scorer()
        dim = 4
        turns = [_turn("system", i, f"content{i}") for i in range(3)]
        token_counts = {i: 50 for i in range(3)}
        # Query emb + 3 doc embs
        query_emb = np.array([[1.0, 0.0, 0.0, 0.0]])
        doc_embs = np.eye(3, 4, dtype=np.float32)

        call_count = [0]
        def fake_embed(texts):
            idx = call_count[0]
            call_count[0] += 1
            if idx == 0:
                return query_emb
            return doc_embs

        with patch.object(scorer, "_embed", side_effect=fake_embed):
            result = scorer.score_turns(turns, "query", token_counts)
        assert len(result) == 3

    def test_scores_are_in_zero_one_range(self):
        scorer = self._make_scorer()
        turns = [_turn("system", 0, "hello"), _turn("system", 1, "world")]
        token_counts = {0: 30, 1: 40}
        dim = 4
        query_emb = np.array([[1.0, 0.0, 0.0, 0.0]])
        doc_embs = np.array([[0.5, 0.5, 0.5, 0.5], [-0.5, 0.5, 0.5, 0.5]], dtype=np.float32)

        call_count = [0]
        def fake_embed(texts):
            idx = call_count[0]
            call_count[0] += 1
            return query_emb if idx == 0 else doc_embs

        with patch.object(scorer, "_embed", side_effect=fake_embed):
            result = scorer.score_turns(turns, "query", token_counts)
        for st in result:
            assert 0.0 <= st.score <= 1.0

    def test_tokens_set_from_token_counts(self):
        scorer = self._make_scorer()
        t = _turn("system", 7, "text")
        query_emb = np.array([[1.0, 0.0]])
        doc_embs = np.array([[1.0, 0.0]], dtype=np.float32)

        call_count = [0]
        def fake_embed(texts):
            idx = call_count[0]
            call_count[0] += 1
            return query_emb if idx == 0 else doc_embs

        with patch.object(scorer, "_embed", side_effect=fake_embed):
            result = scorer.score_turns([t], "query", {7: 999})
        assert result[0].tokens == 999

    def test_returns_scored_turn_instances(self):
        scorer = self._make_scorer()
        t = _turn("system", 0, "content")
        query_emb = np.array([[1.0, 0.0]])
        doc_embs = np.array([[0.0, 1.0]], dtype=np.float32)

        call_count = [0]
        def fake_embed(texts):
            idx = call_count[0]
            call_count[0] += 1
            return query_emb if idx == 0 else doc_embs

        with patch.object(scorer, "_embed", side_effect=fake_embed):
            result = scorer.score_turns([t], "query", {0: 50})
        assert all(isinstance(st, ScoredTurn) for st in result)
