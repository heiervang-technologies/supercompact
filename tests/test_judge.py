"""Tests for lib/eval/judge.py — ProbeAnswer, JudgeResult, scoring logic."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.eval.judge import (
    ProbeAnswer,
    JudgeResult,
    JUDGE_MODEL,
    ANSWER_MODELS,
    MAX_CONCURRENCY,
    MAX_RETRIES,
    _JUDGE_SYSTEM,
    _ANSWER_SYSTEM,
    _score_one_answer,
    _openrouter_generate_async,
)
from lib.eval.probes import Probe, ProbeSet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _probe(pid: str = "p1") -> Probe:
    return Probe(
        id=pid, dimension="error_solution", tier="factual",
        question="What was the error?", gold_answer="ValueError at line 42",
    )


def _answer(probe_id: str = "p1", score: int = -1) -> ProbeAnswer:
    return ProbeAnswer(
        probe_id=probe_id, model_key="capable",
        model_label="TestModel", answer="My answer", score=score,
    )


def _make_http_response(body: dict, status_code: int = 200) -> MagicMock:
    """Build a mock httpx response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = body
    if status_code != 200:
        import httpx
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            message=f"HTTP {status_code}", request=MagicMock(), response=resp
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


# ---------------------------------------------------------------------------
# ProbeAnswer dataclass
# ---------------------------------------------------------------------------

class TestProbeAnswer:
    def test_fields_accessible(self):
        a = ProbeAnswer(
            probe_id="p1", model_key="capable",
            model_label="GPT-4", answer="answer text",
        )
        assert a.probe_id == "p1"
        assert a.model_key == "capable"
        assert a.model_label == "GPT-4"
        assert a.answer == "answer text"

    def test_default_score_minus_one(self):
        a = ProbeAnswer(probe_id="p1", model_key="k", model_label="L", answer="A")
        assert a.score == -1

    def test_default_judge_reasoning_empty(self):
        a = ProbeAnswer(probe_id="p1", model_key="k", model_label="L", answer="A")
        assert a.judge_reasoning == ""

    def test_score_settable(self):
        a = _answer()
        a.score = 3
        assert a.score == 3

    def test_judge_reasoning_settable(self):
        a = _answer()
        a.judge_reasoning = "Great answer"
        assert a.judge_reasoning == "Great answer"


# ---------------------------------------------------------------------------
# JudgeResult dataclass
# ---------------------------------------------------------------------------

class TestJudgeResult:
    def test_fields_accessible(self):
        jr = JudgeResult(method="dedup", budget=1000)
        assert jr.method == "dedup"
        assert jr.budget == 1000

    def test_default_answers_empty(self):
        jr = JudgeResult(method="eitf", budget=2048)
        assert jr.answers == []

    def test_answers_stored(self):
        jr = JudgeResult(method="dedup", budget=1000, answers=[_answer()])
        assert len(jr.answers) == 1


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_judge_model_is_string(self):
        assert isinstance(JUDGE_MODEL, str)
        assert len(JUDGE_MODEL) > 0

    def test_answer_models_has_capable_and_cheap(self):
        assert "capable" in ANSWER_MODELS
        assert "cheap" in ANSWER_MODELS

    def test_answer_model_entries_have_model_and_label(self):
        for key, cfg in ANSWER_MODELS.items():
            assert "model" in cfg, f"Missing 'model' in ANSWER_MODELS[{key!r}]"
            assert "label" in cfg, f"Missing 'label' in ANSWER_MODELS[{key!r}]"

    def test_max_concurrency_positive(self):
        assert MAX_CONCURRENCY >= 1

    def test_max_retries_positive(self):
        assert MAX_RETRIES >= 1

    def test_judge_system_prompt_has_score_rubric(self):
        assert "0" in _JUDGE_SYSTEM
        assert "3" in _JUDGE_SYSTEM

    def test_judge_system_prompt_requests_json(self):
        assert "JSON" in _JUDGE_SYSTEM or "json" in _JUDGE_SYSTEM

    def test_answer_system_prompt_non_empty(self):
        assert len(_ANSWER_SYSTEM) > 0


# ---------------------------------------------------------------------------
# _openrouter_generate_async — no API key
# ---------------------------------------------------------------------------

class TestOpenrouterGenerateAsync:
    def test_no_api_key_raises(self):
        async def _run():
            with patch.dict("os.environ", {}, clear=True):
                # Patch out OPENROUTER_API_KEY
                import os
                os.environ.pop("OPENROUTER_API_KEY", None)
                async with __import__("httpx").AsyncClient() as client:
                    await _openrouter_generate_async(client, "m", "sys", "user")

        with patch.dict("os.environ", {"OPENROUTER_API_KEY": ""}):
            with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
                asyncio.run(_openrouter_generate_async(
                    MagicMock(), "model", "sys", "user"
                ))

    def test_success_returns_content(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "  hello world  "}}]
        }

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_resp)

        async def _run():
            with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
                with patch("asyncio.sleep", new_callable=AsyncMock):
                    return await _openrouter_generate_async(
                        mock_client, "model", "sys", "user"
                    )

        result = asyncio.run(_run())
        assert result == "hello world"

    def test_success_calls_correct_url(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "answer"}}]
        }

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_resp)

        async def _run():
            with patch.dict("os.environ", {"OPENROUTER_API_KEY": "key"}):
                with patch("asyncio.sleep", new_callable=AsyncMock):
                    return await _openrouter_generate_async(
                        mock_client, "my-model", "sys", "user"
                    )

        asyncio.run(_run())
        call_args = mock_client.post.call_args
        assert "openrouter.ai" in call_args[0][0]

    def test_model_passed_in_json_body(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "ok"}}]
        }

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_resp)

        async def _run():
            with patch.dict("os.environ", {"OPENROUTER_API_KEY": "key"}):
                with patch("asyncio.sleep", new_callable=AsyncMock):
                    return await _openrouter_generate_async(
                        mock_client, "my-model", "sys", "user"
                    )

        asyncio.run(_run())
        call_kwargs = mock_client.post.call_args[1]
        assert call_kwargs["json"]["model"] == "my-model"


# ---------------------------------------------------------------------------
# _score_one_answer — JSON parsing and score clamping
# ---------------------------------------------------------------------------

class TestScoreOneAnswer:
    def _run_score(self, raw_response: str, probe: Probe | None = None) -> ProbeAnswer:
        """Helper: run _score_one_answer with a mocked _openrouter_generate_async."""
        answer = _answer()
        sem = asyncio.Semaphore(1)
        p = probe or _probe()

        async def _run():
            with patch(
                "lib.eval.judge._openrouter_generate_async",
                new_callable=AsyncMock,
                return_value=raw_response,
            ):
                with patch.dict("os.environ", {"OPENROUTER_API_KEY": "key"}):
                    await _score_one_answer(
                        MagicMock(), sem, answer, p, JUDGE_MODEL
                    )
            return answer

        return asyncio.run(_run())

    def test_valid_json_sets_score(self):
        a = self._run_score('{"score": 2, "reasoning": "Good"}')
        assert a.score == 2

    def test_valid_json_sets_reasoning(self):
        a = self._run_score('{"score": 1, "reasoning": "Partial"}')
        assert a.judge_reasoning == "Partial"

    def test_score_clamped_above_3(self):
        a = self._run_score('{"score": 10, "reasoning": ""}')
        assert a.score == 3

    def test_score_clamped_below_0(self):
        a = self._run_score('{"score": -5, "reasoning": ""}')
        assert a.score == 0

    def test_score_3_kept(self):
        a = self._run_score('{"score": 3, "reasoning": "Perfect"}')
        assert a.score == 3

    def test_score_0_kept(self):
        a = self._run_score('{"score": 0, "reasoning": "Missing"}')
        assert a.score == 0

    def test_markdown_fenced_json_parsed(self):
        raw = '```json\n{"score": 2, "reasoning": "ok"}\n```'
        a = self._run_score(raw)
        assert a.score == 2

    def test_invalid_json_sets_score_0(self):
        a = self._run_score("not valid json at all")
        assert a.score == 0

    def test_invalid_json_sets_judge_reasoning_error(self):
        a = self._run_score("definitely not json")
        assert "Judge error" in a.judge_reasoning or a.judge_reasoning != ""

    def test_missing_score_key_defaults_0(self):
        a = self._run_score('{"reasoning": "no score key"}')
        assert a.score == 0

    def test_missing_reasoning_defaults_empty(self):
        a = self._run_score('{"score": 2}')
        assert a.judge_reasoning == ""
