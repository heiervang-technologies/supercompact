"""Tests for lib/eval/judge.py — ProbeAnswer, JudgeResult, and scoring helpers."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.eval.judge import (
    ANSWER_MODELS,
    JUDGE_MODEL,
    JudgeResult,
    ProbeAnswer,
    _openrouter_generate_async,
    _score_one_answer,
    generate_answers,
    score_answers,
)
from lib.eval.probes import Probe, ProbeSet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_probe(id="p1", question="Q?", gold_answer="A") -> Probe:
    return Probe(
        id=id,
        dimension="progress",
        tier="factual",
        question=question,
        gold_answer=gold_answer,
    )


def _make_answer(probe_id="p1", score=-1) -> ProbeAnswer:
    return ProbeAnswer(
        probe_id=probe_id,
        model_key="cheap",
        model_label="haiku",
        answer="some answer",
        score=score,
    )


async def _run_score_one(raw_response: str) -> ProbeAnswer:
    """Run _score_one_answer with a mocked _openrouter_generate_async."""
    answer = _make_answer()
    probe = _make_probe()

    async def _mock_gen(*args, **kwargs):
        return raw_response

    sem = asyncio.Semaphore(1)
    client = MagicMock()

    with patch("lib.eval.judge._openrouter_generate_async", side_effect=_mock_gen):
        await _score_one_answer(client, sem, answer, probe, "judge-model")

    return answer


# ---------------------------------------------------------------------------
# ProbeAnswer dataclass
# ---------------------------------------------------------------------------

class TestProbeAnswerDataclass:
    def test_default_score_is_minus_one(self):
        a = ProbeAnswer(probe_id="p1", model_key="cheap", model_label="haiku", answer="x")
        assert a.score == -1

    def test_default_judge_reasoning_is_empty(self):
        a = ProbeAnswer(probe_id="p1", model_key="cheap", model_label="haiku", answer="x")
        assert a.judge_reasoning == ""

    def test_fields_stored(self):
        a = ProbeAnswer(probe_id="myid", model_key="capable", model_label="gpt4", answer="hello")
        assert a.probe_id == "myid"
        assert a.model_key == "capable"
        assert a.model_label == "gpt4"
        assert a.answer == "hello"

    def test_custom_score_and_reasoning(self):
        a = ProbeAnswer(
            probe_id="p2",
            model_key="cheap",
            model_label="haiku",
            answer="blah",
            score=3,
            judge_reasoning="perfect answer",
        )
        assert a.score == 3
        assert a.judge_reasoning == "perfect answer"


# ---------------------------------------------------------------------------
# JudgeResult dataclass
# ---------------------------------------------------------------------------

class TestJudgeResultDataclass:
    def test_method_and_budget_stored(self):
        jr = JudgeResult(method="dedup", budget=80_000)
        assert jr.method == "dedup"
        assert jr.budget == 80_000

    def test_default_answers_is_empty_list(self):
        jr = JudgeResult(method="eitf", budget=40_000)
        assert jr.answers == []

    def test_answers_mutable(self):
        jr = JudgeResult(method="dedup", budget=80_000)
        jr.answers.append(_make_answer())
        assert len(jr.answers) == 1


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_judge_model_is_string(self):
        assert isinstance(JUDGE_MODEL, str)
        assert len(JUDGE_MODEL) > 0

    def test_answer_models_has_capable_and_cheap(self):
        assert "capable" in ANSWER_MODELS
        assert "cheap" in ANSWER_MODELS

    def test_answer_model_entries_have_model_key(self):
        for key, cfg in ANSWER_MODELS.items():
            assert "model" in cfg
            assert "label" in cfg


# ---------------------------------------------------------------------------
# generate_answers — no probes
# ---------------------------------------------------------------------------

class TestGenerateAnswersEmpty:
    def test_no_probes_returns_empty_list(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        probe_set = ProbeSet(probes=[])
        result = generate_answers("some context", probe_set)
        assert result == []

    def test_no_probes_resets_counter(self, monkeypatch):
        import lib.eval.judge as judge_mod
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        probe_set = ProbeSet(probes=[])
        judge_mod._answer_counter = 99
        generate_answers("ctx", probe_set)
        assert judge_mod._answer_counter == 0


# ---------------------------------------------------------------------------
# score_answers — no-API paths
# ---------------------------------------------------------------------------

class TestScoreAnswersEmpty:
    def test_no_answers_returns_same_list(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        probe_set = ProbeSet(probes=[_make_probe()])
        result = score_answers([], probe_set)
        assert result == []


class TestScoreAnswersMissingProbe:
    def test_missing_probe_sets_score_zero(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        probe_set = ProbeSet(probes=[])  # no probes registered
        answer = _make_answer(probe_id="ghost_probe")
        score_answers([answer], probe_set)
        assert answer.score == 0

    def test_missing_probe_sets_reasoning(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        probe_set = ProbeSet(probes=[])
        answer = _make_answer(probe_id="ghost_probe")
        score_answers([answer], probe_set)
        assert "Probe not found" in answer.judge_reasoning


# ---------------------------------------------------------------------------
# _openrouter_generate_async — no API key
# ---------------------------------------------------------------------------

class TestOpenrouterMissingKey:
    def test_missing_key_raises_runtime_error(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        import httpx

        async def run():
            async with httpx.AsyncClient() as client:
                return await _openrouter_generate_async(client, "model", "sys", "user")

        with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
            asyncio.run(run())


# ---------------------------------------------------------------------------
# _score_one_answer — JSON parsing, fence stripping, clamping
# ---------------------------------------------------------------------------

class TestScoreOneAnswer:
    def test_valid_json_sets_score(self):
        answer = asyncio.run(_run_score_one('{"score": 2, "reasoning": "partial"}'))
        assert answer.score == 2

    def test_valid_json_sets_reasoning(self):
        answer = asyncio.run(_run_score_one('{"score": 3, "reasoning": "complete match"}'))
        assert answer.judge_reasoning == "complete match"

    def test_markdown_fence_stripped(self):
        fenced = "```json\n{\"score\": 1, \"reasoning\": \"ok\"}\n```"
        answer = asyncio.run(_run_score_one(fenced))
        assert answer.score == 1

    def test_score_clamped_above_three(self):
        answer = asyncio.run(_run_score_one('{"score": 9, "reasoning": "too high"}'))
        assert answer.score == 3

    def test_score_clamped_below_zero(self):
        answer = asyncio.run(_run_score_one('{"score": -5, "reasoning": "too low"}'))
        assert answer.score == 0

    def test_bad_json_sets_score_zero(self):
        answer = asyncio.run(_run_score_one("not valid json at all"))
        assert answer.score == 0

    def test_bad_json_sets_judge_error_reasoning(self):
        answer = asyncio.run(_run_score_one("not valid json at all"))
        assert "Judge error" in answer.judge_reasoning
