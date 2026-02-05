"""Answer generation and scoring for LLM-as-Judge evaluation.

Two-step process:
1. Answer generation: Given compacted context + probe question, generate an answer.
   Run with two models (cheap + capable) to test if model capability masks context gaps.
2. Scoring: Judge each answer against the gold answer on a 0-3 rubric.

All API calls go through OpenRouter (OpenAI-compatible), which supports both
Anthropic and third-party models with a single API key.

Uses asyncio + httpx.AsyncClient for concurrent API calls (~10x faster).
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field

import httpx

from .probes import Probe, ProbeSet


# --- Model configuration ---

JUDGE_MODEL = "anthropic/claude-opus-4-5"

ANSWER_MODELS = {
    "capable": {
        "model": "anthropic/claude-opus-4-5",
        "label": "Opus-4.5",
    },
    "cheap": {
        "model": "moonshotai/kimi-k2.5",
        "label": "Kimi-K2.5",
    },
}

# Max concurrent API requests (avoid rate limits)
MAX_CONCURRENCY = 5
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0  # seconds, doubled each retry


# --- Data types ---

@dataclass
class ProbeAnswer:
    probe_id: str
    model_key: str          # "capable" or "cheap"
    model_label: str
    answer: str
    score: int = -1         # 0-3, set by judge
    judge_reasoning: str = ""


@dataclass
class JudgeResult:
    method: str
    budget: int
    answers: list[ProbeAnswer] = field(default_factory=list)


# --- Async API client (OpenRouter) ---

async def _openrouter_generate_async(
    client: httpx.AsyncClient,
    model: str,
    system: str,
    user: str,
    max_tokens: int = 1024,
) -> str:
    """Call OpenRouter's OpenAI-compatible API (async) with retries."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY env var required. Get one at https://openrouter.ai/keys"
        )

    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "max_tokens": max_tokens,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                },
                timeout=180.0,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except (httpx.HTTPStatusError, httpx.ReadTimeout, httpx.ConnectError) as e:
            last_err = e
            delay = RETRY_BASE_DELAY * (2 ** attempt)
            print(f"  [retry {attempt+1}/{MAX_RETRIES}] {type(e).__name__}: {e} — waiting {delay:.0f}s", flush=True)
            await asyncio.sleep(delay)

    raise last_err or RuntimeError("All retries exhausted")


# --- Answer generation ---

_ANSWER_SYSTEM = """\
You are a coding assistant continuing a conversation. You have been given a \
compacted summary of the conversation so far. Answer the question based ONLY \
on what you can determine from the provided context. If the context doesn't \
contain enough information, say so explicitly. Be specific and concise."""


async def _generate_one_answer(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    model: str,
    model_key: str,
    model_label: str,
    probe: Probe,
    compacted_context: str,
) -> ProbeAnswer:
    user_prompt = (
        f"<context>\n{compacted_context}\n</context>\n\n"
        f"Question: {probe.question}"
    )
    async with sem:
        try:
            text = await _openrouter_generate_async(
                client, model=model, system=_ANSWER_SYSTEM, user=user_prompt,
            )
        except Exception as e:
            text = f"[ERROR: {e}]"

    return ProbeAnswer(
        probe_id=probe.id,
        model_key=model_key,
        model_label=model_label,
        answer=text,
    )


async def _generate_answers_async(
    compacted_context: str,
    probe_set: ProbeSet,
    model_configs: dict[str, dict] | None = None,
) -> list[ProbeAnswer]:
    configs = model_configs or ANSWER_MODELS
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async with httpx.AsyncClient() as client:
        tasks = []
        for model_key, cfg in configs.items():
            for probe in probe_set.probes:
                tasks.append(_generate_one_answer(
                    client, sem, cfg["model"], model_key, cfg["label"],
                    probe, compacted_context,
                ))

        answers = await asyncio.gather(*tasks)
    return list(answers)


def generate_answers(
    compacted_context: str,
    probe_set: ProbeSet,
    model_configs: dict[str, dict] | None = None,
) -> list[ProbeAnswer]:
    """Generate answers for all probes using each configured model (concurrent)."""
    return asyncio.run(_generate_answers_async(
        compacted_context, probe_set, model_configs,
    ))


# --- Scoring ---

_JUDGE_SYSTEM = """\
You are a strict evaluator for context compaction quality. You will score an \
answer against a reference (gold) answer.

Score on a 0-3 scale:
- 3 (Complete): All key facts present, correct reasoning, actionable
- 2 (Partial): Core understanding present but missing details or nuance
- 1 (Minimal): Touches on topic but misses the key insight
- 0 (Missing): Information absent or answer is wrong

Respond with ONLY a JSON object: {"score": N, "reasoning": "brief explanation"}
No other text."""


async def _score_one_answer(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    answer: ProbeAnswer,
    probe: Probe,
    judge: str,
) -> None:
    """Score a single answer in-place."""
    user_prompt = (
        f"Question: {probe.question}\n\n"
        f"Gold answer: {probe.gold_answer}\n\n"
        f"Candidate answer: {answer.answer}"
    )

    async with sem:
        try:
            raw = await _openrouter_generate_async(
                client, model=judge, system=_JUDGE_SYSTEM,
                user=user_prompt, max_tokens=256,
            )
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1]
                if raw.endswith("```"):
                    raw = raw[:-3]
                raw = raw.strip()
            result = json.loads(raw)
            answer.score = max(0, min(3, int(result.get("score", 0))))
            answer.judge_reasoning = result.get("reasoning", "")
        except Exception as e:
            answer.score = 0
            answer.judge_reasoning = f"Judge error: {e}"


async def _score_answers_async(
    answers: list[ProbeAnswer],
    probe_set: ProbeSet,
    judge_model: str | None = None,
) -> list[ProbeAnswer]:
    judge = judge_model or JUDGE_MODEL
    probe_map = {p.id: p for p in probe_set.probes}
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async with httpx.AsyncClient() as client:
        tasks = []
        for answer in answers:
            probe = probe_map.get(answer.probe_id)
            if not probe:
                answer.score = 0
                answer.judge_reasoning = "Probe not found"
                continue
            tasks.append(_score_one_answer(client, sem, answer, probe, judge))

        await asyncio.gather(*tasks)
    return answers


def score_answers(
    answers: list[ProbeAnswer],
    probe_set: ProbeSet,
    judge_model: str | None = None,
) -> list[ProbeAnswer]:
    """Score each answer against its gold answer using the judge model (concurrent).

    Mutates the answers in-place (sets score and judge_reasoning).
    Returns the same list for convenience.
    """
    return asyncio.run(_score_answers_async(answers, probe_set, judge_model))
