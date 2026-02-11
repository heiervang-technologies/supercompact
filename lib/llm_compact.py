"""Claude Code /compact baseline via LLM summarization.

Simulates Claude Code's /compact by sending the prefix conversation to Claude
via OpenRouter and asking it to produce a summary that preserves key entities.
The summary is then wrapped in a synthetic turn for entity coverage evaluation.
"""

from __future__ import annotations

import os
import httpx

from .parser import Turn, extract_text
from .tokenizer import estimate_tokens


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "anthropic/claude-sonnet-4"

COMPACT_SYSTEM_PROMPT = """\
You are a conversation compactor. Your job is to summarize a coding agent conversation
while preserving ALL technical details that would be needed to continue the work.

You MUST preserve:
- Every file path mentioned (exact paths like /home/user/project/src/foo.py)
- Every error message and exception name (e.g. ModuleNotFoundError, TypeError)
- Every port number (e.g. :8080, :3000, port 5432)
- Every URL (e.g. http://localhost:8080/api/v1/users)
- Every function and method name (e.g. parse_jsonl, build_query)
- Every class name (e.g. ScoredTurn, EntitySet)
- Every shell command (e.g. git commit, docker build, npm install)
- Every package name mentioned (e.g. httpx, transformers, rich)
- Every environment variable (e.g. OPENROUTER_API_KEY, CUDA_VISIBLE_DEVICES)
- Every HTTP status code in error context (e.g. 401 Unauthorized, 404 Not Found)
- Every configuration value, version number, and technical identifier

Format the summary as a structured document with sections. Include exact values inline.
Do NOT paraphrase technical identifiers — copy them exactly as they appear.
The summary should be comprehensive enough that an agent reading only this summary
could continue the work without access to the original conversation."""


def llm_compact(
    prefix_turns: list[Turn],
    budget: int,
) -> str:
    """Summarize prefix turns using Claude via OpenRouter.

    Args:
        prefix_turns: Conversation turns to summarize.
        budget: Target token budget for the summary.

    Returns:
        Summary text preserving key entities.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

    # Build conversation text from prefix turns
    conversation_text = "\n\n---\n\n".join(
        f"[{t.kind.upper()} turn {t.index}]\n{extract_text(t)}"
        for t in prefix_turns
    )

    # Truncate if too long for the API context (leave room for system prompt + response)
    # Claude Sonnet has 200k context; we'll cap input at ~150k tokens worth of chars
    max_input_chars = 600_000  # ~150k tokens at ~4 chars/token
    if len(conversation_text) > max_input_chars:
        conversation_text = conversation_text[:max_input_chars]

    # Target summary length based on budget
    target_tokens = min(budget, 16_000)  # Cap at 16k tokens for summary

    user_prompt = (
        f"Summarize the following coding agent conversation. "
        f"Target approximately {target_tokens:,} tokens for the summary. "
        f"Preserve ALL technical details (file paths, error messages, ports, "
        f"URLs, function names, class names, commands, packages, env vars, "
        f"HTTP status codes).\n\n"
        f"CONVERSATION:\n{conversation_text}"
    )

    response = httpx.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": COMPACT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": target_tokens,
            "temperature": 0.0,
        },
        timeout=300.0,
    )
    if response.status_code != 200:
        try:
            err_body = response.json()
        except Exception:
            err_body = response.text
        raise RuntimeError(
            f"OpenRouter API error {response.status_code}: {err_body}"
        )

    data = response.json()
    summary = data["choices"][0]["message"]["content"]
    return summary


def make_synthetic_turn(summary_text: str, index: int = 0) -> Turn:
    """Wrap summary text in a synthetic Turn for entity extraction.

    Creates a system turn containing the summary as a text content block,
    compatible with extract_text() and entity extraction.
    """
    turn = Turn(kind="system", index=index)
    turn.lines.append({
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": summary_text,
        },
    })
    return turn
