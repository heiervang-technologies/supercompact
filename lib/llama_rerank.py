"""Reranker scorer using a llama.cpp server (Qwen3-Reranker-0.6B-Q8_0-GGUF).

Calls POST /v1/rerank on an external llama-server. The server returns
relevance scores directly — no client-side similarity math needed.

IMPORTANT: Use the ggml-org GGUF conversion for the reranker. Other
conversions are known to produce broken scores.
  Model: ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF
"""

from __future__ import annotations

import httpx

from .parser import Turn, extract_text
from .types import ScoredTurn

MAX_DOC_CHARS = 2048


class LlamaRerankScorer:
    """Scores turns via a llama.cpp reranking server."""

    def __init__(self, base_url: str = "http://localhost:8181"):
        self.url = base_url.rstrip("/") + "/v1/rerank"
        httpx.get(base_url.rstrip("/") + "/health", timeout=5).raise_for_status()

    def score_turns(
        self,
        system_turns: list[Turn],
        query: str,
        token_counts: dict[int, int],
        batch_size: int = 64,
    ) -> list[ScoredTurn]:
        documents = [extract_text(t)[:MAX_DOC_CHARS] for t in system_turns]

        # Collect all scores, batching to avoid overloading the server context
        index_score: dict[int, float] = {}

        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i : i + batch_size]
            resp = httpx.post(
                self.url,
                json={
                    "model": "qwen3",
                    "query": query,
                    "documents": batch_docs,
                },
                timeout=120,
            )
            resp.raise_for_status()
            for item in resp.json()["results"]:
                # item["index"] is relative to this batch
                global_idx = i + item["index"]
                index_score[global_idx] = item["relevance_score"]

            done = min(i + batch_size, len(documents))
            print(f"  reranked {done}/{len(documents)}", flush=True)

        results = []
        for idx, turn in enumerate(system_turns):
            results.append(ScoredTurn(
                turn=turn,
                score=index_score.get(idx, 0.0),
                tokens=token_counts.get(turn.index, 0),
            ))
        return results
