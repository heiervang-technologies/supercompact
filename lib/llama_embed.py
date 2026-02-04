"""Embedding scorer using a llama.cpp server (Qwen3-Embedding-0.6B-GGUF).

Calls POST /v1/embeddings on an external llama-server, computes cosine
similarity client-side with numpy. Drop-in replacement for the PyTorch
scorer — same interface, same output type.
"""

from __future__ import annotations

import numpy as np
import httpx

from .parser import Turn, extract_text
from .types import ScoredTurn

QUERY_INSTRUCTION = (
    "Find assistant responses from an AI coding conversation that contain "
    "information needed to continue the current task: code changes, decisions, "
    "errors, file paths, architectural context, or unfinished work."
)

DOC_INSTRUCTION = (
    "AI coding assistant response from a conversation history"
)

# Truncate document text before sending — mirrors the 512-token cap in the
# PyTorch scorer.  Characters are a rough proxy; the server tokenizer handles
# the real truncation.
MAX_DOC_CHARS = 2048


def _instruct(instruction: str, text: str) -> str:
    return f"Instruct: {instruction}\nQuery: {text}"


class LlamaEmbedScorer:
    """Scores turns via cosine similarity using a llama.cpp embedding server."""

    def __init__(self, base_url: str = "http://localhost:8080"):
        self.url = base_url.rstrip("/") + "/v1/embeddings"
        # Quick health check
        httpx.get(base_url.rstrip("/") + "/health", timeout=5).raise_for_status()

    def _embed(self, texts: list[str]) -> np.ndarray:
        resp = httpx.post(
            self.url,
            json={"input": texts, "model": "qwen3"},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        # Sort by index to guarantee order
        data.sort(key=lambda d: d["index"])
        return np.array([d["embedding"] for d in data], dtype=np.float32)

    def score_turns(
        self,
        system_turns: list[Turn],
        query: str,
        token_counts: dict[int, int],
        batch_size: int = 32,
    ) -> list[ScoredTurn]:
        # Encode query
        query_text = _instruct(QUERY_INSTRUCTION, query)
        query_emb = self._embed([query_text])  # (1, dim)
        query_emb /= np.linalg.norm(query_emb, axis=1, keepdims=True)

        # Encode documents in batches
        doc_embs: list[np.ndarray] = []
        for i in range(0, len(system_turns), batch_size):
            batch = system_turns[i : i + batch_size]
            texts = [
                _instruct(DOC_INSTRUCTION, extract_text(t)[:MAX_DOC_CHARS])
                for t in batch
            ]
            emb = self._embed(texts)
            emb /= np.linalg.norm(emb, axis=1, keepdims=True)
            doc_embs.append(emb)

            done = min(i + batch_size, len(system_turns))
            print(f"  encoded {done}/{len(system_turns)}", flush=True)

        all_doc = np.concatenate(doc_embs, axis=0)  # (N, dim)

        # Cosine similarity
        sims = (query_emb @ all_doc.T).squeeze(0)  # (N,)

        results = []
        for turn, sim in zip(system_turns, sims):
            score = float((sim + 1.0) / 2.0)
            results.append(ScoredTurn(
                turn=turn,
                score=score,
                tokens=token_counts.get(turn.index, 0),
            ))
        return results
