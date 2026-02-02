"""Qwen3-Embedding-0.6B scoring for system turns.

Embeds a query (recent user messages) and all system turn documents,
then ranks by cosine similarity. Much faster than the reranker variant
since all documents can be encoded in a single batched forward pass.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import logging

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel, AutoTokenizer

# Suppress noisy tqdm/progress bars from transformers model loading
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

from .parser import Turn, extract_text

MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
MAX_TOKENS = 8192
# Shorter context for batched encoding — keeps GPU memory bounded.
# 512 tokens is enough to capture the gist of a turn for similarity scoring.
ENCODE_MAX_LENGTH = 512

QUERY_INSTRUCTION = (
    "Find assistant responses from an AI coding conversation that contain "
    "information needed to continue the current task: code changes, decisions, "
    "errors, file paths, architectural context, or unfinished work."
)

DOC_INSTRUCTION = (
    "AI coding assistant response from a conversation history"
)


@dataclass
class ScoredTurn:
    """A turn with its relevance score."""

    turn: Turn
    score: float
    tokens: int


def build_query(user_turns: list[Turn], max_chars: int = 4000) -> str:
    """Build a query from the last 2-3 user messages."""
    recent = user_turns[-3:] if len(user_turns) >= 3 else user_turns
    parts = [extract_text(t) for t in recent]
    query = "\n---\n".join(parts)
    if len(query) > max_chars:
        query = query[-max_chars:]
    return query


def _last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Pool the last non-padding token's hidden state as the embedding."""
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]


def _format_instruct(instruction: str, text: str) -> str:
    """Wrap text with Qwen3 instruct tags."""
    return f"Instruct: {instruction}\nQuery: {text}"


class Scorer:
    """Scores system turns using Qwen3-Embedding-0.6B cosine similarity."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
        dtype = torch.float32 if device == "cpu" else torch.float16
        self.model = AutoModel.from_pretrained(
            MODEL_ID,
            dtype=dtype,
        ).to(device).eval()

    def _encode(self, texts: list[str], max_length: int = ENCODE_MAX_LENGTH) -> Tensor:
        """Encode texts into normalized embeddings."""
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**encoded)
            embeddings = _last_token_pool(
                outputs.last_hidden_state, encoded["attention_mask"]
            )

        return F.normalize(embeddings, p=2, dim=1)

    def score_turns(
        self,
        system_turns: list[Turn],
        query: str,
        token_counts: dict[int, int],
        batch_size: int = 16,
    ) -> list[ScoredTurn]:
        """Score system turns by cosine similarity to the query embedding.

        Encodes query with instruction, documents with doc instruction,
        then computes cosine similarity.
        """
        # Encode query with instruction
        query_text = _format_instruct(QUERY_INSTRUCTION, query)
        query_emb = self._encode([query_text])  # (1, dim)

        # Encode documents in batches
        doc_embeddings: list[Tensor] = []
        for batch_start in range(0, len(system_turns), batch_size):
            batch = system_turns[batch_start : batch_start + batch_size]
            doc_texts = [
                _format_instruct(DOC_INSTRUCTION, extract_text(t))
                for t in batch
            ]
            embs = self._encode(doc_texts)
            doc_embeddings.append(embs)

            done = min(batch_start + batch_size, len(system_turns))
            print(f"  encoded {done}/{len(system_turns)}", flush=True)

        all_doc_emb = torch.cat(doc_embeddings, dim=0)  # (N, dim)

        # Cosine similarities (already normalized, so just dot product)
        sims = (query_emb @ all_doc_emb.T).squeeze(0).cpu().tolist()

        # Convert to 0-1 range (cosine sim is in [-1, 1])
        results = []
        for turn, sim in zip(system_turns, sims):
            score = (sim + 1.0) / 2.0  # map [-1,1] -> [0,1]
            results.append(ScoredTurn(
                turn=turn,
                score=score,
                tokens=token_counts.get(turn.index, 0),
            ))

        return results


def random_scores(
    system_turns: list[Turn],
    token_counts: dict[int, int],
) -> list[ScoredTurn]:
    """Generate random scores for dry-run testing."""
    return [
        ScoredTurn(
            turn=turn,
            score=random.random(),
            tokens=token_counts.get(turn.index, 0),
        )
        for turn in system_turns
    ]
