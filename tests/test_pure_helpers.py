"""Pure-function tests for supercompact lib helpers.

Covers:
  lib/parser.py   — _is_user_message, extract_text, Turn
  lib/fitness.py  — _extract_vocab, _idf, FitnessResult.f1
  lib/scorer.py   — _format_instruct, _last_token_pool
  lib/types.py    — build_query
  lib/dedup.py    — SuffixAutomaton

All pure functions — no file system, network, or GPU access.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest
import torch

# Ensure lib is importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.parser import Turn, _is_user_message, extract_text
from lib.fitness import FitnessResult, _extract_vocab, _idf
from lib.scorer import _format_instruct, _last_token_pool
from lib.types import build_query, ScoredTurn
from lib.dedup import SuffixAutomaton


# ---------------------------------------------------------------------------
# Turn dataclass
# ---------------------------------------------------------------------------

class TestTurn:
    def test_default_lines_empty(self):
        t = Turn(kind="user")
        assert t.lines == []

    def test_append_adds_record(self):
        t = Turn(kind="system")
        t.append({"type": "assistant"})
        assert len(t.lines) == 1

    def test_append_multiple(self):
        t = Turn(kind="user")
        t.append({"a": 1})
        t.append({"b": 2})
        assert len(t.lines) == 2

    def test_kind_stored(self):
        t = Turn(kind="system", index=3)
        assert t.kind == "system"
        assert t.index == 3


# ---------------------------------------------------------------------------
# _is_user_message
# ---------------------------------------------------------------------------

class TestIsUserMessage:
    def test_plain_string_content(self):
        record = {"type": "user", "message": {"content": "hello"}}
        assert _is_user_message(record) is True

    def test_wrong_type_returns_false(self):
        record = {"type": "assistant", "message": {"content": "hi"}}
        assert _is_user_message(record) is False

    def test_tool_result_block_returns_false(self):
        record = {
            "type": "user",
            "message": {
                "content": [{"type": "tool_result", "content": "output"}]
            },
        }
        assert _is_user_message(record) is False

    def test_text_block_returns_true(self):
        record = {
            "type": "user",
            "message": {
                "content": [{"type": "text", "text": "hi there"}]
            },
        }
        assert _is_user_message(record) is True

    def test_source_tool_assistant_uuid_returns_false(self):
        record = {
            "type": "user",
            "sourceToolAssistantUUID": "abc-123",
            "message": {"content": "injected"},
        }
        assert _is_user_message(record) is False

    def test_missing_type_returns_false(self):
        assert _is_user_message({}) is False

    def test_empty_content_list_returns_true(self):
        # An empty list has no tool_result blocks
        record = {"type": "user", "message": {"content": []}}
        assert _is_user_message(record) is True

    def test_mixed_blocks_without_tool_result_returns_true(self):
        record = {
            "type": "user",
            "message": {
                "content": [
                    {"type": "text", "text": "question"},
                    {"type": "image", "source": {}},
                ]
            },
        }
        assert _is_user_message(record) is True


# ---------------------------------------------------------------------------
# extract_text
# ---------------------------------------------------------------------------

def _make_turn(records: list[dict]) -> Turn:
    t = Turn(kind="system")
    for r in records:
        t.append(r)
    return t


class TestExtractText:
    def test_string_content(self):
        t = _make_turn([{"message": {"content": "hello world"}}])
        assert "hello world" in extract_text(t)

    def test_text_block(self):
        t = _make_turn([
            {"message": {"content": [{"type": "text", "text": "foo bar"}]}}
        ])
        assert "foo bar" in extract_text(t)

    def test_thinking_block(self):
        t = _make_turn([
            {"message": {"content": [{"type": "thinking", "thinking": "I think..."}]}}
        ])
        assert "I think..." in extract_text(t)

    def test_tool_use_block(self):
        t = _make_turn([
            {
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Bash",
                            "input": {"command": "ls -la"},
                        }
                    ]
                }
            }
        ])
        result = extract_text(t)
        assert "tool_use: Bash" in result
        assert "command" in result
        assert "ls -la" in result

    def test_tool_result_string_content(self):
        t = _make_turn([
            {
                "message": {
                    "content": [
                        {"type": "tool_result", "content": "output text"}
                    ]
                }
            }
        ])
        assert "output text" in extract_text(t)

    def test_tool_result_list_content(self):
        t = _make_turn([
            {
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "content": [{"type": "text", "text": "nested text"}],
                        }
                    ]
                }
            }
        ])
        assert "nested text" in extract_text(t)

    def test_empty_turn(self):
        t = Turn(kind="system")
        assert extract_text(t) == ""

    def test_multiple_records_joined(self):
        t = _make_turn([
            {"message": {"content": "first"}},
            {"message": {"content": "second"}},
        ])
        result = extract_text(t)
        assert "first" in result
        assert "second" in result

    def test_long_tool_input_truncated(self):
        long_val = "x" * 600
        t = _make_turn([
            {
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Write",
                            "input": {"content": long_val},
                        }
                    ]
                }
            }
        ])
        result = extract_text(t)
        # Value is truncated to 500 chars + "..."
        assert "..." in result
        assert len(result) < 600


# ---------------------------------------------------------------------------
# _extract_vocab
# ---------------------------------------------------------------------------

class TestExtractVocab:
    def test_basic_word(self):
        vocab = _extract_vocab("hello world")
        # 'hello' and 'world' are 5 chars — both >=4 so both included
        assert "hello" in vocab
        assert "world" in vocab

    def test_short_words_excluded(self):
        vocab = _extract_vocab("a the is it")
        assert len(vocab) == 0

    def test_lowercased(self):
        vocab = _extract_vocab("FooBar")
        # "FooBar" matches _WORD_RE since len>=4, stored lowercased
        assert "foobar" in vocab

    def test_file_path_extracted(self):
        vocab = _extract_vocab("/home/user/projects/myfile.py")
        path_keys = [k for k in vocab if "/" in k]
        assert len(path_keys) > 0

    def test_returns_counter(self):
        result = _extract_vocab("python python")
        assert isinstance(result, Counter)

    def test_empty_string(self):
        assert _extract_vocab("") == Counter()

    def test_word_count_correct(self):
        vocab = _extract_vocab("pytest pytest pytest")
        assert vocab["pytest"] == 3


# ---------------------------------------------------------------------------
# _idf
# ---------------------------------------------------------------------------

class TestIdf:
    def test_term_in_all_docs(self):
        docs = [Counter({"word": 1}), Counter({"word": 2})]
        result = _idf("word", docs, 2)
        # df=2, total=2 → log(1 + 2/2) = log(2)
        import math
        assert abs(result - math.log(2)) < 1e-9

    def test_term_in_no_docs(self):
        docs = [Counter({"other": 1})]
        assert _idf("missing", docs, 1) == 0.0

    def test_term_in_one_of_two_docs(self):
        import math
        docs = [Counter({"rare": 1}), Counter({"other": 1})]
        result = _idf("rare", docs, 2)
        # df=1, total=2 → log(1 + 2) = log(3)
        assert abs(result - math.log(3)) < 1e-9

    def test_idf_increases_with_rarity(self):
        docs = [Counter({"common": 1})] * 5 + [Counter({"rare": 1})]
        common_idf = _idf("common", docs, 6)
        rare_idf = _idf("rare", docs, 6)
        assert rare_idf > common_idf

    def test_total_docs_zero_guard(self):
        # df=0 path returns 0, even with total_docs=0
        assert _idf("anything", [], 0) == 0.0


# ---------------------------------------------------------------------------
# FitnessResult.f1
# ---------------------------------------------------------------------------

class TestFitnessResultF1:
    def _make_result(self, recall: float, compression: float) -> FitnessResult:
        return FitnessResult(
            method="test",
            recall=recall,
            speed_s=0.0,
            compression=compression,
            budget=1000,
            total_tokens=1000,
            kept_tokens=int(compression * 1000),
            prefix_turns=10,
            suffix_turns=5,
            suffix_vocab_size=50,
            scored_count=8,
            kept_scored=4,
            dropped_scored=4,
        )

    def test_perfect_recall_zero_compression_gives_zero_f1(self):
        r = self._make_result(recall=1.0, compression=1.0)
        # compression_eff = 0, recall = 1 → harmonic mean = 0
        assert r.f1 == 0.0

    def test_zero_recall_perfect_compression_gives_zero_f1(self):
        r = self._make_result(recall=0.0, compression=0.0)
        # compression_eff = 1, recall = 0 → harmonic mean = 0
        assert r.f1 == 0.0

    def test_balanced_gives_nonzero_f1(self):
        r = self._make_result(recall=0.8, compression=0.5)
        # compression_eff = 0.5
        expected = 2 * 0.8 * 0.5 / (0.8 + 0.5)
        assert abs(r.f1 - expected) < 1e-9

    def test_both_zero_gives_zero(self):
        r = self._make_result(recall=0.0, compression=1.0)
        # recall=0, compression_eff=0 → denom=0 → returns 0
        assert r.f1 == 0.0

    def test_f1_between_zero_and_one(self):
        for recall in [0.2, 0.5, 0.9]:
            for comp in [0.1, 0.5, 0.8]:
                r = self._make_result(recall=recall, compression=comp)
                assert 0.0 <= r.f1 <= 1.0


# ---------------------------------------------------------------------------
# _format_instruct
# ---------------------------------------------------------------------------

class TestFormatInstruct:
    def test_basic_format(self):
        result = _format_instruct("Find relevant docs", "some text")
        assert result == "Instruct: Find relevant docs\nQuery: some text"

    def test_empty_instruction(self):
        result = _format_instruct("", "text")
        assert result == "Instruct: \nQuery: text"

    def test_empty_text(self):
        result = _format_instruct("instruction", "")
        assert result == "Instruct: instruction\nQuery: "

    def test_newline_preserved_in_text(self):
        result = _format_instruct("instr", "line1\nline2")
        assert "line1\nline2" in result

    def test_returns_string(self):
        assert isinstance(_format_instruct("a", "b"), str)


# ---------------------------------------------------------------------------
# _last_token_pool
# ---------------------------------------------------------------------------

class TestLastTokenPool:
    def test_right_padded_single(self):
        # Sequence [token, pad]: last real token is at index 0
        hidden = torch.tensor([[[1.0, 2.0], [0.0, 0.0]]])  # (1, 2, 2)
        mask = torch.tensor([[1, 0]])  # attend to index 0 only
        result = _last_token_pool(hidden, mask)
        assert result.shape == (1, 2)
        assert torch.allclose(result, torch.tensor([[1.0, 2.0]]))

    def test_right_padded_both_real(self):
        # Two real tokens, no padding
        hidden = torch.tensor([[[1.0, 0.0], [3.0, 4.0]]])  # (1, 2, 2)
        mask = torch.tensor([[1, 1]])
        result = _last_token_pool(hidden, mask)
        # Last real token is at index 1
        assert torch.allclose(result, torch.tensor([[3.0, 4.0]]))

    def test_left_padded_single(self):
        # Left padding: [pad, token] → left_padding condition triggers → use last position
        hidden = torch.tensor([[[0.0, 0.0], [5.0, 6.0]]])  # (1, 2, 2)
        mask = torch.tensor([[0, 1]])
        # left_padding: mask[:, -1].sum() == batch_size → True
        result = _last_token_pool(hidden, mask)
        assert torch.allclose(result, torch.tensor([[5.0, 6.0]]))

    def test_batch_right_padded(self):
        # Batch of 2, different sequence lengths
        hidden = torch.tensor([
            [[1.0, 0.0], [2.0, 0.0], [0.0, 0.0]],  # len=2, last real at idx 1
            [[3.0, 0.0], [4.0, 0.0], [5.0, 0.0]],  # len=3, last real at idx 2
        ])  # (2, 3, 2)
        mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
        result = _last_token_pool(hidden, mask)
        assert result.shape == (2, 2)
        assert torch.allclose(result[0], torch.tensor([2.0, 0.0]))
        assert torch.allclose(result[1], torch.tensor([5.0, 0.0]))

    def test_output_shape(self):
        hidden = torch.randn(4, 8, 16)
        mask = torch.ones(4, 8, dtype=torch.long)
        result = _last_token_pool(hidden, mask)
        assert result.shape == (4, 16)


# ---------------------------------------------------------------------------
# build_query
# ---------------------------------------------------------------------------

class TestBuildQuery:
    def _user_turn(self, text: str) -> Turn:
        t = Turn(kind="user")
        t.append({"message": {"content": text}})
        return t

    def test_single_turn(self):
        turns = [self._user_turn("what is this?")]
        result = build_query(turns)
        assert "what is this?" in result

    def test_uses_last_three_turns(self):
        turns = [self._user_turn(f"msg{i}") for i in range(5)]
        result = build_query(turns)
        # Should contain msgs 2, 3, 4 but not 0 or 1
        assert "msg4" in result
        assert "msg3" in result
        assert "msg2" in result

    def test_truncated_at_max_chars(self):
        long_text = "x" * 5000
        turns = [self._user_turn(long_text)]
        result = build_query(turns, max_chars=4000)
        assert len(result) <= 4000

    def test_empty_turns(self):
        result = build_query([])
        assert result == ""

    def test_returns_string(self):
        turns = [self._user_turn("hello")]
        assert isinstance(build_query(turns), str)


# ---------------------------------------------------------------------------
# SuffixAutomaton
# ---------------------------------------------------------------------------

class TestSuffixAutomaton:
    def test_build_and_propagate(self):
        sa = SuffixAutomaton()
        for i, c in enumerate("abcabc"):
            sa.extend(c, i)
        sa.propagate_counts()
        # Should have states
        assert len(sa.states) > 1

    def test_match_repeated_length_all_repeated(self):
        sa = SuffixAutomaton()
        text = "aaaa"
        for i, c in enumerate(text):
            sa.extend(c, i)
        sa.propagate_counts()
        lengths = sa.match_repeated_length(text)
        assert len(lengths) == len(text)

    def test_match_repeated_length_no_repeats(self):
        sa = SuffixAutomaton()
        text = "abcd"
        for i, c in enumerate(text):
            sa.extend(c, i)
        sa.propagate_counts()
        lengths = sa.match_repeated_length(text)
        # Unique chars — nothing repeats more than once
        assert all(l == 0 for l in lengths)

    def test_match_repeated_length_partial_repeat(self):
        sa = SuffixAutomaton()
        # "xyzxyz" — "xyz" appears twice
        text = "xyzxyz"
        for i, c in enumerate(text):
            sa.extend(c, i)
        sa.propagate_counts()
        lengths = sa.match_repeated_length(text)
        # The second "xyz" (positions 3-5) should have repeat lengths > 0
        assert any(l > 0 for l in lengths[3:])

    def test_empty_string(self):
        sa = SuffixAutomaton()
        sa.propagate_counts()
        lengths = sa.match_repeated_length("")
        assert lengths == []

    def test_initial_state_exists(self):
        sa = SuffixAutomaton()
        assert len(sa.states) == 1
        assert sa.states[0].len == 0
        assert sa.states[0].link == -1
