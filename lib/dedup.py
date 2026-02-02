"""Suffix automaton dedup scoring for system turns.

Builds a suffix automaton (DAWG) over the full conversation text to detect
repeated content in O(n). Scores each turn by its unique content ratio —
turns that are mostly duplicated elsewhere get low scores.

No ML model needed. Near-instant on any hardware.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .parser import Turn, extract_text
from .scorer import ScoredTurn


# -- Suffix Automaton (online, O(n) construction) --

@dataclass
class _State:
    len: int = 0
    link: int = -1
    trans: dict[str, int] = field(default_factory=dict)
    cnt: int = 0  # endpos count (propagated after build)
    first_pos: int = 0


class SuffixAutomaton:
    """O(n) suffix automaton supporting occurrence counting."""

    def __init__(self) -> None:
        init = _State(len=0, link=-1)
        self.states: list[_State] = [init]
        self.last = 0

    def extend(self, c: str, pos: int) -> None:
        cur = len(self.states)
        self.states.append(_State(len=self.states[self.last].len + 1, cnt=1, first_pos=pos))
        p = self.last
        while p != -1 and c not in self.states[p].trans:
            self.states[p].trans[c] = cur
            p = self.states[p].link
        if p == -1:
            self.states[cur].link = 0
        else:
            q = self.states[p].trans[c]
            if self.states[p].len + 1 == self.states[q].len:
                self.states[cur].link = q
            else:
                clone = len(self.states)
                self.states.append(_State(
                    len=self.states[p].len + 1,
                    link=self.states[q].link,
                    trans=dict(self.states[q].trans),
                    cnt=0,
                    first_pos=self.states[q].first_pos,
                ))
                while p != -1 and self.states[p].trans.get(c) == q:
                    self.states[p].trans[c] = clone
                    p = self.states[p].link
                self.states[q].link = clone
                self.states[cur].link = clone
        self.last = cur

    def propagate_counts(self) -> None:
        """Propagate endpos counts through suffix links (topological order)."""
        order = sorted(range(len(self.states)), key=lambda i: -self.states[i].len)
        for v in order:
            if self.states[v].link >= 0:
                self.states[self.states[v].link].cnt += self.states[v].cnt

    def match_repeated_length(self, text: str) -> list[int]:
        """For each position in text, find the longest substring ending there
        that occurs MORE THAN ONCE in the automaton's source string.

        Only counts substrings where cnt >= 2 (i.e., appears elsewhere too,
        not just in the current occurrence).

        Returns a list of lengths, one per character.
        """
        lengths: list[int] = []
        cur = 0
        cur_len = 0
        for c in text:
            while cur != 0 and c not in self.states[cur].trans:
                cur = self.states[cur].link
                cur_len = self.states[cur].len
            if c in self.states[cur].trans:
                cur = self.states[cur].trans[c]
                cur_len += 1
            else:
                cur = 0
                cur_len = 0
            # Walk up suffix links until we find a state with cnt >= 2
            # (meaning this substring appears in at least 2 places in the corpus)
            effective = cur
            effective_len = cur_len
            while effective != 0 and self.states[effective].cnt < 2:
                effective = self.states[effective].link
                effective_len = self.states[effective].len
            lengths.append(effective_len)
        return lengths


def _build_automaton(turns: list[Turn]) -> tuple[SuffixAutomaton, dict[int, tuple[int, int]]]:
    """Build a suffix automaton over all turn texts.

    Returns the automaton and a map of turn.index -> (start, end) positions
    in the concatenated string.
    """
    sa = SuffixAutomaton()
    turn_spans: dict[int, tuple[int, int]] = {}
    pos = 0

    for turn in turns:
        text = extract_text(turn)
        start = pos
        for c in text:
            sa.extend(c, pos)
            pos += 1
        # Separator to prevent cross-turn substring matches
        sa.extend("\x00", pos)
        pos += 1
        turn_spans[turn.index] = (start, start + len(text))

    sa.propagate_counts()
    return sa, turn_spans


def _turn_unique_ratio(
    sa: SuffixAutomaton,
    text: str,
    min_repeat_len: int = 64,
) -> float:
    """Compute the fraction of a turn's text that is unique (not repeated).

    Walks the automaton matching the turn's text. Characters that are part of
    a repeated substring (length >= min_repeat_len, count >= 2) are marked as
    duplicated. Returns unique_chars / total_chars.
    """
    if not text:
        return 1.0

    match_lens = sa.match_repeated_length(text)
    duplicated = 0

    # Mark characters covered by long repeated substrings
    # A match of length L at position i covers characters [i-L+1, i]
    # We use a simple greedy sweep
    covered_until = -1
    for i, ml in enumerate(match_lens):
        if ml >= min_repeat_len:
            start = i - ml + 1
            if start > covered_until:
                duplicated += ml
            elif i > covered_until:
                duplicated += i - covered_until
            covered_until = max(covered_until, i)

    unique = len(text) - duplicated
    return unique / len(text)


def dedup_scores(
    turns: list[Turn],
    system_turns: list[Turn],
    token_counts: dict[int, int],
    min_repeat_len: int = 64,
) -> list[ScoredTurn]:
    """Score system turns by unique content ratio using suffix automaton.

    Returns ScoredTurn objects where score = unique_ratio (0-1).
    Turns with mostly unique content score high; mostly-repeated content scores low.
    """
    print("  Building suffix automaton...", flush=True)
    sa, turn_spans = _build_automaton(turns)
    print(f"  Automaton: {len(sa.states):,} states over {sum(e-s for s,e in turn_spans.values()):,} chars", flush=True)

    print(f"  Scoring {len(system_turns)} system turns...", flush=True)
    results: list[ScoredTurn] = []
    for turn in system_turns:
        text = extract_text(turn)
        ratio = _turn_unique_ratio(sa, text, min_repeat_len)
        results.append(ScoredTurn(
            turn=turn,
            score=ratio,
            tokens=token_counts.get(turn.index, 0),
        ))

    return results
