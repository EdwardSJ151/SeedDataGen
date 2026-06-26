"""Tests for the single-turn-aware judge — no live LLM required."""

from SeedDataGen.judge.prompts import JUDGE_PROMPT, JUDGE_PROMPT_SINGLE_TURN
from SeedDataGen.utils import is_single_turn, parse_judge_scores


def test_parse_judge_scores_accepts_diversidade():
    out = "Fidelidade: 5\nCorreção: 4\nClareza: 5\nCoerência: 4\nDiversidade: 3"
    assert parse_judge_scores(out) == [5.0, 4.0, 5.0, 4.0, 3.0]


def test_parse_judge_scores_accepts_completude():
    # single-turn judge emits Completude as the 5th line; still 5 scores.
    out = "Fidelidade: 5\nCorreção: 4\nClareza: 5\nCoerência: 4\nCompletude: 4,5"
    assert parse_judge_scores(out) == [5.0, 4.0, 5.0, 4.0, 4.5]


def test_parse_judge_scores_requires_five():
    out = "Fidelidade: 5\nCorreção: 4\nClareza: 5\nCoerência: 4"
    assert parse_judge_scores(out) is None


def test_is_single_turn():
    single = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    ]
    multi = single + [
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ]
    assert is_single_turn(single) is True
    assert is_single_turn(multi) is False


def test_prompt_variants_differ_on_fifth_criterion():
    assert "Completude" in JUDGE_PROMPT_SINGLE_TURN
    assert "Diversidade" not in JUDGE_PROMPT_SINGLE_TURN
    assert "Diversidade" in JUDGE_PROMPT
    assert "Completude" not in JUDGE_PROMPT
