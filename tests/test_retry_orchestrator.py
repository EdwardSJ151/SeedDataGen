"""
Tests for the retry orchestrator in run_pipeline.py.
Uses real JSONL files and mocked phase.run() calls to verify the
_build_gen_pairs, _build_passed_pairs, and stamp_statuses interactions.
"""

import json
import os

import pytest

from SeedDataGen.run_pipeline import _build_gen_pairs, _build_passed_pairs
from SeedDataGen.utils import get_failed_pairs, stamp_statuses


def _write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f]


# ---------------------------------------------------------------------------
# _build_gen_pairs
# ---------------------------------------------------------------------------

def test_build_gen_pairs_excludes_already_failed(tmp_path):
    gen_out = str(tmp_path / "gen.jsonl")
    _write_jsonl(gen_out, [
        {"id": 0, "sample_id": [1], "question_style": "A"},
        {"id": 1, "sample_id": [1], "question_style": "B", "status": "failed"},
        {"id": 2, "sample_id": [2], "question_style": "A"},
    ])
    pairs = _build_gen_pairs(gen_out)
    assert ("1", "A") in pairs
    assert ("2", "A") in pairs
    # status="failed" rows are excluded from gen_pairs
    assert ("1", "B") not in pairs


def test_build_gen_pairs_tracks_latest_ids(tmp_path):
    gen_out = str(tmp_path / "gen.jsonl")
    _write_jsonl(gen_out, [
        {"id": 0, "sample_id": [5], "question_style": "X"},
        {"id": 10, "sample_id": [5], "question_style": "X"},  # retry row
    ])
    pairs = _build_gen_pairs(gen_out)
    assert pairs[("5", "X")] == [0, 10]


# ---------------------------------------------------------------------------
# _build_passed_pairs
# ---------------------------------------------------------------------------

def test_build_passed_pairs(tmp_path):
    embed_out = str(tmp_path / "embed.jsonl")
    _write_jsonl(embed_out, [
        {"id": 0, "sample_id": [1], "question_style": "A", "scores": {}, "avg_score": 4.5, "messages": []},
        {"id": 1, "sample_id": [2], "question_style": "B", "scores": {}, "avg_score": 4.5, "messages": []},
    ])
    passed = _build_passed_pairs(embed_out)
    assert passed == {("1", "A"), ("2", "B")}


def test_build_passed_pairs_empty_file(tmp_path):
    p = str(tmp_path / "empty.jsonl")
    _write_jsonl(p, [])
    assert _build_passed_pairs(p) == set()


# ---------------------------------------------------------------------------
# stamp_statuses integration
# ---------------------------------------------------------------------------

def test_stamp_and_detect_cycle(tmp_path):
    """Full cycle: build pairs → identify failures → stamp → verify get_failed_pairs."""
    gen_out = str(tmp_path / "gen.jsonl")
    _write_jsonl(gen_out, [
        {"id": 0, "sample_id": [1], "question_style": "A"},
        {"id": 1, "sample_id": [1], "question_style": "B"},
        {"id": 2, "sample_id": [2], "question_style": "A"},
    ])

    embed_out = str(tmp_path / "embed.jsonl")
    _write_jsonl(embed_out, [
        {"id": 0, "sample_id": [1], "question_style": "A", "scores": {}, "avg_score": 4.5, "messages": []},
    ])

    all_gen_pairs = _build_gen_pairs(gen_out)
    passed_pairs = _build_passed_pairs(embed_out)
    failed_pairs = {k for k in all_gen_pairs if k not in passed_pairs}

    assert failed_pairs == {("1", "B"), ("2", "A")}

    passed_ids = {max(all_gen_pairs[k]) for k in all_gen_pairs if k in passed_pairs}
    failed_ids = {max(all_gen_pairs[k]) for k in all_gen_pairs if k in failed_pairs}

    stamp_statuses(gen_out, passed_ids, failed_ids)

    rows = _read_jsonl(gen_out)
    assert rows[0]["status"] == "passed"   # (1, A)
    assert rows[1]["status"] == "failed"   # (1, B)
    assert rows[2]["status"] == "failed"   # (2, A)

    failed_after = get_failed_pairs(gen_out)
    assert failed_after == {("1", "B"), ("2", "A")}


# ---------------------------------------------------------------------------
# retry convergence: second attempt passes
# ---------------------------------------------------------------------------

def test_retry_converges_when_no_failures(tmp_path):
    """If all pairs are in the embed output, failed_pairs should be empty."""
    gen_out = str(tmp_path / "gen.jsonl")
    _write_jsonl(gen_out, [
        {"id": 0, "sample_id": [1], "question_style": "A"},
        {"id": 1, "sample_id": [2], "question_style": "B"},
    ])
    embed_out = str(tmp_path / "embed.jsonl")
    _write_jsonl(embed_out, [
        {"id": 0, "sample_id": [1], "question_style": "A", "scores": {}, "avg_score": 4.5, "messages": []},
        {"id": 1, "sample_id": [2], "question_style": "B", "scores": {}, "avg_score": 4.5, "messages": []},
    ])

    all_gen_pairs = _build_gen_pairs(gen_out)
    passed_pairs = _build_passed_pairs(embed_out)
    failed_pairs = {k for k in all_gen_pairs if k not in passed_pairs}

    assert failed_pairs == set()
