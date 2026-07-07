import json

import pytest

from SeedDataGen.utils import get_failed_pairs, get_processed_sample_ids, stamp_statuses


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# get_processed_sample_ids
# ---------------------------------------------------------------------------

def test_get_processed_sample_ids_baseline(tmp_path):
    p = tmp_path / "out.jsonl"
    _write_jsonl(p, [
        {"id": 0, "sample_id": [1]},
        {"id": 1, "sample_id": [2], "status": "passed"},
        {"id": 2, "sample_id": [3], "status": "failed"},
    ])
    result = get_processed_sample_ids(str(p))
    assert result == {"1", "2", "3"}


def test_get_processed_sample_ids_exclude_failed(tmp_path):
    p = tmp_path / "out.jsonl"
    _write_jsonl(p, [
        {"id": 0, "sample_id": [1], "status": "passed"},
        {"id": 1, "sample_id": [2], "status": "failed"},
        {"id": 2, "sample_id": [3]},
    ])
    result = get_processed_sample_ids(str(p), exclude_status=["failed"])
    assert result == {"1", "3"}
    assert "2" not in result


def test_get_processed_sample_ids_exclude_failed_all_failed(tmp_path):
    p = tmp_path / "out.jsonl"
    _write_jsonl(p, [
        {"id": 0, "sample_id": [5], "status": "failed"},
        {"id": 1, "sample_id": [6], "status": "failed"},
    ])
    result = get_processed_sample_ids(str(p), exclude_status=["failed"])
    assert result == set()


def test_get_processed_sample_ids_missing_file(tmp_path):
    result = get_processed_sample_ids(str(tmp_path / "missing.jsonl"), exclude_status=["failed"])
    assert result == set()


# ---------------------------------------------------------------------------
# get_failed_pairs
# ---------------------------------------------------------------------------

def test_get_failed_pairs_returns_failed_only(tmp_path):
    p = tmp_path / "gen.jsonl"
    _write_jsonl(p, [
        {"id": 0, "sample_id": [10], "question_style": "A", "status": "passed"},
        {"id": 1, "sample_id": [10], "question_style": "B", "status": "failed"},
        {"id": 2, "sample_id": [11], "question_style": "A"},
        {"id": 3, "sample_id": [11], "question_style": "C", "status": "failed"},
    ])
    result = get_failed_pairs(str(p))
    assert result == {("10", "B"), ("11", "C")}


def test_get_failed_pairs_no_style_uses_empty_string(tmp_path):
    p = tmp_path / "gen.jsonl"
    _write_jsonl(p, [
        {"id": 0, "sample_id": [5], "status": "failed"},
    ])
    result = get_failed_pairs(str(p))
    assert result == {("5", "")}


def test_get_failed_pairs_empty_file(tmp_path):
    p = tmp_path / "gen.jsonl"
    p.write_text("")
    assert get_failed_pairs(str(p)) == set()


def test_get_failed_pairs_missing_file(tmp_path):
    assert get_failed_pairs(str(tmp_path / "nope.jsonl")) == set()


# ---------------------------------------------------------------------------
# stamp_statuses
# ---------------------------------------------------------------------------

def test_stamp_statuses_basic(tmp_path):
    p = tmp_path / "gen.jsonl"
    _write_jsonl(p, [
        {"id": 0, "sample_id": [1]},
        {"id": 1, "sample_id": [2]},
        {"id": 2, "sample_id": [3]},
    ])
    stamp_statuses(str(p), passed_ids={0, 2}, failed_ids={1})
    rows = _read_jsonl(p)
    assert rows[0]["status"] == "passed"
    assert rows[1]["status"] == "failed"
    assert rows[2]["status"] == "passed"


def test_stamp_statuses_untouched_rows_unchanged(tmp_path):
    p = tmp_path / "gen.jsonl"
    _write_jsonl(p, [
        {"id": 0, "sample_id": [1], "status": "passed"},
        {"id": 1, "sample_id": [2]},
        {"id": 2, "sample_id": [3], "status": "failed"},
    ])
    stamp_statuses(str(p), passed_ids=set(), failed_ids=set())
    rows = _read_jsonl(p)
    assert rows[0]["status"] == "passed"
    assert "status" not in rows[1] or rows[1].get("status") is None
    assert rows[2]["status"] == "failed"


def test_stamp_statuses_overwrites_existing_status(tmp_path):
    p = tmp_path / "gen.jsonl"
    _write_jsonl(p, [
        {"id": 0, "sample_id": [1], "status": "failed"},
    ])
    stamp_statuses(str(p), passed_ids={0}, failed_ids=set())
    rows = _read_jsonl(p)
    assert rows[0]["status"] == "passed"


def test_stamp_statuses_missing_file_is_noop(tmp_path):
    stamp_statuses(str(tmp_path / "nope.jsonl"), passed_ids={1}, failed_ids={2})
