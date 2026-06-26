"""Tests for run_pipeline._finalize_phase_output (empty-intermediate handling)."""

from SeedDataGen.base_phase import PhaseRole
from SeedDataGen.run_pipeline import _finalize_phase_output


class _FakePhase:
    def __init__(self, name="fake", role=PhaseRole.FILTER):
        self.name = name
        self.role = role


def test_missing_output_is_touched_and_recorded(tmp_path):
    out = tmp_path / "nested" / "out.jsonl"  # parent does not exist yet
    empties: list[tuple[str, str]] = []
    _finalize_phase_output(_FakePhase(), str(out), empties)
    assert out.exists()
    assert out.read_text() == ""
    assert empties == [("fake", str(out))]


def test_nonempty_output_is_left_alone(tmp_path):
    out = tmp_path / "out.jsonl"
    out.write_text('{"id": 0}\n', encoding="utf-8")
    empties: list[tuple[str, str]] = []
    _finalize_phase_output(_FakePhase(), str(out), empties)
    assert empties == []
    assert out.read_text() == '{"id": 0}\n'


def test_preprocess_phase_skipped(tmp_path):
    out = tmp_path / "should_not_be_created.jsonl"
    empties: list[tuple[str, str]] = []
    _finalize_phase_output(_FakePhase(role=PhaseRole.PREPROCESS), str(out), empties)
    assert not out.exists()
    assert empties == []
