"""Tests for filter/phase_qa_filter.py — pure Python, no LLM required."""

from SeedDataGen.filter.phase_qa_filter import QAFilterConfig, QAFilterPhase, _filter_qa_group
from SeedDataGen.schemas import StyledQARow


def _row(question: str, answer: str, question_style: str = "general", **extra) -> dict:
    return {
        "id": 0,
        "origin_id": 0,
        "sample_id": 1,
        "sample_text": "some text",
        "question": question,
        "answer": answer,
        "question_style": question_style,
        **extra,
    }


def _cfg(**overrides) -> QAFilterConfig:
    defaults = dict(min_answer_len=10, levenshtein_threshold=20)
    return QAFilterConfig(**{**defaults, **overrides})


# --- schema contract ---

def test_input_schema_is_styled_qa_row():
    assert QAFilterPhase.input_schema is StyledQARow


def test_output_schema_is_styled_qa_row():
    assert QAFilterPhase.output_schema is StyledQARow


# --- answer length filter ---

def test_drops_short_answer():
    rows = [_row("What is X?", "Short")]  # 5 chars < 10
    cfg = _cfg(min_answer_len=10)
    assert _filter_qa_group(rows, cfg) == []


def test_keeps_answer_exactly_at_threshold():
    rows = [_row("What is X?", "A" * 10)]
    cfg = _cfg(min_answer_len=10)
    assert len(_filter_qa_group(rows, cfg)) == 1


def test_keeps_long_answer():
    rows = [_row("What is X?", "This is a sufficiently long answer.")]
    cfg = _cfg(min_answer_len=10)
    assert len(_filter_qa_group(rows, cfg)) == 1


def test_strips_whitespace_before_length_check():
    # Whitespace-only padding shouldn't inflate the effective length.
    rows = [_row("What is X?", "   short   ")]  # strip → 5 chars
    cfg = _cfg(min_answer_len=10)
    assert _filter_qa_group(rows, cfg) == []


# --- near-duplicate question filter ---

def test_drops_near_duplicate_question():
    q1 = "What is the operating voltage?"
    q2 = "What is the operating voltages?"  # 1 edit from q1
    rows = [_row(q1, "Long enough answer here."), _row(q2, "Long enough answer here.")]
    cfg = _cfg(levenshtein_threshold=5)
    result = _filter_qa_group(rows, cfg)
    assert len(result) == 1
    assert result[0]["question"] == q1


def test_keeps_distinct_questions():
    q1 = "What is the operating voltage of the low-voltage network?"
    q2 = "How does the protection relay calibration procedure work?"
    rows = [_row(q1, "Long enough answer here."), _row(q2, "Another sufficiently long answer.")]
    cfg = _cfg(levenshtein_threshold=5)
    assert len(_filter_qa_group(rows, cfg)) == 2


def test_dedup_is_order_dependent_keeps_first():
    q1 = "AAA BBB CCC"
    q2 = "AAA BBB CCD"  # 1 edit
    rows = [_row(q1, "Answer long enough."), _row(q2, "Answer long enough.")]
    cfg = _cfg(levenshtein_threshold=5)
    result = _filter_qa_group(rows, cfg)
    assert result[0]["question"] == q1


# --- passthrough ---

def test_passthrough_preserves_question_style():
    rows = [_row("Some question here?", "A long enough answer for this test.", question_style="specific")]
    cfg = _cfg()
    result = _filter_qa_group(rows, cfg)
    assert result[0]["question_style"] == "specific"


def test_passthrough_preserves_extra_fields():
    rows = [_row("Some question here?", "A long enough answer for this test.", GEN_TYPE="qa_gen_var", num_chunks=2)]
    cfg = _cfg()
    result = _filter_qa_group(rows, cfg)
    assert result[0]["GEN_TYPE"] == "qa_gen_var"
    assert result[0]["num_chunks"] == 2


def test_empty_group_returns_empty():
    assert _filter_qa_group([], _cfg()) == []
