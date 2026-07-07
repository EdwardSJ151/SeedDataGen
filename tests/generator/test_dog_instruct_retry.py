import json

import pytest

from SeedDataGen.generator.phase_dog_instruct import (
    DogInstructConfig,
    _process_batch,
)


def _cfg(**overrides) -> DogInstructConfig:
    defaults = dict(
        personas=["persona_a", "persona_b"],
        question_temperature=0.7,
        question_top_p=0.9,
        question_max_tokens=256,
        rewrite_temperature=0.4,
        rewrite_top_p=0.9,
        rewrite_max_tokens=1024,
        batch_size=32,
        max_concurrent=8,
    )
    defaults.update(overrides)
    return DogInstructConfig(**defaults)


def _sample(hf_row_id: str = "42") -> dict:
    return {
        "hf_row_id": hf_row_id,
        "sample_text": {"text": "Texto técnico.", "document_name": "ND-1"},
        "prompt_text": "[Chunk 1 | ND-1]\nTexto técnico.",
        "answer_text": "Texto técnico.",
    }


# ---------------------------------------------------------------------------
# _process_batch with retry_pairs filter
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_process_batch_retry_pairs_skips_non_retry_style(monkeypatch, tmp_path):
    """Only persona_a should be generated when retry_pairs targets persona_a."""
    calls: list[str] = []

    async def _fake_question(client, cfg, model_id, sample_text, persona, **kw):
        calls.append(persona)
        return f"Pergunta para {persona}"

    async def _fake_rewrite(*args, **kwargs):
        return "Resposta reescrita."

    monkeypatch.setattr(
        "SeedDataGen.generator.phase_dog_instruct._generate_user_question", _fake_question
    )
    monkeypatch.setattr(
        "SeedDataGen.generator.phase_dog_instruct._rewrite_assistant_answer", _fake_rewrite
    )

    output_file = tmp_path / "out.jsonl"
    sample = _sample("10")
    retry_pairs = {("10", "persona_a")}

    await _process_batch(
        client=None,
        cfg=_cfg(),
        model_id="model",
        samples=[sample],
        next_row_id=0,
        output_file=str(output_file),
        summary_map={},
        retry_pairs=retry_pairs,
    )

    assert calls == ["persona_a"], "persona_b should be skipped when not in retry_pairs"

    rows = [json.loads(l) for l in output_file.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["question_style"] == "persona_a"


@pytest.mark.asyncio
async def test_process_batch_no_retry_pairs_generates_all(monkeypatch, tmp_path):
    """Without retry_pairs, all personas are generated (existing behaviour)."""
    calls: list[str] = []

    async def _fake_question(client, cfg, model_id, sample_text, persona, **kw):
        calls.append(persona)
        return f"Pergunta para {persona}"

    async def _fake_rewrite(*args, **kwargs):
        return "Resposta."

    monkeypatch.setattr(
        "SeedDataGen.generator.phase_dog_instruct._generate_user_question", _fake_question
    )
    monkeypatch.setattr(
        "SeedDataGen.generator.phase_dog_instruct._rewrite_assistant_answer", _fake_rewrite
    )

    output_file = tmp_path / "out.jsonl"
    await _process_batch(
        client=None,
        cfg=_cfg(),
        model_id="model",
        samples=[_sample("20")],
        next_row_id=0,
        output_file=str(output_file),
        summary_map={},
        retry_pairs=None,
    )

    assert set(calls) == {"persona_a", "persona_b"}


@pytest.mark.asyncio
async def test_process_batch_retry_pairs_multiple_samples(monkeypatch, tmp_path):
    """Only the targeted (chunk, style) pair is generated across multiple samples."""
    calls: list[tuple] = []

    async def _fake_question(client, cfg, model_id, sample_text, persona, **kw):
        calls.append(persona)
        return f"Q-{persona}"

    async def _fake_rewrite(*args, **kwargs):
        return "A."

    monkeypatch.setattr(
        "SeedDataGen.generator.phase_dog_instruct._generate_user_question", _fake_question
    )
    monkeypatch.setattr(
        "SeedDataGen.generator.phase_dog_instruct._rewrite_assistant_answer", _fake_rewrite
    )

    output_file = tmp_path / "out.jsonl"
    # chunk 10 → retry persona_b; chunk 11 → retry persona_a
    retry_pairs = {("10", "persona_b"), ("11", "persona_a")}
    samples = [_sample("10"), _sample("11")]

    await _process_batch(
        client=None,
        cfg=_cfg(),
        model_id="model",
        samples=samples,
        next_row_id=0,
        output_file=str(output_file),
        summary_map={},
        retry_pairs=retry_pairs,
    )

    assert calls.count("persona_a") == 1
    assert calls.count("persona_b") == 1
    rows = [json.loads(l) for l in output_file.read_text().splitlines()]
    assert len(rows) == 2


# ---------------------------------------------------------------------------
# skip_ids excludes failed rows
# ---------------------------------------------------------------------------

def test_run_skips_failed_sample_ids_via_exclude_status(tmp_path):
    """
    get_processed_sample_ids with exclude_status=["failed"] must not include
    sample_ids from failed rows — so generators treat failed chunks as re-eligible.
    """
    from SeedDataGen.utils import get_processed_sample_ids

    gen_out = tmp_path / "gen.jsonl"
    rows = [
        {"id": 0, "sample_id": [1], "question_style": "persona_a", "status": "passed"},
        {"id": 1, "sample_id": [1], "question_style": "persona_b", "status": "failed"},
        {"id": 2, "sample_id": [2], "question_style": "persona_a", "status": "passed"},
    ]
    with open(gen_out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    skip_ids = get_processed_sample_ids(str(gen_out), exclude_status=["failed"])
    # sample 1 still appears (via the passed row), sample 2 appears
    assert "1" in skip_ids
    assert "2" in skip_ids

    # If ALL rows for a sample_id are failed, it should NOT be in skip_ids
    gen_out2 = tmp_path / "gen2.jsonl"
    rows2 = [
        {"id": 0, "sample_id": [3], "question_style": "persona_a", "status": "failed"},
        {"id": 1, "sample_id": [3], "question_style": "persona_b", "status": "failed"},
    ]
    with open(gen_out2, "w") as f:
        for r in rows2:
            f.write(json.dumps(r) + "\n")

    skip_ids2 = get_processed_sample_ids(str(gen_out2), exclude_status=["failed"])
    assert "3" not in skip_ids2
