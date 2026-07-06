import json

import pytest
from SeedDataGen.generator.phase_dog_instruct import (
    GEN_TYPE,
    DogInstructConfig,
    _clean_answer_output,
    _clean_question_output,
    _process_batch,
)


def _cfg(**overrides) -> DogInstructConfig:
    defaults = dict(
        personas=["direto_objetivo"],
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


def _sample(sample_id: str = "1", *, doc_id: int | None = None) -> dict:
    sample = {
        "hf_row_id": sample_id,
        "sample_text": {"text": "Texto base técnico.", "document_name": "ND-1"},
        "prompt_text": "[Chunk 1 | ND-1]\nTexto base técnico.",
        "answer_text": "Texto base técnico.",
    }
    if doc_id is not None:
        sample["doc_id"] = doc_id
    return sample


def test_clean_question_output_strips_label_and_whitespace():
    assert (
        _clean_question_output(" Pergunta:   Qual e a tensao?  ") == "Qual e a tensao?"
    )


def test_clean_answer_output_strips_label():
    assert _clean_answer_output("Resposta: Conteudo final") == "Conteudo final"


@pytest.mark.asyncio
async def test_process_batch_writes_generated_row(monkeypatch, tmp_path):
    async def _fake_question(*args, **kwargs):
        return "Qual e a tensao nominal?"

    async def _fake_rewrite(*args, **kwargs):
        return "A tensao nominal e 13,8 kV."

    monkeypatch.setattr(
        "SeedDataGen.generator.phase_dog_instruct._generate_user_question",
        _fake_question,
    )
    monkeypatch.setattr(
        "SeedDataGen.generator.phase_dog_instruct._rewrite_assistant_answer",
        _fake_rewrite,
    )

    output_file = tmp_path / "dog_instruct.jsonl"
    next_id = await _process_batch(
        client=None,
        cfg=_cfg(),
        model_id="model-id",
        samples=[_sample(doc_id=7)],
        next_row_id=0,
        output_file=str(output_file),
        summary_map={"7": "Resumo do documento."},
    )

    assert next_id == 1
    rows = [
        json.loads(line)
        for line in output_file.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 1
    assert rows[0]["messages"] == [
        {"role": "user", "content": "Qual e a tensao nominal?"},
        {"role": "assistant", "content": "A tensao nominal e 13,8 kV."},
    ]
    assert rows[0]["GEN_TYPE"] == GEN_TYPE
    assert rows[0]["question_style"] == "direto_objetivo"
    assert rows[0]["document_id"] == 7


@pytest.mark.asyncio
async def test_process_batch_falls_back_to_original_answer_when_rewrite_fails(
    monkeypatch, tmp_path
):
    async def _fake_question(*args, **kwargs):
        return "O que o texto explica?"

    async def _fake_rewrite(*args, **kwargs):
        return None

    monkeypatch.setattr(
        "SeedDataGen.generator.phase_dog_instruct._generate_user_question",
        _fake_question,
    )
    monkeypatch.setattr(
        "SeedDataGen.generator.phase_dog_instruct._rewrite_assistant_answer",
        _fake_rewrite,
    )

    output_file = tmp_path / "dog_instruct_fallback.jsonl"
    await _process_batch(
        client=None,
        cfg=_cfg(),
        model_id="model-id",
        samples=[_sample()],
        next_row_id=0,
        output_file=str(output_file),
        summary_map={},
    )

    rows = [
        json.loads(line)
        for line in output_file.read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["messages"][1]["content"] == "Texto base técnico."


@pytest.mark.asyncio
async def test_process_batch_skips_sample_when_question_generation_fails(
    monkeypatch, tmp_path
):
    async def _fake_question(*args, **kwargs):
        return None

    async def _fake_rewrite(*args, **kwargs):
        raise AssertionError(
            "rewrite should not be called when question generation fails"
        )

    monkeypatch.setattr(
        "SeedDataGen.generator.phase_dog_instruct._generate_user_question",
        _fake_question,
    )
    monkeypatch.setattr(
        "SeedDataGen.generator.phase_dog_instruct._rewrite_assistant_answer",
        _fake_rewrite,
    )

    output_file = tmp_path / "dog_instruct_skip.jsonl"
    next_id = await _process_batch(
        client=None,
        cfg=_cfg(),
        model_id="model-id",
        samples=[_sample()],
        next_row_id=0,
        output_file=str(output_file),
        summary_map={},
    )

    assert next_id == 0
    assert not output_file.exists()
