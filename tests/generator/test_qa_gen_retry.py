import json

import pytest

from SeedDataGen.generator.phase_qa_gen import QAGenConfig, _process_batch


def _cfg(**overrides) -> QAGenConfig:
    defaults = dict(
        temperature=0.7,
        top_p=0.9,
        max_tokens=1024,
        batch_size=32,
        max_concurrent=8,
    )
    defaults.update(overrides)
    return QAGenConfig(**defaults)


def _sample(hf_row_id: str = "42") -> dict:
    return {
        "hf_row_id": hf_row_id,
        "sample_text": {"text": "Texto técnico.", "document_name": "ND-1"},
        "prompt_text": "[Chunk 1 | ND-1]\nTexto técnico.",
    }


def _fake_pairs(n: int):
    return [{"question": f"Q{i}", "answer": f"A{i}"} for i in range(n)]


# ---------------------------------------------------------------------------
# question_style = str(pair_index) on normal generation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_process_batch_assigns_pair_index(monkeypatch, tmp_path):
    """Each emitted row gets question_style = str(i) matching its position."""
    monkeypatch.setattr(
        "SeedDataGen.generator.phase_qa_gen.parse_qa_pairs",
        lambda raw: _fake_pairs(3),
    )

    async def _fake_gen(client, cfg, model_id, samples):
        return ["raw"] * len(samples)

    output_file = str(tmp_path / "out.jsonl")

    # Patch the inner LLM call by monkeypatching the semaphore-wrapped coroutine
    import asyncio
    original_gather = asyncio.gather

    async def _patched_create(**kwargs):
        class Choice:
            message = type("M", (), {"content": "raw"})()
        class Resp:
            choices = [Choice()]
        return Resp()

    class FakeClient:
        class chat:
            class completions:
                @staticmethod
                async def create(**kwargs):
                    return _patched_create.__wrapped__()

    # Simpler: monkeypatch parse_qa_pairs and stub the client's create method
    created_calls = []

    class FakeCompletion:
        choices = [type("C", (), {"message": type("M", (), {"content": "raw"})()})()]

    class FakeCompletions:
        async def create(self, **kwargs):
            created_calls.append(kwargs)
            return FakeCompletion()

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient2:
        chat = FakeChat()

    await _process_batch(
        client=FakeClient2(),
        cfg=_cfg(),
        model_id="model",
        samples=[_sample("10")],
        next_row_id=0,
        output_file=output_file,
        retry_pairs=None,
    )

    rows = [json.loads(l) for l in open(output_file)]
    assert len(rows) == 3
    assert rows[0]["question_style"] == "0"
    assert rows[1]["question_style"] == "1"
    assert rows[2]["question_style"] == "2"


@pytest.mark.asyncio
async def test_process_batch_retry_pairs_filters_by_index(monkeypatch, tmp_path):
    """In retry mode only pairs whose index is in retry_pairs are emitted."""
    monkeypatch.setattr(
        "SeedDataGen.generator.phase_qa_gen.parse_qa_pairs",
        lambda raw: _fake_pairs(3),
    )

    class FakeCompletion:
        choices = [type("C", (), {"message": type("M", (), {"content": "raw"})()})()]

    class FakeCompletions:
        async def create(self, **kwargs):
            return FakeCompletion()

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    output_file = str(tmp_path / "out.jsonl")
    # Only retry pair index 1 from chunk "10"
    retry_pairs = {("10", "1")}

    await _process_batch(
        client=FakeClient(),
        cfg=_cfg(),
        model_id="model",
        samples=[_sample("10")],
        next_row_id=0,
        output_file=output_file,
        retry_pairs=retry_pairs,
    )

    rows = [json.loads(l) for l in open(output_file)]
    assert len(rows) == 1
    assert rows[0]["question_style"] == "1"
    assert rows[0]["question"] == "Q1"


@pytest.mark.asyncio
async def test_process_batch_retry_multiple_indices(monkeypatch, tmp_path):
    """Multiple failed pair indices from the same chunk are all retried."""
    monkeypatch.setattr(
        "SeedDataGen.generator.phase_qa_gen.parse_qa_pairs",
        lambda raw: _fake_pairs(4),
    )

    class FakeCompletion:
        choices = [type("C", (), {"message": type("M", (), {"content": "raw"})()})()]

    class FakeCompletions:
        async def create(self, **kwargs):
            return FakeCompletion()

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    output_file = str(tmp_path / "out.jsonl")
    # Pairs 0 and 3 failed; 1 and 2 passed
    retry_pairs = {("10", "0"), ("10", "3")}

    await _process_batch(
        client=FakeClient(),
        cfg=_cfg(),
        model_id="model",
        samples=[_sample("10")],
        next_row_id=5,
        output_file=output_file,
        retry_pairs=retry_pairs,
    )

    rows = [json.loads(l) for l in open(output_file)]
    assert len(rows) == 2
    styles = {r["question_style"] for r in rows}
    assert styles == {"0", "3"}
    # IDs start from next_row_id=5
    assert rows[0]["id"] == 5
    assert rows[1]["id"] == 6


@pytest.mark.asyncio
async def test_process_batch_retry_pairs_different_chunks(monkeypatch, tmp_path):
    """When retrying multiple chunks, each chunk's pair indices are filtered independently."""
    monkeypatch.setattr(
        "SeedDataGen.generator.phase_qa_gen.parse_qa_pairs",
        lambda raw: _fake_pairs(3),
    )

    class FakeCompletion:
        choices = [type("C", (), {"message": type("M", (), {"content": "raw"})()})()]

    class FakeCompletions:
        async def create(self, **kwargs):
            return FakeCompletion()

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    output_file = str(tmp_path / "out.jsonl")
    # chunk "10" retries pair 2; chunk "11" retries pair 0
    retry_pairs = {("10", "2"), ("11", "0")}

    await _process_batch(
        client=FakeClient(),
        cfg=_cfg(),
        model_id="model",
        samples=[_sample("10"), _sample("11")],
        next_row_id=0,
        output_file=output_file,
        retry_pairs=retry_pairs,
    )

    rows = [json.loads(l) for l in open(output_file)]
    assert len(rows) == 2
    by_chunk = {r["sample_id"][0]: r["question_style"] for r in rows}
    assert by_chunk["10"] == "2"
    assert by_chunk["11"] == "0"
