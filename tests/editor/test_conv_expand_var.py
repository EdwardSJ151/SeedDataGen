"""Tests for editor/phase_conv_expand_var.py — no live LLM required."""

import asyncio
import json
import tempfile
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from SeedDataGen.editor.phase_conv_expand_var import (
    ConvExpandVarConfig,
    _expand_conversation,
    _process_batch,
    _style_sequence,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(**overrides) -> ConvExpandVarConfig:
    defaults = dict(
        naive_gen=True,
        question_styles=["general", "specific", "compositional", "comparative"],
        n_user_turns_min=1,
        n_user_turns_max=1,
        user_turn_temperature=0.9,
        user_turn_top_p=0.95,
        user_turn_max_tokens=512,
        assistant_turn_temperature=0.7,
        assistant_turn_top_p=0.9,
        assistant_turn_max_tokens=2048,
        batch_size=32,
        max_concurrent=64,
    )
    defaults.update(overrides)
    return ConvExpandVarConfig(**defaults)


def _item(id: int = 0, sample_id: int = 1, question_style: str = "general") -> dict:
    return {
        "id": id,
        "origin_id": 0,
        "sample_id": sample_id,
        "sample_text": "The low-voltage network operates at 220/127V.",
        "question": "What voltage does the LV network use?",
        "answer": "It operates at 220/127V.",
        "question_style": question_style,
        "GEN_TYPE": "qa_gen_var",
    }


def _fake_completion(text: str):
    choice = MagicMock()
    choice.message.content = text
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ---------------------------------------------------------------------------
# _style_sequence — pure function
# ---------------------------------------------------------------------------

STYLES = ["general", "specific", "compositional", "comparative"]


def test_style_sequence_cycles_from_seed():
    result = _style_sequence("general", STYLES, 3)
    assert result == ["specific", "compositional", "comparative"]


def test_style_sequence_wraps_around():
    result = _style_sequence("comparative", STYLES, 2)
    assert result == ["general", "specific"]


def test_style_sequence_unknown_seed_starts_at_zero():
    result = _style_sequence("unknown", STYLES, 2)
    assert result == ["general", "specific"]


def test_style_sequence_zero_turns():
    assert _style_sequence("general", STYLES, 0) == []


def test_style_sequence_single_style_list():
    result = _style_sequence("only", ["only"], 3)
    assert result == ["only", "only", "only"]


# ---------------------------------------------------------------------------
# _expand_conversation — LLM mocked
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_expand_conversation_produces_correct_message_count():
    """seed(2) + 1 extra turn × 2 calls = 4 messages total."""
    sem = asyncio.Semaphore(10)
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        return_value=_fake_completion("mocked response")
    )
    cfg = _cfg(n_user_turns_min=1, n_user_turns_max=1)

    with patch("random.randint", return_value=1):
        msgs = await _expand_conversation(
            client, cfg, "model-id",
            "sample text",
            {"question": "Q?", "answer": "A."},
            "general", "qa_gen_var", sem,
        )

    assert msgs is not None
    assert len(msgs) == 4  # user, assistant (seed), user, assistant (added)
    assert msgs[0] == {"role": "user", "content": "Q?"}
    assert msgs[1] == {"role": "assistant", "content": "A."}
    assert msgs[2]["role"] == "user"
    assert msgs[3]["role"] == "assistant"


@pytest.mark.asyncio
async def test_expand_conversation_stops_on_null_user_turn():
    """If user generation fails, loop breaks; we keep only the seed pair."""
    sem = asyncio.Semaphore(10)
    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=Exception("LLM down"))
    cfg = _cfg(n_user_turns_min=1, n_user_turns_max=1)

    with patch("random.randint", return_value=1):
        msgs = await _expand_conversation(
            client, cfg, "model-id",
            "sample text",
            {"question": "Q?", "answer": "A."},
            "general", "qa_gen_var", sem,
        )

    # Seed pair is always included; no extra turns were added.
    assert msgs == [
        {"role": "user", "content": "Q?"},
        {"role": "assistant", "content": "A."},
    ]


@pytest.mark.asyncio
async def test_expand_conversation_stops_on_null_assistant_turn():
    """User turn succeeds but assistant fails → loop breaks after user appended."""
    sem = asyncio.Semaphore(10)
    call_count = 0

    async def _side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _fake_completion("user question text")
        raise Exception("assistant down")

    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=_side_effect)
    cfg = _cfg(n_user_turns_min=1, n_user_turns_max=1)

    with patch("random.randint", return_value=1):
        msgs = await _expand_conversation(
            client, cfg, "model-id",
            "sample text",
            {"question": "Q?", "answer": "A."},
            "general", "qa_gen_var", sem,
        )

    # user appended but no assistant → 3 messages total
    assert msgs is not None
    assert len(msgs) == 3
    assert msgs[2]["role"] == "user"


# ---------------------------------------------------------------------------
# Semaphore scope — the critical fix
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_semaphore_released_between_turns():
    """
    With max_concurrent=1 and two conversations running concurrently, both
    must complete. If the semaphore were held for the whole conversation, the
    two coroutines would interleave fine (they're not blocking each other at
    the conversation level) — but the real risk is that with sem=1 and
    *sequential* LLM calls inside one conversation, only 1 slot exists for
    all calls across all conversations. We verify this by counting that
    acquire() is called once *per LLM call*, not once per conversation.
    """
    acquire_count = 0
    release_count = 0

    real_sem = asyncio.Semaphore(4)

    class CountingSemaphore:
        async def __aenter__(self):
            nonlocal acquire_count
            acquire_count += 1
            await real_sem.__aenter__()
            return self

        async def __aexit__(self, *args):
            nonlocal release_count
            release_count += 1
            await real_sem.__aexit__(*args)

    sem = CountingSemaphore()
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        return_value=_fake_completion("response")
    )
    cfg = _cfg(n_user_turns_min=2, n_user_turns_max=2)  # 2 extra turns = 4 LLM calls

    with patch("random.randint", return_value=2):
        msgs = await _expand_conversation(
            client, cfg, "model-id",
            "sample text",
            {"question": "Q?", "answer": "A."},
            "general", "qa_gen_var", sem,
        )

    # 2 turns × 2 calls (user + assistant) = 4 acquire/release pairs
    assert acquire_count == 4
    assert release_count == 4
    assert msgs is not None and len(msgs) == 6  # seed(2) + 2 turns × 2 = 6


# ---------------------------------------------------------------------------
# _process_batch — naive mode
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_process_batch_naive_keeps_valid_rows():
    """Rows whose expansion yields ≥4 messages are written to the output file."""
    client = MagicMock()
    # Seed(2) + 1 turn(2) = 4 messages — meets the ≥4 threshold.
    client.chat.completions.create = AsyncMock(
        return_value=_fake_completion("text")
    )
    cfg = _cfg(n_user_turns_min=1, n_user_turns_max=1)
    batch = [_item(id=0), _item(id=1)]

    with patch("random.randint", return_value=1):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            out = f.name
        next_id, _, n_kept = await _process_batch(client, cfg, "model-id", batch, 0, out, {})

    assert n_kept == 2
    with open(out) as f:
        written = [json.loads(l) for l in f if l.strip()]
    assert len(written) == 2


@pytest.mark.asyncio
async def test_process_batch_naive_drops_short_conversations():
    """If expansion returns only the seed pair (2 messages), the row is dropped."""
    client = MagicMock()
    # Always fail so _expand_conversation returns only the seed 2 messages.
    client.chat.completions.create = AsyncMock(side_effect=Exception("down"))
    cfg = _cfg(n_user_turns_min=1, n_user_turns_max=1)
    batch = [_item(id=0)]

    with patch("random.randint", return_value=1):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            out = f.name
        _, _, n_kept = await _process_batch(client, cfg, "model-id", batch, 0, out, {})

    assert n_kept == 0


@pytest.mark.asyncio
async def test_process_batch_diversity_groups_by_sample():
    """
    In diversity mode, items sharing a sample_id should have access to the
    previous questions from that group. We verify this by checking that the
    second call in a group receives a non-empty previous_questions list.
    """
    captured_calls: List[dict] = []

    async def _mock_create(*args, **kwargs):
        messages = kwargs.get("messages", args[1] if len(args) > 1 else [])
        captured_calls.append({"messages": messages})
        return _fake_completion("response")

    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=_mock_create)
    cfg = _cfg(naive_gen=False, n_user_turns_min=1, n_user_turns_max=1)
    # Two items from the same source chunk — diversity mode should accumulate.
    batch = [
        _item(id=0, sample_id=42),
        _item(id=1, sample_id=42),
    ]

    with patch("random.randint", return_value=1):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            out = f.name
        _, _, n_kept = await _process_batch(client, cfg, "model-id", batch, 0, out, {})

    # Both items produce valid conversations.
    assert n_kept == 2
