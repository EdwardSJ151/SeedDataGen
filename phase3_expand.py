"""
Phase 3: Conversation Initialization + Expansion

For each QA pair from Phase 2:
  1. Initialize a conversation with the QA pair (Step 3 from plan).
  2. Expand by generating N_USER_TURNS extra user→assistant rounds (Step 4).

Each user/assistant turn is a separate async vLLM call.

Output JSONL fields:
    id, input_id, origin_id, sample_id, sample_text, messages
"""

import argparse
import asyncio
import random
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
from tqdm import tqdm

from SeedDataGen.config import (
    VLLM_BASE_URL,
    VLLM_API_KEY,
    USER_TURN_TEMPERATURE,
    USER_TURN_TOP_P,
    USER_TURN_MAX_TOKENS,
    ASSISTANT_TURN_TEMPERATURE,
    ASSISTANT_TURN_TOP_P,
    ASSISTANT_TURN_MAX_TOKENS,
    N_USER_TURNS_MIN,
    N_USER_TURNS_MAX,
    BATCH_SIZE,
    MAX_CONCURRENT,
    PHASE2_OUTPUT,
    PHASE3_OUTPUT,
    STOP_STRINGS,
)
from SeedDataGen.prompts import USER_TURN_PROMPT, ASSISTANT_TURN_PROMPT
from SeedDataGen.utils import (
    iter_jsonl_batches,
    get_last_processed_id,
    get_max_int_field,
    write_jsonl_batch,
    count_jsonl_lines,
    format_conversation_history,
)

client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)


# LLM helpers
async def _llm_call(
    model_id: str,
    prompt_text: str,
    *,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> Optional[str]:
    try:
        resp = await client.chat.completions.create(
            model=model_id,
            messages=[{"role": "system", "content": prompt_text}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
                "stop": STOP_STRINGS,
            },
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"Phase3 LLM error: {e}")
        return None


async def generate_user_turn(
    model_id: str,
    sample_text: str,
    messages: List[Dict[str, str]],
) -> Optional[str]:
    history_str = format_conversation_history(messages)
    prompt = USER_TURN_PROMPT.format(
        sample_text=sample_text,
        conversation_history=history_str,
    )
    return await _llm_call(
        model_id,
        prompt,
        temperature=USER_TURN_TEMPERATURE,
        top_p=USER_TURN_TOP_P,
        max_tokens=USER_TURN_MAX_TOKENS,
    )


async def generate_assistant_turn(
    model_id: str,
    sample_text: str,
    messages: List[Dict[str, str]],
) -> Optional[str]:
    history_str = format_conversation_history(messages)
    prompt = ASSISTANT_TURN_PROMPT.format(
        sample_text=sample_text,
        conversation_history=history_str,
    )
    return await _llm_call(
        model_id,
        prompt,
        temperature=ASSISTANT_TURN_TEMPERATURE,
        top_p=ASSISTANT_TURN_TOP_P,
        max_tokens=ASSISTANT_TURN_MAX_TOKENS,
    )


# Single conversation expansion
async def expand_conversation(
    model_id: str,
    sample_text: str,
    qa: Dict[str, str],
) -> Optional[List[Dict[str, str]]]:
    """
    Init from QA pair, then add N random user→assistant rounds.
    Returns the full message list or None on failure.
    """
    messages: List[Dict[str, str]] = [
        {"role": "user", "content": qa["question"]},
        {"role": "assistant", "content": qa["answer"]},
    ]

    n_turns = random.randint(N_USER_TURNS_MIN, N_USER_TURNS_MAX)

    for _ in range(n_turns):
        user_msg = await generate_user_turn(model_id, sample_text, messages)
        if not user_msg:
            break
        messages.append({"role": "user", "content": user_msg})

        asst_msg = await generate_assistant_turn(model_id, sample_text, messages)
        if not asst_msg:
            break
        messages.append({"role": "assistant", "content": asst_msg})

    return messages


# Batch processing
async def _process_batch(
    model_id: str,
    batch: List[Dict[str, Any]],
    next_id: int,
    output_file: str,
) -> int:
    sem = asyncio.Semaphore(MAX_CONCURRENT)

    async def _one(item: Dict[str, Any]):
        async with sem:
            return await expand_conversation(
                model_id,
                item["sample_text"],
                {"question": item["question"], "answer": item["answer"]},
            )

    results = await asyncio.gather(*[_one(it) for it in batch])

    rows: List[Dict[str, Any]] = []
    max_input_id = -1
    for item, msgs in zip(batch, results):
        if item["id"] > max_input_id:
            max_input_id = item["id"]
        if msgs is None or len(msgs) < 4:
            continue
        rows.append({
            "id": next_id,
            "input_id": item["id"],
            "origin_id": item["origin_id"],
            "sample_id": item["sample_id"],
            "sample_text": item["sample_text"],
            "messages": msgs,
        })
        next_id += 1

    n_kept = len(rows)
    if rows:
        write_jsonl_batch(output_file, rows)
    return next_id, max_input_id, n_kept


# Main
async def main(
    input_file: str = PHASE2_OUTPUT,
    output_file: str = PHASE3_OUTPUT,
    batch_size: int = BATCH_SIZE,
):
    models = await client.models.list()
    model_id = models.data[0].id
    print(f"Phase 3 — model: {model_id}")

    last_out_id = get_last_processed_id(output_file)
    next_id = last_out_id + 1 if last_out_id >= 0 else 0

    last_input_id = get_max_int_field(output_file, "input_id")
    resume_from_input = last_input_id + 1 if last_input_id >= 0 else 0

    total = count_jsonl_lines(input_file)
    pbar = tqdm(total=total, desc="Phase 3: expand conversations")

    total_seen = 0
    total_kept = 0
    for batch in iter_jsonl_batches(
        input_file,
        batch_size=batch_size,
        start_from_id=resume_from_input,
        required_fields=["sample_text", "question", "answer"],
    ):
        total_seen += len(batch)
        next_id, _, n_kept = await _process_batch(model_id, batch, next_id, output_file)
        total_kept += n_kept
        pbar.update(len(batch))

    pbar.close()
    dropped = total_seen - total_kept
    print(
        f"Phase 3 done — kept {total_kept}, dropped {dropped} "
        f"(from {total_seen} input QA rows) → {output_file}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3: conversation expansion")
    parser.add_argument("--input", default=PHASE2_OUTPUT)
    parser.add_argument("--output", default=PHASE3_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    asyncio.run(main(args.input, args.output, args.batch_size))
