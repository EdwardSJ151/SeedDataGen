"""
Phase 1B: Answer Rewrite (optional)

For each QA row from Phase 1, asks the model to rewrite the answer to be
more information-rich using the source text — without adding anything that
isn't already there.

Enabled via ENABLE_ANSWER_REWRITE=true in config / .env.

Input:  Phase 1 JSONL  (id, origin_id, sample_id, sample_text, question, answer)
Output: Phase 1B JSONL (same schema; answer field replaced with rewritten version)
"""

import argparse
import asyncio
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
from tqdm import tqdm

from SeedDataGen.config import (
    VLLM_BASE_URL,
    VLLM_API_KEY,
    ANSWER_REWRITE_TEMPERATURE,
    ANSWER_REWRITE_TOP_P,
    ANSWER_REWRITE_MAX_TOKENS,
    BATCH_SIZE,
    MAX_CONCURRENT,
    PHASE1_OUTPUT,
    PHASE1B_OUTPUT,
    STOP_STRINGS,
)
from SeedDataGen.prompts import ANSWER_REWRITE_PROMPT
from SeedDataGen.utils import (
    iter_jsonl_batches,
    write_jsonl_batch,
    count_jsonl_lines,
    get_last_processed_id,
    get_max_int_field,
)

client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)


async def _rewrite_answer(
    model_id: str,
    sample_text: str,
    question: str,
    answer: str,
) -> Optional[str]:
    prompt = ANSWER_REWRITE_PROMPT.format(
        sample_text=sample_text,
        question=question,
        answer=answer,
    )
    try:
        resp = await client.chat.completions.create(
            model=model_id,
            messages=[{"role": "system", "content": prompt}],
            temperature=ANSWER_REWRITE_TEMPERATURE,
            top_p=ANSWER_REWRITE_TOP_P,
            max_tokens=ANSWER_REWRITE_MAX_TOKENS,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
                "stop": STOP_STRINGS,
            },
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"Phase1B LLM error: {e}")
        return None


async def _process_batch(
    model_id: str,
    batch: List[Dict[str, Any]],
    next_id: int,
    output_file: str,
) -> int:
    sem = asyncio.Semaphore(MAX_CONCURRENT)

    async def _one(item: Dict[str, Any]):
        async with sem:
            return await _rewrite_answer(
                model_id,
                item["sample_text"],
                item["question"],
                item["answer"],
            )

    results = await asyncio.gather(*[_one(it) for it in batch])

    rows: List[Dict[str, Any]] = []
    max_input_id = -1
    for item, rewritten in zip(batch, results):
        if item["id"] > max_input_id:
            max_input_id = item["id"]
        row = dict(item)
        row["input_id"] = row["id"]
        row["id"] = next_id
        if rewritten:
            row["answer"] = rewritten
        rows.append(row)
        next_id += 1

    if rows:
        write_jsonl_batch(output_file, rows)
    return next_id, max_input_id


async def main(
    input_file: str = PHASE1_OUTPUT,
    output_file: str = PHASE1B_OUTPUT,
    batch_size: int = BATCH_SIZE,
):
    models = await client.models.list()
    model_id = models.data[0].id
    print(f"Phase 1B — model: {model_id}")

    last_out_id = get_last_processed_id(output_file)
    next_id = last_out_id + 1 if last_out_id >= 0 else 0

    last_input_id = get_max_int_field(output_file, "input_id")
    resume_from_input = last_input_id + 1 if last_input_id >= 0 else 0

    total = count_jsonl_lines(input_file)
    pbar = tqdm(total=total, desc="Phase 1B: answer rewrite")

    total_seen = 0
    for batch in iter_jsonl_batches(
        input_file,
        batch_size=batch_size,
        start_from_id=resume_from_input,
        required_fields=["sample_text", "question", "answer"],
    ):
        total_seen += len(batch)
        next_id, _ = await _process_batch(model_id, batch, next_id, output_file)
        pbar.update(len(batch))

    pbar.close()
    print(f"Phase 1B done — {total_seen} answers rewritten → {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1B: answer rewrite")
    parser.add_argument("--input", default=PHASE1_OUTPUT)
    parser.add_argument("--output", default=PHASE1B_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    asyncio.run(main(args.input, args.output, args.batch_size))
