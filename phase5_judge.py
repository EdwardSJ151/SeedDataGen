"""
Phase 5: LLM Judge Scoring + Average Score Filter

For each conversation from Phase 4, call the LLM with JUDGE_PROMPT,
parse the five scores, compute the average, and keep only rows where
avg > JUDGE_MIN_AVG_SCORE.

Output JSONL adds: input_id, scores, avg_score
origin_id is preserved from input unchanged.
"""

import argparse
import asyncio
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
from tqdm import tqdm

from SeedDataGen.config import (
    VLLM_BASE_URL,
    VLLM_API_KEY,
    JUDGE_TEMPERATURE,
    JUDGE_TOP_P,
    JUDGE_MAX_TOKENS,
    JUDGE_MIN_AVG_SCORE,
    BATCH_SIZE,
    MAX_CONCURRENT,
    PHASE4_OUTPUT,
    PHASE5_OUTPUT,
    STOP_STRINGS,
)
from SeedDataGen.prompts import JUDGE_PROMPT
from SeedDataGen.utils import (
    iter_jsonl_batches,
    get_last_processed_id,
    get_max_int_field,
    write_jsonl_batch,
    count_jsonl_lines,
    format_conversation_for_judge,
    parse_judge_scores,
)

client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)


# -----------------------------------------------------------------------
# LLM call
# -----------------------------------------------------------------------
async def _judge_conversation(
    model_id: str,
    sample_text: str,
    messages: List[Dict[str, str]],
) -> Optional[str]:
    conv_str = format_conversation_for_judge(messages)
    prompt = JUDGE_PROMPT.format(sample_text=sample_text, conversation=conv_str)
    try:
        resp = await client.chat.completions.create(
            model=model_id,
            messages=[{"role": "system", "content": prompt}],
            temperature=JUDGE_TEMPERATURE,
            top_p=JUDGE_TOP_P,
            max_tokens=JUDGE_MAX_TOKENS,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
                "stop": STOP_STRINGS,
            },
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"Phase5 LLM error: {e}")
        return None


# -----------------------------------------------------------------------
# Batch processing
# -----------------------------------------------------------------------
async def _process_batch(
    model_id: str,
    batch: List[Dict[str, Any]],
    next_id: int,
    output_file: str,
) -> int:
    sem = asyncio.Semaphore(MAX_CONCURRENT)

    async def _one(item: Dict[str, Any]):
        async with sem:
            return await _judge_conversation(
                model_id, item["sample_text"], item["messages"]
            )

    raw_outputs = await asyncio.gather(*[_one(it) for it in batch])

    rows: List[Dict[str, Any]] = []
    max_input_id = -1
    for item, raw in zip(batch, raw_outputs):
        if item["id"] > max_input_id:
            max_input_id = item["id"]
        if raw is None:
            continue
        scores = parse_judge_scores(raw)
        if scores is None:
            continue
        avg = sum(scores) / len(scores)
        if avg <= JUDGE_MIN_AVG_SCORE:
            continue
        item["input_id"] = item["id"]
        item["id"] = next_id
        item["scores"] = {
            "fidelidade": scores[0],
            "correcao": scores[1],
            "clareza": scores[2],
            "coerencia": scores[3],
            "diversidade": scores[4],
        }
        item["avg_score"] = round(avg, 2)
        rows.append(item)
        next_id += 1

    if rows:
        write_jsonl_batch(output_file, rows)
    return next_id, max_input_id


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
async def main(
    input_file: str = PHASE4_OUTPUT,
    output_file: str = PHASE5_OUTPUT,
    batch_size: int = BATCH_SIZE,
):
    models = await client.models.list()
    model_id = models.data[0].id
    print(f"Phase 5 — model: {model_id}")

    last_out_id = get_last_processed_id(output_file)
    next_id = last_out_id + 1 if last_out_id >= 0 else 0

    last_input_id = get_max_int_field(output_file, "input_id")
    resume_from_input = last_input_id + 1 if last_input_id >= 0 else 0

    total = count_jsonl_lines(input_file)
    pbar = tqdm(total=total, desc="Phase 5: LLM judge")

    for batch in iter_jsonl_batches(
        input_file,
        batch_size=batch_size,
        start_from_id=resume_from_input,
        required_fields=["sample_text", "messages"],
    ):
        next_id, _ = await _process_batch(model_id, batch, next_id, output_file)
        pbar.update(len(batch))

    pbar.close()
    print(f"Phase 5 done — kept conversations with avg > {JUDGE_MIN_AVG_SCORE} → {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 5: LLM judge")
    parser.add_argument("--input", default=PHASE4_OUTPUT)
    parser.add_argument("--output", default=PHASE5_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    asyncio.run(main(args.input, args.output, args.batch_size))
