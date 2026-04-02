"""
Phase 1: QA Pair Generation

For each document streamed from the source dataset, call the LLM with the
QA_GENERATION_PROMPT and parse the output into individual QA pairs.

Output JSONL fields per row:
    id, origin_id, sample_id, sample_text, question, answer

origin_id is set equal to id here and stays unchanged through all phases.
"""

import argparse
import asyncio
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
from tqdm import tqdm

from SeedDataGen.config import (
    VLLM_BASE_URL,
    VLLM_API_KEY,
    DATASET_ID,
    DATASET_SUBSET,
    DATASET_SPLIT,
    DATASET_TEXT_FIELD,
    DATASET_MAX_CHARS,
    DATASET_MIN_CHARS,
    QA_TEMPERATURE,
    QA_TOP_P,
    QA_MAX_TOKENS,
    BATCH_SIZE,
    NUM_ROWS,
    MAX_CONCURRENT,
    PHASE1_OUTPUT,
    STOP_STRINGS,
)
from SeedDataGen.prompts import QA_GENERATION_PROMPT
from SeedDataGen.utils import (
    get_last_processed_id,
    get_max_int_field,
    write_jsonl_batch,
    parse_qa_pairs,
)

client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)


# Text helpers
def _normalize(s: str) -> str:
    return " ".join(s.split())


def _truncate(txt: str) -> str:
    t = _normalize(txt)
    if len(t) <= DATASET_MAX_CHARS:
        return t
    cut = t.rfind(".", 0, DATASET_MAX_CHARS)
    if cut == -1 or cut < DATASET_MIN_CHARS:
        return t[:DATASET_MAX_CHARS]
    return t[: cut + 1]


# Dataset streaming
def _stream_dataset():
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID, DATASET_SUBSET, split=DATASET_SPLIT, streaming=True)
    return iter(ds)


def _next_valid_samples(ds_iter, n: int) -> List[Dict[str, Any]]:
    """Pull up to *n* valid (long enough) samples from the stream."""
    out: List[Dict[str, Any]] = []
    for rec in ds_iter:
        txt = rec.get(DATASET_TEXT_FIELD, "")
        if not isinstance(txt, str):
            continue
        txt = _truncate(txt)
        if len(txt) < DATASET_MIN_CHARS:
            continue
        out.append({"sample_text": txt, "title": rec.get("title", "")})
        if len(out) >= n:
            break
    return out


# LLM call
async def _generate_qa_for_sample(
    model_id: str,
    sample_text: str,
) -> Optional[str]:
    prompt = QA_GENERATION_PROMPT.format(sample_text=sample_text)
    try:
        resp = await client.chat.completions.create(
            model=model_id,
            messages=[{"role": "system", "content": prompt}],
            temperature=QA_TEMPERATURE,
            top_p=QA_TOP_P,
            max_tokens=QA_MAX_TOKENS,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
                "stop": STOP_STRINGS,
            },
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        print(f"Phase1 LLM error: {e}")
        return None


# Batch processing
async def _process_batch(
    model_id: str,
    samples: List[Dict[str, Any]],
    sample_id_start: int,
    next_row_id: int,
    output_file: str,
) -> int:
    """
    Generate QA pairs for a batch of samples.
    Returns the updated next_row_id.
    """
    sem = asyncio.Semaphore(MAX_CONCURRENT)

    async def _gen(sample: Dict[str, Any]):
        async with sem:
            return await _generate_qa_for_sample(model_id, sample["sample_text"])

    raw_outputs = await asyncio.gather(*[_gen(s) for s in samples])

    rows: List[Dict[str, Any]] = []
    for idx, (sample, raw) in enumerate(zip(samples, raw_outputs)):
        if raw is None:
            continue
        qa_pairs = parse_qa_pairs(raw)
        sample_id = sample_id_start + idx
        for qa in qa_pairs:
            rows.append({
                "id": next_row_id,
                "origin_id": next_row_id,
                "sample_id": sample_id,
                "sample_text": sample["sample_text"],
                "question": qa["question"],
                "answer": qa["answer"],
            })
            next_row_id += 1

    if rows:
        write_jsonl_batch(output_file, rows)
    return next_row_id


# Main
async def main(
    output_file: str = PHASE1_OUTPUT,
    num_rows: int = NUM_ROWS,
    batch_size: int = BATCH_SIZE,
):
    models = await client.models.list()
    model_id = models.data[0].id
    print(f"Phase 1 — model: {model_id}")

    last_id = get_last_processed_id(output_file)
    next_row_id = last_id + 1 if last_id >= 0 else 0

    last_sample_id = get_max_int_field(output_file, "sample_id")
    samples_to_skip = last_sample_id + 1 if last_sample_id >= 0 else 0

    ds_iter = _stream_dataset()

    skipped = 0
    for rec in ds_iter:
        txt = rec.get(DATASET_TEXT_FIELD, "")
        if isinstance(txt, str) and len(_normalize(txt)) >= DATASET_MIN_CHARS:
            skipped += 1
            if skipped >= samples_to_skip:
                break

    sample_id_counter = samples_to_skip
    pbar = tqdm(desc="Phase 1: QA generation", initial=next_row_id, total=num_rows)

    while next_row_id < num_rows:
        samples = _next_valid_samples(ds_iter, batch_size)
        if not samples:
            print("Dataset exhausted.")
            break

        next_row_id = await _process_batch(
            model_id, samples, sample_id_counter, next_row_id, output_file,
        )
        sample_id_counter += len(samples)
        pbar.n = min(next_row_id, num_rows)
        pbar.refresh()

    pbar.close()
    print(f"Phase 1 done — {next_row_id} QA rows written to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1: QA generation")
    parser.add_argument("--output", default=PHASE1_OUTPUT)
    parser.add_argument("--num-rows", type=int, default=NUM_ROWS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    asyncio.run(main(args.output, args.num_rows, args.batch_size))
