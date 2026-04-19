"""
Phase: qa_gen

Streams documents from the configured HuggingFace dataset, calls the LLM with
QA_GENERATION_PROMPT for each document, parses the output into individual QA
pairs and writes one QARow per pair.

Role:   GENERATOR
Input:  HuggingFace dataset (streaming)
Output: QARow  (id, origin_id, sample_id, sample_text, question, answer)

origin_id equals id at generation time and is preserved unchanged through all
downstream phases.
"""

import asyncio
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
from pydantic_settings import BaseSettings, SettingsConfigDict
from tqdm import tqdm

from SeedDataGen.base_phase import Phase, PhaseRole
from SeedDataGen.config import (
    DATASET_ID,
    DATASET_MAX_CHARS,
    DATASET_MIN_CHARS,
    DATASET_SPLIT,
    DATASET_SUBSET,
    DATASET_TEXT_FIELD,
    NUM_ROWS,
    STOP_STRINGS,
    VLLM_API_KEY,
    VLLM_BASE_URL,
)
from SeedDataGen.prompts import QA_GENERATION_PROMPT
from SeedDataGen.registry import register
from SeedDataGen.schemas import QARow
from SeedDataGen.utils import get_last_processed_id, get_max_int_field, write_jsonl_batch, parse_qa_pairs


class QAGenConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="QA_GEN_", env_file=".env", extra="ignore")

    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 2048
    batch_size: int = 32
    max_concurrent: int = 64


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
async def _generate_qa(
    client: AsyncOpenAI,
    cfg: QAGenConfig,
    sample_text: str,
) -> Optional[str]:
    prompt = QA_GENERATION_PROMPT.format(sample_text=sample_text)
    try:
        resp = await client.chat.completions.create(
            model=(await client.models.list()).data[0].id,
            messages=[{"role": "system", "content": prompt}],
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_tokens,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
                "stop": STOP_STRINGS,
            },
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        print(f"[qa_gen] LLM error: {e}")
        return None


# Batch processing
async def _process_batch(
    client: AsyncOpenAI,
    cfg: QAGenConfig,
    model_id: str,
    samples: List[Dict[str, Any]],
    sample_id_start: int,
    next_row_id: int,
    output_file: str,
) -> int:
    sem = asyncio.Semaphore(cfg.max_concurrent)

    async def _gen(sample: Dict[str, Any]) -> Optional[str]:
        async with sem:
            prompt = QA_GENERATION_PROMPT.format(sample_text=sample["sample_text"])
            try:
                resp = await client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    max_tokens=cfg.max_tokens,
                    extra_body={
                        "chat_template_kwargs": {"enable_thinking": False},
                        "stop": STOP_STRINGS,
                    },
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                print(f"[qa_gen] LLM error: {e}")
                return None

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


# Phase class
@register
class QAGenPhase(Phase):
    name = "qa_gen"
    role = PhaseRole.GENERATOR
    input_schema = None
    output_schema = QARow

    def describe_prompts(self):
        cfg = QAGenConfig()
        prompt = QA_GENERATION_PROMPT.format(sample_text="[SAMPLE_TEXT]")
        return [("qa_gen / main prompt (user)", prompt)]

    async def run(self, input_file: str, output_file: str, **kwargs) -> None:
        cfg = QAGenConfig()
        num_rows: int = kwargs.get("num_rows", NUM_ROWS)
        batch_size: int = kwargs.get("batch_size", cfg.batch_size)

        client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
        models = await client.models.list()
        model_id = models.data[0].id
        print(f"[qa_gen] model: {model_id}")

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
        pbar = tqdm(desc="[qa_gen] generating", initial=next_row_id, total=num_rows)

        while next_row_id < num_rows:
            samples = _next_valid_samples(ds_iter, batch_size)
            if not samples:
                print("[qa_gen] Dataset exhausted.")
                break

            next_row_id = await _process_batch(
                client, cfg, model_id, samples, sample_id_counter, next_row_id, output_file
            )
            sample_id_counter += len(samples)
            pbar.n = min(next_row_id, num_rows)
            pbar.refresh()

        pbar.close()
        print(f"[qa_gen] done — {next_row_id} QA rows → {output_file}")
