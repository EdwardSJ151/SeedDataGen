"""
Phase: qa_gen

Streams documents from the configured HuggingFace dataset, calls the LLM with
QA_GENERATION_PROMPT for each document, parses the output into individual QA
pairs and writes one QARow per pair.

Role:   GENERATOR
Input:  HuggingFace dataset (streaming)
Output: QARow  (id, origin_id, sample_id, sample_text, question, answer, ...)

Row shape:
  - sample_id   : [hf_row_id]                  (list with the single source id)
  - sample_text : {str(hf_row_id): text}       (dict keyed by source id)
  - GEN_TYPE    : "qa_gen", num_chunks=1, doc_constraint=None

origin_id equals id at generation time and is preserved unchanged through all
downstream phases.  hf_row_id is read from DATASET_ID_FIELD, falling back to the
streaming index when that column is absent.
"""

import asyncio
import math
import os
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
from pydantic_settings import BaseSettings, SettingsConfigDict
from tqdm import tqdm

from SeedDataGen.base_phase import Phase, PhaseRole
from SeedDataGen.config import (
    DATASET_ID,
    DATASET_ID_FIELD,
    DATASET_MAX_CHARS,
    DATASET_MIN_CHARS,
    DATASET_SPLIT,
    DATASET_SUBSET,
    DATASET_TEXT_FIELD,
    NUM_ROWS,
    STOP_STRINGS,
    VLLM_API_KEY,
    VLLM_BASE_URL,
    get_dataset_doc_name_field,
    validate_pipeline_env,
)
from SeedDataGen.generator.prompts import QA_GENERATION_PROMPT
from SeedDataGen.registry import register
from SeedDataGen.schemas import QARow
from SeedDataGen.utils import (
    assert_hf_dataset_has_fields,
    format_sample_text_for_prompt,
    get_last_processed_id,
    get_processed_sample_ids,
    make_chunk_entry,
    parse_qa_pairs,
    require_hf_field,
    write_jsonl_batch,
)

GEN_TYPE = "qa_gen"


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
    max_chars = int(os.environ.get("DATASET_MAX_CHARS", DATASET_MAX_CHARS))
    min_chars = int(os.environ.get("DATASET_MIN_CHARS", DATASET_MIN_CHARS))
    t = _normalize(txt)
    if len(t) <= max_chars:
        return t
    cut = t.rfind(".", 0, max_chars)
    if cut == -1 or cut < min_chars:
        return t[:max_chars]
    return t[: cut + 1]


# Dataset streaming
def _stream_dataset():
    from datasets import load_dataset
    dataset_id = os.environ.get("DATASET_ID", DATASET_ID)
    dataset_subset = os.environ.get("DATASET_SUBSET", DATASET_SUBSET)
    dataset_split = os.environ.get("DATASET_SPLIT", DATASET_SPLIT)
    ds = load_dataset(dataset_id, dataset_subset, split=dataset_split, streaming=True)
    return iter(ds)


def _next_valid_samples(ds_iter, n: int, skip_ids: set) -> List[Dict[str, Any]]:
    text_field = os.environ.get("DATASET_TEXT_FIELD", DATASET_TEXT_FIELD)
    id_field = os.environ.get("DATASET_ID_FIELD", DATASET_ID_FIELD)
    doc_name_field = get_dataset_doc_name_field()
    min_chars = int(os.environ.get("DATASET_MIN_CHARS", DATASET_MIN_CHARS))
    out: List[Dict[str, Any]] = []
    for stream_idx, rec in _enumerate_global(ds_iter):
        txt = rec.get(text_field, "")
        if not isinstance(txt, str):
            continue
        txt = _truncate(txt)
        if len(txt) < min_chars:
            continue
        hf_row_id = str(require_hf_field(rec, id_field, row_label=f"row {stream_idx}"))
        if hf_row_id in skip_ids:
            continue
        document_name = str(require_hf_field(rec, doc_name_field, row_label=f"row {stream_idx}"))
        chunk_entry = make_chunk_entry(txt, document_name)
        out.append({
            "hf_row_id": hf_row_id,
            "sample_text": chunk_entry,
            "prompt_text": format_sample_text_for_prompt({hf_row_id: chunk_entry}),
        })
        if len(out) >= n:
            break
    return out


# Global stream counter so the streaming index is stable across batches.
_GLOBAL_STREAM_IDX = 0


def _enumerate_global(ds_iter):
    global _GLOBAL_STREAM_IDX
    for rec in ds_iter:
        yield _GLOBAL_STREAM_IDX, rec
        _GLOBAL_STREAM_IDX += 1


# Batch processing
async def _process_batch(
    client: AsyncOpenAI,
    cfg: QAGenConfig,
    model_id: str,
    samples: List[Dict[str, Any]],
    next_row_id: int,
    output_file: str,
) -> int:
    sem = asyncio.Semaphore(cfg.max_concurrent)

    async def _gen(sample: Dict[str, Any]) -> Optional[str]:
        async with sem:
            prompt = QA_GENERATION_PROMPT.format(sample_text=sample["prompt_text"])
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
    for sample, raw in zip(samples, raw_outputs):
        if raw is None:
            continue
        hf_row_id = sample["hf_row_id"]
        for qa in parse_qa_pairs(raw):
            rows.append({
                "id": next_row_id,
                "origin_id": next_row_id,
                "sample_id": [hf_row_id],
                "sample_text": {hf_row_id: sample["sample_text"]},
                "question": qa["question"],
                "answer": qa["answer"],
                "GEN_TYPE": GEN_TYPE,
                "num_chunks": 1,
                "doc_constraint": None,
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

    async def estimate(self, **kwargs) -> Optional[int]:
        """
        Count valid HF chunks without LLM calls.
        qa_gen produces 1–5 QA pairs per chunk, so exact row count depends on
        LLM output.  We return the LLM call count and print the row range.
        If NUM_ROWS is set (not exhaustive), we return NUM_ROWS as the cap.
        """
        num_rows = int(kwargs.get("num_rows", NUM_ROWS))
        exhaustive = num_rows < 0

        dataset_id = os.environ.get("DATASET_ID", DATASET_ID)
        dataset_subset = os.environ.get("DATASET_SUBSET", DATASET_SUBSET)
        dataset_split = os.environ.get("DATASET_SPLIT", DATASET_SPLIT)
        text_field = os.environ.get("DATASET_TEXT_FIELD", DATASET_TEXT_FIELD)
        id_field = os.environ.get("DATASET_ID_FIELD", DATASET_ID_FIELD)
        min_chars = int(os.environ.get("DATASET_MIN_CHARS", DATASET_MIN_CHARS))

        # If not exhaustive, the generator stops at num_rows QA rows regardless
        # of how many chunks it processes, so the answer is simply num_rows.
        if not exhaustive:
            print(f"  [qa_gen] capped at NUM_ROWS={num_rows:,} QA rows (1–5 pairs/chunk)")
            return num_rows

        from datasets import load_dataset
        ds = load_dataset(dataset_id, dataset_subset, split=dataset_split, streaming=True)
        print(f"  [qa_gen] scanning {dataset_id} for valid chunks...", flush=True)
        valid = 0
        for rec in ds:
            txt = rec.get(text_field, "")
            if not isinstance(txt, str):
                continue
            if len(" ".join(txt.split())) < min_chars:
                continue
            if rec.get(id_field) is None:
                continue
            valid += 1

        lo, hi = valid * 1, valid * 5
        print(f"  [qa_gen] {valid:,} valid chunks → {lo:,}–{hi:,} QA rows (1–5 pairs/chunk)")
        # Return midpoint estimate; caller prints it
        return valid * 3

    def describe_prompts(self):
        prompt = QA_GENERATION_PROMPT.format(sample_text="[SAMPLE_TEXT]")
        return [("qa_gen / main prompt (user)", prompt)]

    async def run(self, input_file: str, output_file: str, **kwargs) -> None:
        global _GLOBAL_STREAM_IDX
        _GLOBAL_STREAM_IDX = 0

        validate_pipeline_env()
        cfg = QAGenConfig()
        num_rows: int = kwargs.get("num_rows", NUM_ROWS)
        batch_size: int = kwargs.get("batch_size", cfg.batch_size)
        exhaustive = num_rows is not None and int(num_rows) < 0

        client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
        models = await client.models.list()
        model_id = models.data[0].id
        print(f"[qa_gen] model: {model_id}")

        last_id = get_last_processed_id(output_file)
        next_row_id = last_id + 1 if last_id >= 0 else 0
        skip_ids = get_processed_sample_ids(output_file)

        ds_iter = _stream_dataset()
        dataset_id = os.environ.get("DATASET_ID", DATASET_ID)
        id_field = os.environ.get("DATASET_ID_FIELD", DATASET_ID_FIELD)
        doc_name_field = get_dataset_doc_name_field()
        ds_iter = assert_hf_dataset_has_fields(
            ds_iter, [id_field, doc_name_field], dataset_id=dataset_id
        )

        total = None if exhaustive else num_rows
        pbar = tqdm(desc="[qa_gen] generating", initial=next_row_id, total=total)

        while exhaustive or next_row_id < num_rows:
            samples = _next_valid_samples(ds_iter, batch_size, skip_ids)
            if not samples:
                print("[qa_gen] Dataset exhausted.")
                break
            for s in samples:
                skip_ids.add(s["hf_row_id"])

            next_row_id = await _process_batch(
                client, cfg, model_id, samples, next_row_id, output_file
            )
            if total is not None:
                pbar.n = min(next_row_id, num_rows)
            else:
                pbar.n = next_row_id
            pbar.refresh()

        pbar.close()
        print(f"[qa_gen] done — {next_row_id} QA rows → {output_file}")
