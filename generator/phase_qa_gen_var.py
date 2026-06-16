"""
Phase: qa_gen_var

Style-constrained QA generation.  For each document, makes one LLM call per
configured style and extracts exactly one QA pair per call.  The number of QA
pairs produced per document equals the number of styles configured.

Available built-in styles live in QA_GEN_VAR_STYLE_INSTRUCTIONS
(SeedDataGen/generator/prompts.py).

Role:   GENERATOR
Input:  HuggingFace dataset (streaming)
Output: StyledQARow  — fully compatible with qa_filter and all downstream phases.

Row shape:
  - sample_id   : [hf_row_id]
  - sample_text : {str(hf_row_id): text}
  - GEN_TYPE    : "qa_gen_var", num_chunks=1, doc_constraint=None
  - question_style : the style that produced the pair
"""

import asyncio
import math
import os
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
from pydantic import field_validator
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
)
from SeedDataGen.generator.prompts import QA_GEN_VAR_STYLE_INSTRUCTIONS, QA_GEN_VAR_SYSTEM_PROMPT
from SeedDataGen.registry import register
from SeedDataGen.schemas import StyledQARow
from SeedDataGen.utils import (
    assert_hf_dataset_has_fields,
    get_last_processed_id,
    get_processed_sample_ids,
    parse_qa_pairs,
    require_hf_field,
    write_jsonl_batch,
)

GEN_TYPE = "qa_gen_var"


class QAGenVarConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="QA_GEN_VAR_", env_file=".env", extra="ignore", enable_decoding=False)

    question_styles: List[str] = ["general", "specific", "compositional", "comparative"]
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 512
    batch_size: int = 32
    max_concurrent: int = 64

    @field_validator("question_styles", mode="before")
    @classmethod
    def _parse_styles(cls, v):
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v


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


# Global stream counter so the streaming index is stable across batches.
_GLOBAL_STREAM_IDX = 0


def _enumerate_global(ds_iter):
    global _GLOBAL_STREAM_IDX
    for rec in ds_iter:
        yield _GLOBAL_STREAM_IDX, rec
        _GLOBAL_STREAM_IDX += 1


def _next_valid_samples(ds_iter, n: int, skip_ids: set) -> List[Dict[str, Any]]:
    text_field = os.environ.get("DATASET_TEXT_FIELD", DATASET_TEXT_FIELD)
    id_field = os.environ.get("DATASET_ID_FIELD", DATASET_ID_FIELD)
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
        out.append({"hf_row_id": hf_row_id, "sample_text": txt})
        if len(out) >= n:
            break
    return out


# LLM call — one style at a time
async def _generate_qa_for_style(
    client: AsyncOpenAI,
    cfg: QAGenVarConfig,
    model_id: str,
    sample_text: str,
    style: str,
) -> Optional[str]:
    style_instruction = QA_GEN_VAR_STYLE_INSTRUCTIONS.get(style)
    if style_instruction is None:
        print(f"[qa_gen_var] Unknown style '{style}'. Available: {list(QA_GEN_VAR_STYLE_INSTRUCTIONS)}")
        return None

    prompt = QA_GEN_VAR_SYSTEM_PROMPT.format(
        style_instruction=style_instruction,
        sample_text=sample_text,
    )
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
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"[qa_gen_var] LLM error (style={style}): {e}")
        return None


# Batch processing
async def _process_batch(
    client: AsyncOpenAI,
    cfg: QAGenVarConfig,
    model_id: str,
    samples: List[Dict[str, Any]],
    next_row_id: int,
    output_file: str,
) -> int:
    sem = asyncio.Semaphore(cfg.max_concurrent)

    tasks = []
    for sample in samples:
        for style in cfg.question_styles:
            tasks.append((sample, style))

    async def _gen(sample: Dict[str, Any], style: str) -> Optional[str]:
        async with sem:
            return await _generate_qa_for_style(client, cfg, model_id, sample["sample_text"], style)

    raw_outputs = await asyncio.gather(*[_gen(s, st) for s, st in tasks])

    rows: List[Dict[str, Any]] = []
    for (sample, style), raw in zip(tasks, raw_outputs):
        if raw is None:
            continue
        pairs = parse_qa_pairs(raw)
        if not pairs:
            continue
        qa = pairs[0]
        hf_row_id = sample["hf_row_id"]
        rows.append({
            "id": next_row_id,
            "origin_id": next_row_id,
            "sample_id": [hf_row_id],
            "sample_text": {hf_row_id: sample["sample_text"]},
            "question": qa["question"],
            "answer": qa["answer"],
            "question_style": style,
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
class QAGenVarPhase(Phase):
    name = "qa_gen_var"
    role = PhaseRole.GENERATOR
    input_schema = None
    output_schema = StyledQARow

    async def estimate(self, **kwargs) -> Optional[int]:
        """
        Count valid HF chunks without LLM calls.
        Each chunk produces exactly one QA row per style, so:
            rows = min(valid_chunks × n_styles, NUM_ROWS)
        Streams the full dataset when NUM_ROWS=-1 (exhaustive).
        """
        cfg = QAGenVarConfig()
        n_styles = len(cfg.question_styles)
        num_rows = int(kwargs.get("num_rows", NUM_ROWS))
        exhaustive = num_rows < 0

        dataset_id = os.environ.get("DATASET_ID", DATASET_ID)
        dataset_subset = os.environ.get("DATASET_SUBSET", DATASET_SUBSET)
        dataset_split = os.environ.get("DATASET_SPLIT", DATASET_SPLIT)
        text_field = os.environ.get("DATASET_TEXT_FIELD", DATASET_TEXT_FIELD)
        id_field = os.environ.get("DATASET_ID_FIELD", DATASET_ID_FIELD)
        min_chars = int(os.environ.get("DATASET_MIN_CHARS", DATASET_MIN_CHARS))

        # Stop streaming once we have enough chunks to fill NUM_ROWS.
        need_chunks = None if exhaustive else math.ceil(num_rows / max(n_styles, 1))

        from datasets import load_dataset
        ds = load_dataset(dataset_id, dataset_subset, split=dataset_split, streaming=True)

        print(f"  [qa_gen_var] scanning {dataset_id} for valid chunks...", flush=True)
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
            if need_chunks and valid >= need_chunks:
                break

        rows = valid * n_styles
        if not exhaustive:
            rows = min(rows, num_rows)

        print(
            f"  [qa_gen_var] {valid:,} valid chunks × {n_styles} styles"
            + (f" (capped at NUM_ROWS={num_rows:,})" if not exhaustive and rows < valid * n_styles else "")
        )
        return rows

    def describe_prompts(self):
        cfg = QAGenVarConfig()
        prompts = []
        for style in cfg.question_styles:
            style_instruction = QA_GEN_VAR_STYLE_INSTRUCTIONS.get(style, f"[UNKNOWN STYLE: {style}]")
            prompt = QA_GEN_VAR_SYSTEM_PROMPT.format(
                style_instruction=style_instruction,
                sample_text="[SAMPLE_TEXT]",
            )
            prompts.append((f"qa_gen_var / style={style} (user)", prompt))
        return prompts

    async def run(self, input_file: str, output_file: str, **kwargs) -> None:
        global _GLOBAL_STREAM_IDX
        _GLOBAL_STREAM_IDX = 0

        cfg = QAGenVarConfig()

        if not cfg.question_styles:
            raise ValueError("[qa_gen_var] question_styles is empty. Configure QA_GEN_VAR_QUESTION_STYLES.")

        unknown = [s for s in cfg.question_styles if s not in QA_GEN_VAR_STYLE_INSTRUCTIONS]
        if unknown:
            raise ValueError(
                f"[qa_gen_var] Unknown styles: {unknown}. "
                f"Available: {list(QA_GEN_VAR_STYLE_INSTRUCTIONS)}"
            )

        num_rows: int = kwargs.get("num_rows", NUM_ROWS)
        batch_size: int = kwargs.get("batch_size", cfg.batch_size)
        exhaustive = num_rows is not None and int(num_rows) < 0

        print(
            f"[qa_gen_var] styles: {cfg.question_styles}  "
            f"({len(cfg.question_styles)} pair(s) per document)"
        )

        client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
        models = await client.models.list()
        model_id = models.data[0].id
        print(f"[qa_gen_var] model: {model_id}")

        last_id = get_last_processed_id(output_file)
        next_row_id = last_id + 1 if last_id >= 0 else 0
        skip_ids = get_processed_sample_ids(output_file)

        ds_iter = _stream_dataset()
        dataset_id = os.environ.get("DATASET_ID", DATASET_ID)
        id_field = os.environ.get("DATASET_ID_FIELD", DATASET_ID_FIELD)
        ds_iter = assert_hf_dataset_has_fields(ds_iter, [id_field], dataset_id=dataset_id)

        total = None if exhaustive else num_rows
        pbar = tqdm(desc="[qa_gen_var] generating", initial=next_row_id, total=total)

        while exhaustive or next_row_id < num_rows:
            samples = _next_valid_samples(ds_iter, batch_size, skip_ids)
            if not samples:
                print("[qa_gen_var] Dataset exhausted.")
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
        print(f"[qa_gen_var] done — {next_row_id} QA rows → {output_file}")
