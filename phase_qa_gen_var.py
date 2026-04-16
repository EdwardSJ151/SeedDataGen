"""
Phase: qa_gen_var

Style-constrained QA generation.  For each document, makes one LLM call per
configured style and extracts exactly one QA pair per call.  The number of QA
pairs produced per document equals the number of styles configured.

Available built-in styles (keys in QA_GEN_VAR_STYLE_INSTRUCTIONS):
  - general: broad question about the main topic
  - specific: focused on a concrete detail or fact
  - compositional: requires combining multiple pieces of information
  - comparative: compares two concepts or entities from the text

Additional styles can be added to QA_GEN_VAR_STYLE_INSTRUCTIONS in prompts.py.

Role:   GENERATOR
Input:  HuggingFace dataset (streaming)
Output: QARow  — fully compatible with qa_filter and all downstream phases.
        Each row carries an extra `question_style` field recording which style
        produced it (preserved through downstream phases via extra='ignore').
"""

import asyncio
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
from pydantic import field_validator
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
from SeedDataGen.prompts import QA_GEN_VAR_STYLE_INSTRUCTIONS, QA_GEN_VAR_SYSTEM_PROMPT
from SeedDataGen.registry import register
from SeedDataGen.schemas import QARow, StyledQARow
from SeedDataGen.utils import get_last_processed_id, get_max_int_field, parse_qa_pairs, write_jsonl_batch


class QAGenVarConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="QA_GEN_VAR_", env_file=".env", extra="ignore")

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


# Text helpers (identical to phase_qa_gen)
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
    sample_id_start: int,
    next_row_id: int,
    output_file: str,
) -> int:
    sem = asyncio.Semaphore(cfg.max_concurrent)

    # Build one task per (sample, style) pair
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
        # We expect exactly one pair; take the first one if the model produced more.
        if not pairs:
            continue
        qa = pairs[0]
        sample_idx = samples.index(sample)
        rows.append({
            "id": next_row_id,
            "origin_id": next_row_id,
            "sample_id": sample_id_start + sample_idx,
            "sample_text": sample["sample_text"],
            "question": qa["question"],
            "answer": qa["answer"],
            "question_style": style,
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

    async def run(self, input_file: str, output_file: str, **kwargs) -> None:
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
        pbar = tqdm(desc="[qa_gen_var] generating", initial=next_row_id, total=num_rows)

        while next_row_id < num_rows:
            samples = _next_valid_samples(ds_iter, batch_size)
            if not samples:
                print("[qa_gen_var] Dataset exhausted.")
                break

            next_row_id = await _process_batch(
                client, cfg, model_id, samples, sample_id_counter, next_row_id, output_file
            )
            sample_id_counter += len(samples)
            pbar.n = min(next_row_id, num_rows)
            pbar.refresh()

        pbar.close()
        print(f"[qa_gen_var] done — {next_row_id} QA rows → {output_file}")
