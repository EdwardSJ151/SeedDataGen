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
  - sample_text : {str(hf_row_id): {text, document_name}}
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
    DATASET_CHUNK_TYPE_FIELD,
    DATASET_DOC_ID_FIELD,
    DATASET_ID,
    DATASET_ID_FIELD,
    DATASET_MAX_CHARS,
    DATASET_MIN_CHARS,
    DATASET_SPLIT,
    DATASET_SUBSET,
    DATASET_SUMMARY_TYPE_VALUE,
    DATASET_TEXT_FIELD,
    NUM_ROWS,
    STOP_STRINGS,
    VLLM_API_KEY,
    VLLM_BASE_URL,
    get_dataset_doc_name_field,
    validate_pipeline_env,
)
from SeedDataGen.generator.prompts import QA_GEN_VAR_STYLE_INSTRUCTIONS, QA_GEN_VAR_SYSTEM_PROMPT
from SeedDataGen.registry import register
from SeedDataGen.schemas import StyledQARow
from SeedDataGen.utils import (
    assert_hf_dataset_has_fields,
    format_doc_summary,
    format_sample_text_for_prompt,
    get_last_processed_id,
    get_processed_sample_ids,
    get_sample_group_key,
    is_summary_enabled,
    load_doc_summaries,
    make_chunk_entry,
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


def _is_summary_row(rec: Dict[str, Any]) -> bool:
    chunk_type_field = os.environ.get("DATASET_CHUNK_TYPE_FIELD", DATASET_CHUNK_TYPE_FIELD)
    summary_value = os.environ.get("DATASET_SUMMARY_TYPE_VALUE", DATASET_SUMMARY_TYPE_VALUE)
    return rec.get(chunk_type_field) == summary_value


def _is_valid_chunk_rec(rec: Dict[str, Any], *, stream_idx: int = 0) -> Optional[str]:
    """Return hf_row_id if *rec* passes the same filters as _next_valid_samples, else None."""
    text_field = os.environ.get("DATASET_TEXT_FIELD", DATASET_TEXT_FIELD)
    id_field = os.environ.get("DATASET_ID_FIELD", DATASET_ID_FIELD)
    doc_name_field = get_dataset_doc_name_field()
    min_chars = int(os.environ.get("DATASET_MIN_CHARS", DATASET_MIN_CHARS))

    if _is_summary_row(rec):
        return None
    txt = rec.get(text_field, "")
    if not isinstance(txt, str):
        return None
    txt = _truncate(txt)
    if len(txt) < min_chars:
        return None
    try:
        hf_row_id = str(require_hf_field(rec, id_field, row_label=f"row {stream_idx}"))
        require_hf_field(rec, doc_name_field, row_label=f"row {stream_idx}")
    except ValueError:
        return None
    return hf_row_id


def _count_valid_chunks(
    *,
    skip_ids: Optional[set] = None,
    need_chunks: Optional[int] = None,
) -> int:
    """Count HF chunks that would be processed (same filters as _next_valid_samples)."""
    valid = 0
    for stream_idx, rec in enumerate(_stream_dataset()):
        hf_row_id = _is_valid_chunk_rec(rec, stream_idx=stream_idx)
        if hf_row_id is None:
            continue
        if skip_ids and hf_row_id in skip_ids:
            continue
        valid += 1
        if need_chunks is not None and valid >= need_chunks:
            break
    return valid


def _next_valid_samples(ds_iter, n: int, skip_ids: set) -> List[Dict[str, Any]]:
    doc_id_field = os.environ.get("DATASET_DOC_ID_FIELD", DATASET_DOC_ID_FIELD)
    out: List[Dict[str, Any]] = []
    for stream_idx, rec in _enumerate_global(ds_iter):
        hf_row_id = _is_valid_chunk_rec(rec, stream_idx=stream_idx)
        if hf_row_id is None:
            continue
        if hf_row_id in skip_ids:
            continue
        text_field = os.environ.get("DATASET_TEXT_FIELD", DATASET_TEXT_FIELD)
        doc_id_field = os.environ.get("DATASET_DOC_ID_FIELD", DATASET_DOC_ID_FIELD)
        doc_name_field = get_dataset_doc_name_field()
        txt = _truncate(rec.get(text_field, ""))
        document_name = str(require_hf_field(rec, doc_name_field, row_label=f"row {stream_idx}"))
        chunk_entry = make_chunk_entry(txt, document_name)
        sample: Dict[str, Any] = {
            "hf_row_id": hf_row_id,
            "sample_text": chunk_entry,
            "prompt_text": format_sample_text_for_prompt({hf_row_id: chunk_entry}),
        }
        doc_id = rec.get(doc_id_field)
        if doc_id is not None:
            sample["doc_id"] = doc_id
        out.append(sample)
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
    *,
    doc_summary: str = "",
) -> Optional[str]:
    style_instruction = QA_GEN_VAR_STYLE_INSTRUCTIONS.get(style)
    if style_instruction is None:
        print(f"[qa_gen_var] Unknown style '{style}'. Available: {list(QA_GEN_VAR_STYLE_INSTRUCTIONS)}")
        return None

    prompt = QA_GEN_VAR_SYSTEM_PROMPT.format(
        style_instruction=style_instruction,
        doc_summary=doc_summary,
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
    summary_map: Dict[str, str],
    retry_pairs: Optional[set] = None,
) -> int:
    sem = asyncio.Semaphore(cfg.max_concurrent)

    tasks = []
    for sample in samples:
        doc_summary = format_doc_summary(
            summary_map.get(str(sample["doc_id"])) if sample.get("doc_id") is not None else None
        )
        for style in cfg.question_styles:
            if retry_pairs is not None and (sample["hf_row_id"], style) not in retry_pairs:
                continue
            tasks.append((sample, style, doc_summary))

    async def _gen(sample: Dict[str, Any], style: str, doc_summary: str) -> Optional[str]:
        async with sem:
            return await _generate_qa_for_style(
                client, cfg, model_id, sample["prompt_text"], style, doc_summary=doc_summary
            )

    raw_outputs = await asyncio.gather(*[_gen(s, st, ds) for s, st, ds in tasks])

    rows: List[Dict[str, Any]] = []
    for (sample, style, _), raw in zip(tasks, raw_outputs):
        if raw is None:
            continue
        pairs = parse_qa_pairs(raw)
        if not pairs:
            continue
        qa = pairs[0]
        hf_row_id = sample["hf_row_id"]
        row: Dict[str, Any] = {
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
        }
        if sample.get("doc_id") is not None:
            row["document_id"] = sample["doc_id"]
        rows.append(row)
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

        need_chunks = None if exhaustive else math.ceil(num_rows / max(n_styles, 1))

        print(f"  [qa_gen_var] scanning {dataset_id} for valid chunks...", flush=True)
        valid = _count_valid_chunks(need_chunks=need_chunks)

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
            doc_summary = (
                format_doc_summary("[DOCUMENT_SUMMARY]")
                if is_summary_enabled()
                else ""
            )
            prompt = QA_GEN_VAR_SYSTEM_PROMPT.format(
                style_instruction=style_instruction,
                doc_summary=doc_summary,
                sample_text="[SAMPLE_TEXT]",
            )
            prompts.append((f"qa_gen_var / style={style} (user)", prompt))
        return prompts

    async def run(self, input_file: str, output_file: str, **kwargs) -> None:
        global _GLOBAL_STREAM_IDX
        _GLOBAL_STREAM_IDX = 0

        validate_pipeline_env()
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

        summary_map: Dict[str, str] = {}
        if is_summary_enabled():
            print("[qa_gen_var] loading document summaries...")
            summary_map = load_doc_summaries()
            print(f"[qa_gen_var] loaded {len(summary_map):,} document summaries")

        last_id = get_last_processed_id(output_file)
        next_row_id = last_id + 1 if last_id >= 0 else 0
        retry_pairs: Optional[set] = kwargs.get("retry_pairs")
        skip_ids = get_processed_sample_ids(
            output_file, exclude_status=["failed"] if retry_pairs else None
        )
        if retry_pairs:
            skip_ids -= {gk for (gk, _) in retry_pairs}

        ds_iter = _stream_dataset()
        dataset_id = os.environ.get("DATASET_ID", DATASET_ID)
        id_field = os.environ.get("DATASET_ID_FIELD", DATASET_ID_FIELD)
        doc_name_field = get_dataset_doc_name_field()
        ds_iter = assert_hf_dataset_has_fields(
            ds_iter, [id_field, doc_name_field], dataset_id=dataset_id
        )

        if exhaustive:
            print("[qa_gen_var] counting valid chunks for progress bar...", flush=True)
            valid = _count_valid_chunks()
            total = valid * len(cfg.question_styles)
            print(
                f"[qa_gen_var] plan: {valid:,} chunks × {len(cfg.question_styles)} styles "
                f"= {total:,} QA rows"
            )
        else:
            total = num_rows
        pbar = tqdm(desc="[qa_gen_var] generating", initial=next_row_id, total=total)

        while exhaustive or next_row_id < num_rows:
            samples = _next_valid_samples(ds_iter, batch_size, skip_ids)
            if not samples:
                print("[qa_gen_var] Dataset exhausted.")
                break
            for s in samples:
                skip_ids.add(s["hf_row_id"])

            next_row_id = await _process_batch(
                client, cfg, model_id, samples, next_row_id, output_file, summary_map,
                retry_pairs=retry_pairs,
            )
            pbar.n = min(next_row_id, total)
            pbar.refresh()

        pbar.close()
        print(f"[qa_gen_var] done — {next_row_id} QA rows → {output_file}")
