"""
Phase: rewrite_gen

Single-turn rewrite/summarization generation.  For each chunk, makes one
*pair* of LLM calls per configured style and emits a finished single-turn
conversation (one user request + one assistant answer).  This is NOT expanded
by conv_expand_var — it is already a complete Q&A, so it goes straight to the
shared tail (conv_filter → judge → embed_filter).

Two calls per (chunk × style), mirroring conv_expand_var's split but isolated
here (this phase reuses none of editor/'s prompts):
  1. user turn  — a persona that writes ONLY the request (summarize / simplify /
     reframe a named section of a named document), parametrized by the style.
  2. assistant turn — answers the request using the chunk as its own internal
     knowledge, with no meta-references, emitting REFUSAL_STRING when the chunk
     lacks the basis.

Role:   GENERATOR
Input:  HuggingFace dataset (streaming)
Output: ConversationRow

Row shape:
  - sample_id   : [hf_row_id]
  - sample_text : {str(hf_row_id): {text, document_name}}
  - messages    : [{"role": "user", ...}, {"role": "assistant", ...}]
  - GEN_TYPE    : "rewrite_gen", num_chunks=1
  - question_style : the style that produced the request

Config env prefix: REWRITE_GEN_
"""

import asyncio
import os
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from tqdm import tqdm

from SeedDataGen.base_phase import Phase, PhaseRole
from SeedDataGen.config import (
    DATASET_DOC_ID_FIELD,
    DATASET_ID,
    DATASET_ID_FIELD,
    DATASET_MAX_CHARS,
    DATASET_MIN_CHARS,
    DATASET_SPLIT,
    DATASET_SUBSET,
    DATASET_TEXT_FIELD,
    NUM_ROWS,
    REFUSAL_STRING,
    STOP_STRINGS,
    VLLM_API_KEY,
    VLLM_BASE_URL,
    get_dataset_doc_name_field,
    validate_pipeline_env,
)
from SeedDataGen.generator.prompts import (
    REWRITE_GEN_ASSISTANT_SYSTEM_PROMPT,
    REWRITE_GEN_ASSISTANT_USER_MSG,
    REWRITE_GEN_STYLE_INSTRUCTIONS,
    REWRITE_GEN_USER_TURN_SYSTEM_PROMPT,
    REWRITE_GEN_USER_TURN_USER_MSG,
)
from SeedDataGen.registry import register
from SeedDataGen.schemas import ConversationRow
from SeedDataGen.utils import (
    assert_hf_dataset_has_fields,
    format_sample_text_for_prompt,
    get_last_processed_id,
    get_processed_sample_ids,
    get_sample_group_key,
    make_chunk_entry,
    require_hf_field,
    write_jsonl_batch,
)

GEN_TYPE = "rewrite_gen"


class RewriteGenConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="REWRITE_GEN_", env_file=".env", extra="ignore", enable_decoding=False
    )

    question_styles: List[str] = ["summary", "simplify", "focus"]
    user_turn_temperature: float = 0.9
    user_turn_top_p: float = 0.95
    user_turn_max_tokens: int = 256
    assistant_turn_temperature: float = 0.7
    assistant_turn_top_p: float = 0.9
    assistant_turn_max_tokens: int = 2048
    batch_size: int = 32
    max_concurrent: int = 64

    @field_validator("question_styles", mode="before")
    @classmethod
    def _parse_styles(cls, v):
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Dataset streaming
# ---------------------------------------------------------------------------
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


def _is_valid_chunk_rec(rec: Dict[str, Any], *, stream_idx: int = 0) -> Optional[str]:
    """Return hf_row_id if *rec* is a usable chunk, else None."""
    text_field = os.environ.get("DATASET_TEXT_FIELD", DATASET_TEXT_FIELD)
    id_field = os.environ.get("DATASET_ID_FIELD", DATASET_ID_FIELD)
    doc_name_field = get_dataset_doc_name_field()
    min_chars = int(os.environ.get("DATASET_MIN_CHARS", DATASET_MIN_CHARS))

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


def _count_valid_chunks(*, skip_ids: Optional[set] = None) -> int:
    valid = 0
    for stream_idx, rec in enumerate(_stream_dataset()):
        hf_row_id = _is_valid_chunk_rec(rec, stream_idx=stream_idx)
        if hf_row_id is None:
            continue
        if skip_ids and hf_row_id in skip_ids:
            continue
        valid += 1
    return valid


def _next_valid_samples(ds_iter, n: int, skip_ids: set) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for stream_idx, rec in _enumerate_global(ds_iter):
        hf_row_id = _is_valid_chunk_rec(rec, stream_idx=stream_idx)
        if hf_row_id is None or hf_row_id in skip_ids:
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


# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------
async def _generate_user_turn(
    client: AsyncOpenAI,
    cfg: RewriteGenConfig,
    model_id: str,
    sample_text: str,
    style: str,
    *,
    sem: asyncio.Semaphore,
) -> Optional[str]:
    style_instruction = REWRITE_GEN_STYLE_INSTRUCTIONS.get(style)
    if style_instruction is None:
        print(f"[rewrite_gen] unknown style '{style}'")
        return None
    user_msg = REWRITE_GEN_USER_TURN_USER_MSG.format(style_instruction=style_instruction, sample_text=sample_text)
    try:
        async with sem:
            resp = await client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": REWRITE_GEN_USER_TURN_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=cfg.user_turn_temperature,
                top_p=cfg.user_turn_top_p,
                max_tokens=cfg.user_turn_max_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}, "stop": STOP_STRINGS},
            )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"[rewrite_gen] user turn LLM error (style={style}): {e}")
        return None


async def _generate_assistant_turn(
    client: AsyncOpenAI,
    cfg: RewriteGenConfig,
    model_id: str,
    sample_text: str,
    request: str,
    *,
    sem: asyncio.Semaphore,
) -> Optional[str]:
    user_msg = REWRITE_GEN_ASSISTANT_USER_MSG.format(
        sample_text=sample_text, request=request, refusal_string=REFUSAL_STRING
    )
    try:
        async with sem:
            resp = await client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": REWRITE_GEN_ASSISTANT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=cfg.assistant_turn_temperature,
                top_p=cfg.assistant_turn_top_p,
                max_tokens=cfg.assistant_turn_max_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}, "stop": STOP_STRINGS},
            )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"[rewrite_gen] assistant turn LLM error: {e}")
        return None


async def _generate_conversation(
    client: AsyncOpenAI,
    cfg: RewriteGenConfig,
    model_id: str,
    sample_text: str,
    style: str,
    sem: asyncio.Semaphore,
) -> Optional[List[Dict[str, str]]]:
    """Two sequential calls → a single-turn conversation, or None on failure."""
    request = await _generate_user_turn(client, cfg, model_id, sample_text, style, sem=sem)
    if not request:
        return None
    answer = await _generate_assistant_turn(client, cfg, model_id, sample_text, request, sem=sem)
    if not answer:
        return None
    return [
        {"role": "user", "content": request},
        {"role": "assistant", "content": answer},
    ]


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------
async def _process_batch(
    client: AsyncOpenAI,
    cfg: RewriteGenConfig,
    model_id: str,
    samples: List[Dict[str, Any]],
    next_row_id: int,
    output_file: str,
    retry_pairs: Optional[set] = None,
) -> int:
    sem = asyncio.Semaphore(cfg.max_concurrent)

    tasks = [
        (sample, style)
        for sample in samples
        for style in cfg.question_styles
        if retry_pairs is None or (sample["hf_row_id"], style) in retry_pairs
    ]

    async def _gen(sample: Dict[str, Any], style: str):
        return await _generate_conversation(
            client, cfg, model_id, sample["prompt_text"], style, sem
        )

    results = await asyncio.gather(*[_gen(s, st) for s, st in tasks])

    rows: List[Dict[str, Any]] = []
    for (sample, style), messages in zip(tasks, results):
        if not messages:
            continue
        hf_row_id = sample["hf_row_id"]
        row: Dict[str, Any] = {
            "id": next_row_id,
            "origin_id": next_row_id,
            "sample_id": [hf_row_id],
            "sample_text": {hf_row_id: sample["sample_text"]},
            "messages": messages,
            "question_style": style,
            "GEN_TYPE": GEN_TYPE,
            "num_chunks": 1,
        }
        if sample.get("doc_id") is not None:
            row["document_id"] = sample["doc_id"]
        rows.append(row)
        next_row_id += 1

    if rows:
        write_jsonl_batch(output_file, rows)
    return next_row_id


# ---------------------------------------------------------------------------
# Phase class
# ---------------------------------------------------------------------------
@register
class RewriteGenPhase(Phase):
    name = "rewrite_gen"
    role = PhaseRole.GENERATOR
    input_schema = None
    output_schema = ConversationRow

    async def estimate(self, **kwargs) -> Optional[int]:
        cfg = RewriteGenConfig()
        num_rows: int = kwargs.get("num_rows", NUM_ROWS)
        if num_rows is not None and int(num_rows) >= 0:
            return int(num_rows)
        return _count_valid_chunks() * len(cfg.question_styles)

    def describe_prompts(self):
        results = [
            ("rewrite_gen / user turn — system persona", REWRITE_GEN_USER_TURN_SYSTEM_PROMPT),
            ("rewrite_gen / assistant turn — system persona", REWRITE_GEN_ASSISTANT_SYSTEM_PROMPT),
        ]
        for style in RewriteGenConfig().question_styles:
            instr = REWRITE_GEN_STYLE_INSTRUCTIONS.get(style, f"[UNKNOWN STYLE: {style}]")
            results.append((
                f"rewrite_gen / user turn — style={style} (user msg)",
                REWRITE_GEN_USER_TURN_USER_MSG.format(style_instruction=instr, sample_text="[<documento> SAMPLE_TEXT]"),
            ))
        results.append((
            "rewrite_gen / assistant turn (user msg)",
            REWRITE_GEN_ASSISTANT_USER_MSG.format(
                sample_text="[<documento> SAMPLE_TEXT]",
                request="[USER_REQUEST]",
                refusal_string=REFUSAL_STRING,
            ),
        ))
        return results

    async def run(self, input_file: str, output_file: str, **kwargs) -> None:
        global _GLOBAL_STREAM_IDX
        _GLOBAL_STREAM_IDX = 0

        validate_pipeline_env()
        cfg = RewriteGenConfig()

        if not cfg.question_styles:
            raise ValueError("[rewrite_gen] question_styles is empty. Configure REWRITE_GEN_QUESTION_STYLES.")
        unknown = [s for s in cfg.question_styles if s not in REWRITE_GEN_STYLE_INSTRUCTIONS]
        if unknown:
            raise ValueError(
                f"[rewrite_gen] Unknown styles: {unknown}. "
                f"Available: {list(REWRITE_GEN_STYLE_INSTRUCTIONS)}"
            )

        num_rows: int = kwargs.get("num_rows", NUM_ROWS)
        batch_size: int = kwargs.get("batch_size", cfg.batch_size)
        exhaustive = num_rows is not None and int(num_rows) < 0

        print(
            f"[rewrite_gen] styles: {cfg.question_styles}  "
            f"({len(cfg.question_styles)} conversation(s) per chunk)"
        )

        client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
        models = await client.models.list()
        model_id = models.data[0].id
        print(f"[rewrite_gen] model: {model_id}")

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
            print("[rewrite_gen] counting valid chunks for progress bar...", flush=True)
            valid = _count_valid_chunks()
            total = valid * len(cfg.question_styles)
            print(f"[rewrite_gen] plan: {valid:,} chunks × {len(cfg.question_styles)} styles = {total:,} rows")
        else:
            total = num_rows
        pbar = tqdm(desc="[rewrite_gen] generating", initial=next_row_id, total=total)

        while exhaustive or next_row_id < num_rows:
            samples = _next_valid_samples(ds_iter, batch_size, skip_ids)
            if not samples:
                print("[rewrite_gen] Dataset exhausted.")
                break
            for s in samples:
                skip_ids.add(s["hf_row_id"])

            next_row_id = await _process_batch(
                client, cfg, model_id, samples, next_row_id, output_file,
                retry_pairs=retry_pairs,
            )
            pbar.n = min(next_row_id, total)
            pbar.refresh()

        pbar.close()
        print(f"[rewrite_gen] done — {next_row_id} rows → {output_file}")
