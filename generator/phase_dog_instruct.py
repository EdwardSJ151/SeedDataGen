"""
Phase: dog_instruct

DogInstruct-style single-turn generation. For each valid chunk, first generate a
user question that the chunk could answer, then rewrite the chunk lightly so it
reads like a natural assistant response to that question.

Output is a finished single-turn conversation (one user question + one assistant
answer) — back-translation is single-turn, so it is NOT expanded by
conv_expand_var; it goes straight to the tail (conv_filter → judge → embed_filter).

For each valid chunk, one QA row is produced per configured persona: the persona
steers ONLY the user question (its perspective/style is injected into the question
prompt); the assistant rewrite is unaffected. Available personas live in
DOG_INSTRUCT_PERSONAS (SeedDataGen/generator/prompts.py).

Role:   GENERATOR
Input:  HuggingFace dataset (streaming)
Output: ConversationRow (single-turn)

Row shape:
  - sample_id   : [hf_row_id]
  - sample_text : {str(hf_row_id): {text, document_name}}
  - messages    : [{"role": "user", ...}, {"role": "assistant", ...}]
  - GEN_TYPE    : "dog_instruct", num_chunks=1
  - question_style : the persona that produced the question
"""

import asyncio
import math
import os
import re
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
from SeedDataGen.generator.prompts import (
    DOG_INSTRUCT_PERSONAS,
    DOG_INSTRUCT_QUESTION_SYSTEM_PROMPT,
    DOG_INSTRUCT_QUESTION_USER_PROMPT,
    DOG_INSTRUCT_REWRITE_SYSTEM_PROMPT,
    DOG_INSTRUCT_REWRITE_USER_PROMPT,
)
from SeedDataGen.registry import register
from SeedDataGen.schemas import ConversationRow
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
    require_hf_field,
    write_jsonl_batch,
)

GEN_TYPE = "dog_instruct"


class DogInstructConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DOG_INSTRUCT_",
        env_file=".env",
        extra="ignore",
        enable_decoding=False,
    )

    personas: List[str] = list(DOG_INSTRUCT_PERSONAS)
    question_temperature: float = 0.7
    question_top_p: float = 0.9
    question_max_tokens: int = 256
    rewrite_temperature: float = 0.4
    rewrite_top_p: float = 0.9
    rewrite_max_tokens: int = 1024
    batch_size: int = 32
    max_concurrent: int = 64

    @field_validator("personas", mode="before")
    @classmethod
    def _parse_personas(cls, v):
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v


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


def _stream_dataset():
    from datasets import load_dataset

    dataset_id = os.environ.get("DATASET_ID", DATASET_ID)
    dataset_subset = os.environ.get("DATASET_SUBSET", DATASET_SUBSET)
    dataset_split = os.environ.get("DATASET_SPLIT", DATASET_SPLIT)
    ds = load_dataset(dataset_id, dataset_subset, split=dataset_split, streaming=True)
    return iter(ds)


_GLOBAL_STREAM_IDX = 0


def _enumerate_global(ds_iter):
    global _GLOBAL_STREAM_IDX
    for rec in ds_iter:
        yield _GLOBAL_STREAM_IDX, rec
        _GLOBAL_STREAM_IDX += 1


def _is_summary_row(rec: Dict[str, Any]) -> bool:
    chunk_type_field = os.environ.get(
        "DATASET_CHUNK_TYPE_FIELD", DATASET_CHUNK_TYPE_FIELD
    )
    summary_value = os.environ.get(
        "DATASET_SUMMARY_TYPE_VALUE", DATASET_SUMMARY_TYPE_VALUE
    )
    return rec.get(chunk_type_field) == summary_value


def _is_valid_chunk_rec(rec: Dict[str, Any], *, stream_idx: int = 0) -> Optional[str]:
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


def _count_valid_chunks(*, need_chunks: Optional[int] = None) -> int:
    valid = 0
    for stream_idx, rec in enumerate(_stream_dataset()):
        if _is_valid_chunk_rec(rec, stream_idx=stream_idx) is None:
            continue
        valid += 1
        if need_chunks is not None and valid >= need_chunks:
            break
    return valid


def _next_valid_samples(ds_iter, n: int, skip_ids: set) -> List[Dict[str, Any]]:
    text_field = os.environ.get("DATASET_TEXT_FIELD", DATASET_TEXT_FIELD)
    doc_id_field = os.environ.get("DATASET_DOC_ID_FIELD", DATASET_DOC_ID_FIELD)
    doc_name_field = get_dataset_doc_name_field()

    out: List[Dict[str, Any]] = []
    for stream_idx, rec in _enumerate_global(ds_iter):
        hf_row_id = _is_valid_chunk_rec(rec, stream_idx=stream_idx)
        if hf_row_id is None or hf_row_id in skip_ids:
            continue

        txt = _truncate(rec.get(text_field, ""))
        document_name = str(
            require_hf_field(rec, doc_name_field, row_label=f"row {stream_idx}")
        )
        chunk_entry = make_chunk_entry(txt, document_name)
        sample: Dict[str, Any] = {
            "hf_row_id": hf_row_id,
            "sample_text": chunk_entry,
            "prompt_text": format_sample_text_for_prompt({hf_row_id: chunk_entry}),
            "answer_text": txt,
        }
        doc_id = rec.get(doc_id_field)
        if doc_id is not None:
            sample["doc_id"] = doc_id
        out.append(sample)
        if len(out) >= n:
            break
    return out


def _clean_question_output(text: str) -> str:
    cleaned = re.sub(
        r"^\s*pergunta\s*:\s*", "", (text or "").strip(), flags=re.IGNORECASE
    )
    return " ".join(cleaned.split())


def _clean_answer_output(text: str) -> str:
    cleaned = re.sub(
        r"^\s*resposta\s*:\s*", "", (text or "").strip(), flags=re.IGNORECASE
    )
    return cleaned.strip()


async def _generate_user_question(
    client: AsyncOpenAI,
    cfg: DogInstructConfig,
    model_id: str,
    sample_text: str,
    persona: str,
    *,
    doc_summary: str = "",
) -> Optional[str]:
    persona_instruction = DOG_INSTRUCT_PERSONAS.get(persona)
    if persona_instruction is None:
        print(
            f"[dog_instruct] Unknown persona '{persona}'. "
            f"Available: {list(DOG_INSTRUCT_PERSONAS)}"
        )
        return None

    system_prompt = DOG_INSTRUCT_QUESTION_SYSTEM_PROMPT.format(
        persona_instruction=persona_instruction
    )
    try:
        resp = await client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": DOG_INSTRUCT_QUESTION_USER_PROMPT.format(
                        doc_summary=doc_summary,
                        sample_text=sample_text,
                    ),
                },
            ],
            temperature=cfg.question_temperature,
            top_p=cfg.question_top_p,
            max_tokens=cfg.question_max_tokens,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
                "stop": STOP_STRINGS,
            },
        )
    except Exception as e:
        print(f"[dog_instruct] question generation error: {e}")
        return None

    question = _clean_question_output(resp.choices[0].message.content or "")
    return question or None


async def _rewrite_assistant_answer(
    client: AsyncOpenAI,
    cfg: DogInstructConfig,
    model_id: str,
    question: str,
    sample_text: str,
    *,
    doc_summary: str = "",
) -> Optional[str]:
    try:
        resp = await client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": DOG_INSTRUCT_REWRITE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": DOG_INSTRUCT_REWRITE_USER_PROMPT.format(
                        question=question,
                        doc_summary=doc_summary,
                        sample_text=sample_text,
                    ),
                },
            ],
            temperature=cfg.rewrite_temperature,
            top_p=cfg.rewrite_top_p,
            max_tokens=cfg.rewrite_max_tokens,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
                "stop": STOP_STRINGS,
            },
        )
    except Exception as e:
        print(f"[dog_instruct] answer rewrite error: {e}")
        return None

    answer_text = _clean_answer_output(resp.choices[0].message.content or "")
    return answer_text or None


async def _process_batch(
    client: AsyncOpenAI,
    cfg: DogInstructConfig,
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
            summary_map.get(str(sample["doc_id"]))
            if sample.get("doc_id") is not None
            else None
        )
        for persona in cfg.personas:
            if retry_pairs is not None and (sample["hf_row_id"], persona) not in retry_pairs:
                continue
            tasks.append((sample, persona, doc_summary))

    async def _one(
        sample: Dict[str, Any], persona: str, doc_summary: str
    ) -> Optional[Dict[str, str]]:
        async with sem:
            question = await _generate_user_question(
                client,
                cfg,
                model_id,
                sample["prompt_text"],
                persona,
                doc_summary=doc_summary,
            )
        if not question:
            return None
        async with sem:
            answer = await _rewrite_assistant_answer(
                client,
                cfg,
                model_id,
                question,
                sample["prompt_text"],
                doc_summary=doc_summary,
            )
        return {
            "question": question,
            "answer": answer or sample["answer_text"],
        }

    outputs = await asyncio.gather(*[_one(s, p, ds) for s, p, ds in tasks])

    rows: List[Dict[str, Any]] = []
    for (sample, persona, _), generated in zip(tasks, outputs):
        if generated is None:
            continue
        row: Dict[str, Any] = {
            "id": next_row_id,
            "origin_id": next_row_id,
            "sample_id": [sample["hf_row_id"]],
            "sample_text": {sample["hf_row_id"]: sample["sample_text"]},
            "messages": [
                {"role": "user", "content": generated["question"]},
                {"role": "assistant", "content": generated["answer"]},
            ],
            "question_style": persona,
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


@register
class DogInstructPhase(Phase):
    name = "dog_instruct"
    role = PhaseRole.GENERATOR
    input_schema = None
    output_schema = ConversationRow

    async def estimate(self, **kwargs) -> Optional[int]:
        cfg = DogInstructConfig()
        n_personas = len(cfg.personas)
        num_rows = int(kwargs.get("num_rows", NUM_ROWS))
        exhaustive = num_rows < 0
        dataset_id = os.environ.get("DATASET_ID", DATASET_ID)

        need_chunks = None if exhaustive else math.ceil(num_rows / max(n_personas, 1))
        print(f"  [dog_instruct] scanning {dataset_id} for valid chunks...", flush=True)
        valid = _count_valid_chunks(need_chunks=need_chunks)
        rows = valid * n_personas
        if not exhaustive:
            rows = min(rows, num_rows)
        print(
            f"  [dog_instruct] {valid:,} valid chunks x {n_personas} personas"
            + (
                f" (capped at NUM_ROWS={num_rows:,})"
                if not exhaustive and rows < valid * n_personas
                else ""
            )
        )
        return rows

    def describe_prompts(self):
        cfg = DogInstructConfig()
        doc_summary = (
            format_doc_summary("[DOCUMENT_SUMMARY]") if is_summary_enabled() else ""
        )
        prompts = []
        for persona in cfg.personas:
            persona_instruction = DOG_INSTRUCT_PERSONAS.get(
                persona, f"[UNKNOWN PERSONA: {persona}]"
            )
            prompts.append(
                (
                    f"dog_instruct / question generation (system) [persona={persona}]",
                    DOG_INSTRUCT_QUESTION_SYSTEM_PROMPT.format(
                        persona_instruction=persona_instruction
                    ),
                )
            )
        prompts += [
            (
                "dog_instruct / question generation (user)",
                DOG_INSTRUCT_QUESTION_USER_PROMPT.format(
                    doc_summary=doc_summary,
                    sample_text="[SAMPLE_TEXT]",
                ),
            ),
            (
                "dog_instruct / answer rewrite (system)",
                DOG_INSTRUCT_REWRITE_SYSTEM_PROMPT,
            ),
            (
                "dog_instruct / answer rewrite (user)",
                DOG_INSTRUCT_REWRITE_USER_PROMPT.format(
                    question="[QUESTION]",
                    doc_summary=doc_summary,
                    sample_text="[DOCUMENTO]",
                ),
            ),
        ]
        return prompts

    async def run(self, input_file: str, output_file: str, **kwargs) -> None:
        global _GLOBAL_STREAM_IDX
        _GLOBAL_STREAM_IDX = 0

        validate_pipeline_env()
        cfg = DogInstructConfig()

        if not cfg.personas:
            raise ValueError(
                "[dog_instruct] personas is empty. Configure DOG_INSTRUCT_PERSONAS."
            )
        unknown = [p for p in cfg.personas if p not in DOG_INSTRUCT_PERSONAS]
        if unknown:
            raise ValueError(
                f"[dog_instruct] Unknown personas: {unknown}. "
                f"Available: {list(DOG_INSTRUCT_PERSONAS)}"
            )

        num_rows: int = kwargs.get("num_rows", NUM_ROWS)
        batch_size: int = kwargs.get("batch_size", cfg.batch_size)
        exhaustive = num_rows is not None and int(num_rows) < 0

        print(
            f"[dog_instruct] personas: {cfg.personas}  "
            f"({len(cfg.personas)} pair(s) per document)"
        )

        client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
        models = await client.models.list()
        model_id = models.data[0].id
        print(f"[dog_instruct] model: {model_id}")

        summary_map: Dict[str, str] = {}
        if is_summary_enabled():
            print("[dog_instruct] loading document summaries...")
            summary_map = load_doc_summaries()
            print(f"[dog_instruct] loaded {len(summary_map):,} document summaries")

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
            ds_iter,
            [id_field, doc_name_field],
            dataset_id=dataset_id,
        )

        if exhaustive:
            print(
                "[dog_instruct] counting valid chunks for progress bar...", flush=True
            )
            valid = _count_valid_chunks()
            total = valid * len(cfg.personas)
            print(
                f"[dog_instruct] plan: {valid:,} chunks x {len(cfg.personas)} personas "
                f"= {total:,} QA rows"
            )
        else:
            total = num_rows
        pbar = tqdm(desc="[dog_instruct] generating", initial=next_row_id, total=total)

        while exhaustive or next_row_id < num_rows:
            samples = _next_valid_samples(ds_iter, batch_size, skip_ids)
            if not samples:
                print("[dog_instruct] Dataset exhausted.")
                break
            for sample in samples:
                skip_ids.add(sample["hf_row_id"])

            next_row_id = await _process_batch(
                client,
                cfg,
                model_id,
                samples,
                next_row_id,
                output_file,
                summary_map,
                retry_pairs=retry_pairs,
            )
            pbar.n = min(next_row_id, total) if total is not None else next_row_id
            pbar.refresh()

        pbar.close()
        print(f"[dog_instruct] done — {next_row_id} QA rows → {output_file}")
