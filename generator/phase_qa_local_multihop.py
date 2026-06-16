"""
Phase: qa_local_multihop

Multihop QA generation over adjacent chunks.  For every document in the Chroma
collection, slides a window of NUM_CHUNKS consecutive chunks (step WINDOW_STRIDE)
and, for each window, makes one LLM call per configured style, emitting one QA
pair per (window × style).

Role:   GENERATOR (depends on the chroma_preprocess collection)
Input:  Chroma collection (not a JSONL file)
Output: StyledQARow

Row shape:
  - sample_id   : [hf_row_id, ...]   (one id per chunk in the window, in order)
  - sample_text : {str(hf_row_id): {text, document_name}, ...}
  - GEN_TYPE    : "qa_local_multihop", num_chunks=NUM_CHUNKS, doc_constraint=None
  - question_style : the style that produced the pair

Styles are shared with qa_gen_var via QA_GEN_VAR_QUESTION_STYLES.
"""

import asyncio
import math
import os
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
from pydantic_settings import BaseSettings, SettingsConfigDict
from tqdm import tqdm

from SeedDataGen.base_phase import Phase, PhaseRole
from SeedDataGen.config import NUM_ROWS, STOP_STRINGS, VLLM_API_KEY, VLLM_BASE_URL
from SeedDataGen.generator.prompts import QA_GEN_VAR_STYLE_INSTRUCTIONS, QA_LOCAL_MULTIHOP_PROMPT
from SeedDataGen.preprocess.chunk_index import get_collection_from_env
from SeedDataGen.preprocess.chunk_retrieval import doc_chunk_map, get_doc_chunks
from SeedDataGen.registry import register
from SeedDataGen.schemas import StyledQARow
from SeedDataGen.utils import (
    format_doc_summary,
    format_sample_text,
    get_last_processed_id,
    get_sample_group_key,
    is_summary_enabled,
    load_doc_summaries,
    parse_qa_pairs,
    sample_text_from_chunks,
    write_jsonl_batch,
)

GEN_TYPE = "qa_local_multihop"


class QALocalMultihopConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="QA_LOCAL_MULTIHOP_", env_file=".env", extra="ignore")

    num_chunks: int = 3
    window_stride: int = 1
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 512
    batch_size: int = 32
    max_concurrent: int = 64


def _styles_from_env() -> List[str]:
    raw = os.environ.get("QA_GEN_VAR_QUESTION_STYLES", "")
    styles = [s.strip() for s in raw.split(",") if s.strip()]
    return styles or ["general", "specific", "compositional", "comparative"]


def _processed_window_styles(filepath: str) -> set:
    """Set of (sample_group_key, question_style) already present in output."""
    import json

    seen: set = set()
    if not os.path.exists(filepath):
        return seen
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = obj.get("sample_id")
            style = obj.get("question_style")
            if sid is None:
                continue
            seen.add((get_sample_group_key(sid), style))
    return seen


def _iter_windows(collection, num_chunks: int, stride: int):
    """Yield windows (lists of chunk dicts) of size num_chunks across all docs."""
    chunk_map = doc_chunk_map(collection)
    for doc_id in sorted(chunk_map.keys(), key=lambda d: str(d)):
        chunks = get_doc_chunks(collection, doc_id)
        if len(chunks) < num_chunks:
            continue
        start = 0
        while start + num_chunks <= len(chunks):
            yield chunks[start : start + num_chunks]
            start += max(1, stride)


async def _generate(
    client: AsyncOpenAI,
    cfg: QALocalMultihopConfig,
    model_id: str,
    context_text: str,
    style: str,
    *,
    doc_summary: str = "",
) -> Optional[str]:
    style_instruction = QA_GEN_VAR_STYLE_INSTRUCTIONS.get(style, "")
    prompt = QA_LOCAL_MULTIHOP_PROMPT.format(
        style_instruction=style_instruction,
        doc_summary=doc_summary,
        sample_text=context_text,
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
        print(f"[qa_local_multihop] LLM error (style={style}): {e}")
        return None


async def _process_batch(
    client: AsyncOpenAI,
    cfg: QALocalMultihopConfig,
    model_id: str,
    tasks: List[Dict[str, Any]],
    next_row_id: int,
    output_file: str,
) -> int:
    sem = asyncio.Semaphore(cfg.max_concurrent)

    async def _one(task: Dict[str, Any]) -> Optional[str]:
        async with sem:
            return await _generate(
                client,
                cfg,
                model_id,
                task["context_text"],
                task["style"],
                doc_summary=task.get("doc_summary", ""),
            )

    raw_outputs = await asyncio.gather(*[_one(t) for t in tasks])

    rows: List[Dict[str, Any]] = []
    for task, raw in zip(tasks, raw_outputs):
        if raw is None:
            continue
        pairs = parse_qa_pairs(raw)
        if not pairs:
            continue
        qa = pairs[0]
        row: Dict[str, Any] = {
            "id": next_row_id,
            "origin_id": next_row_id,
            "sample_id": task["sample_id"],
            "sample_text": task["sample_text"],
            "question": qa["question"],
            "answer": qa["answer"],
            "question_style": task["style"],
            "GEN_TYPE": GEN_TYPE,
            "num_chunks": cfg.num_chunks,
            "doc_constraint": None,
        }
        if task.get("document_id") is not None:
            row["document_id"] = task["document_id"]
        rows.append(row)
        next_row_id += 1

    if rows:
        write_jsonl_batch(output_file, rows)
    return next_row_id


@register
class QALocalMultihopPhase(Phase):
    name = "qa_local_multihop"
    role = PhaseRole.GENERATOR
    input_schema = None
    output_schema = StyledQARow

    async def estimate(self, **kwargs) -> Optional[int]:
        """
        Count windows from Chroma metadata without LLM calls.
        Each window produces exactly one QA row per style, so:
            rows = min(total_windows × n_styles, NUM_ROWS)
        Reads only metadata from Chroma — no embeddings loaded.
        """
        cfg = QALocalMultihopConfig()
        styles = _styles_from_env()
        n_styles = len(styles)
        num_rows = int(kwargs.get("num_rows", NUM_ROWS))
        exhaustive = num_rows < 0

        collection = get_collection_from_env()
        print(
            f"  [qa_local_multihop] counting windows "
            f"(num_chunks={cfg.num_chunks}, stride={cfg.window_stride})...",
            flush=True,
        )

        total_windows = 0
        for _window in _iter_windows(collection, cfg.num_chunks, cfg.window_stride):
            total_windows += 1
            if not exhaustive and total_windows * n_styles >= num_rows:
                total_windows = math.ceil(num_rows / max(n_styles, 1))
                break

        rows = total_windows * n_styles
        if not exhaustive:
            rows = min(rows, num_rows)

        print(
            f"  [qa_local_multihop] {total_windows:,} windows × {n_styles} styles"
            + (f" (capped at NUM_ROWS={num_rows:,})" if not exhaustive and rows < total_windows * n_styles else "")
        )
        return rows

    def describe_prompts(self):
        prompts = []
        for style in _styles_from_env():
            style_instruction = QA_GEN_VAR_STYLE_INSTRUCTIONS.get(style, f"[UNKNOWN STYLE: {style}]")
            doc_summary = (
                format_doc_summary("[DOCUMENT_SUMMARY]")
                if is_summary_enabled()
                else ""
            )
            prompt = QA_LOCAL_MULTIHOP_PROMPT.format(
                style_instruction=style_instruction,
                doc_summary=doc_summary,
                sample_text="[MULTI_CHUNK_CONTEXT]",
            )
            prompts.append((f"qa_local_multihop / style={style} (user)", prompt))
        return prompts

    async def run(self, input_file: str, output_file: str, **kwargs) -> None:
        cfg = QALocalMultihopConfig()
        styles = _styles_from_env()
        num_rows: int = kwargs.get("num_rows", NUM_ROWS)
        batch_size: int = kwargs.get("batch_size", cfg.batch_size)
        exhaustive = num_rows is not None and int(num_rows) < 0

        print(f"[qa_local_multihop] num_chunks={cfg.num_chunks} stride={cfg.window_stride} styles={styles}")

        collection = get_collection_from_env()

        client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
        models = await client.models.list()
        model_id = models.data[0].id
        print(f"[qa_local_multihop] model: {model_id}")

        summary_map: Dict[str, str] = {}
        if is_summary_enabled():
            print("[qa_local_multihop] loading document summaries...")
            summary_map = load_doc_summaries()
            print(f"[qa_local_multihop] loaded {len(summary_map):,} document summaries")

        last_id = get_last_processed_id(output_file)
        next_row_id = last_id + 1 if last_id >= 0 else 0
        done = _processed_window_styles(output_file)

        pbar = tqdm(desc="[qa_local_multihop] generating", initial=next_row_id,
                    total=None if exhaustive else num_rows)

        pending: List[Dict[str, Any]] = []

        async def _flush() -> None:
            nonlocal next_row_id, pending
            if not pending:
                return
            next_row_id = await _process_batch(client, cfg, model_id, pending, next_row_id, output_file)
            pending = []
            pbar.n = next_row_id if exhaustive else min(next_row_id, num_rows)
            pbar.refresh()

        for window in _iter_windows(collection, cfg.num_chunks, cfg.window_stride):
            sample_id = [c["hf_row_id"] for c in window]
            sample_text = sample_text_from_chunks(window)
            group_key = get_sample_group_key(sample_id)
            context_text = format_sample_text(sample_text)
            doc_id = window[0].get("doc_id")
            doc_summary = format_doc_summary(
                summary_map.get(str(doc_id)) if doc_id is not None else None
            )

            for style in styles:
                if (group_key, style) in done:
                    continue
                pending.append({
                    "sample_id": sample_id,
                    "sample_text": sample_text,
                    "context_text": context_text,
                    "style": style,
                    "doc_summary": doc_summary,
                    "document_id": doc_id,
                })
                if len(pending) >= batch_size:
                    await _flush()
                    if not exhaustive and next_row_id >= num_rows:
                        break
            if not exhaustive and next_row_id >= num_rows:
                break

        await _flush()
        pbar.close()
        print(f"[qa_local_multihop] done — {next_row_id} QA rows → {output_file}")
