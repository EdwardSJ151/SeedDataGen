"""
Phase: qa_similarity_multihop

Multihop QA generation over similarity-selected chunk groups.  Chunk groups are
produced by similarity_groups_iter() according to one or more *jobs*, each
defining a similarity criterion (above / below / range), optional lexical
overlap (min matching words), and its own set of question styles.

Jobs are passed by the runner via kwargs["similarity_jobs"] (they come from the
YAML key QA_SIMILARITY_MULTIHOP_JOBS, which does not map cleanly onto env vars).

Role:   GENERATOR (depends on the chroma_preprocess collection)
Input:  Chroma collection (not a JSONL file)
Output: StyledQARow

Row shape:
  - sample_id   : [hf_row_id, ...]  (one id per chunk in the group)
  - sample_text : {str(hf_row_id): {text, document_name}, ...}
  - GEN_TYPE    : "qa_similarity_multihop", num_chunks, doc_constraint
  - similarity_job_index, similarity_mode, similarity_threshold,
    similarity_min, similarity_max, min_matching_words, chunk_group_similarity
  - question_style

Pairwise exclusion: every chunk pair emitted is recorded; resuming rebuilds the
exclusion set from prior output so no pair is reused.
"""

import asyncio
import json
import os
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
from pydantic_settings import BaseSettings, SettingsConfigDict
from tqdm import tqdm

from SeedDataGen.base_phase import Phase, PhaseRole
from SeedDataGen.config import NUM_ROWS, STOP_STRINGS, VLLM_API_KEY, VLLM_BASE_URL
from SeedDataGen.generator.prompts import (
    QA_GEN_VAR_STYLE_INSTRUCTIONS,
    QA_SIMILARITY_MULTIHOP_PROMPT_NEGATIVE,
    QA_SIMILARITY_MULTIHOP_PROMPT_POSITIVE,
)
from SeedDataGen.preprocess.chunk_index import get_collection_from_env
from SeedDataGen.preprocess.chunk_retrieval import _pairs, similarity_groups_iter
from SeedDataGen.registry import register
from SeedDataGen.schemas import StyledQARow
from SeedDataGen.utils import (
    format_doc_summaries_for_docs,
    format_doc_summary,
    format_sample_text_for_prompt,
    get_last_processed_id,
    get_sample_group_key,
    is_summary_enabled,
    load_doc_summaries,
    parse_qa_pairs,
    sample_text_from_chunks,
    write_jsonl_batch,
)

GEN_TYPE = "qa_similarity_multihop"


class QASimilarityMultihopConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="QA_SIMILARITY_MULTIHOP_", env_file=".env", extra="ignore")

    num_chunks: int = 3
    doc_constraint: Optional[str] = None  # "same" | "different" | None
    min_docs: int = 1
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 512
    batch_size: int = 32
    max_concurrent: int = 64
    max_candidates: int = 200


def _parse_styles(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(s).strip() for s in value if str(s).strip()]
    if isinstance(value, str):
        return [s.strip() for s in value.split(",") if s.strip()]
    return []


def _default_styles() -> List[str]:
    return _parse_styles(os.environ.get("QA_GEN_VAR_QUESTION_STYLES", "")) or [
        "general", "specific", "compositional", "comparative"
    ]


def _job_styles(job: Dict[str, Any], fallback: List[str]) -> List[str]:
    styles = _parse_styles(job.get("QA_GEN_VAR_QUESTION_STYLES"))
    return styles or fallback


def _is_positive(mode: str) -> bool:
    return mode in ("above", "range")


def _rebuild_used_pairs(filepath: str) -> set:
    """Rebuild the pairwise-exclusion set from a prior output file."""
    used: set = set()
    if not os.path.exists(filepath):
        return used
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = obj.get("sample_id")
            if isinstance(sid, list) and len(sid) >= 2:
                for p in _pairs([str(s) for s in sid]):
                    used.add(p)
    return used


async def _generate(
    client: AsyncOpenAI,
    cfg: QASimilarityMultihopConfig,
    model_id: str,
    context_text: str,
    style: str,
    positive: bool,
    *,
    doc_summary: str = "",
) -> Optional[str]:
    style_instruction = QA_GEN_VAR_STYLE_INSTRUCTIONS.get(style, "")
    template = QA_SIMILARITY_MULTIHOP_PROMPT_POSITIVE if positive else QA_SIMILARITY_MULTIHOP_PROMPT_NEGATIVE
    prompt = template.format(
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
        print(f"[qa_similarity_multihop] LLM error (style={style}): {e}")
        return None


async def _process_batch(
    client: AsyncOpenAI,
    cfg: QASimilarityMultihopConfig,
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
                task["positive"],
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
        row = {
            "id": next_row_id,
            "origin_id": next_row_id,
            "sample_id": task["sample_id"],
            "sample_text": task["sample_text"],
            "question": qa["question"],
            "answer": qa["answer"],
            "question_style": task["style"],
            "GEN_TYPE": GEN_TYPE,
            "num_chunks": cfg.num_chunks,
            "doc_constraint": cfg.doc_constraint,
        }
        row.update(task["similarity_meta"])
        if task.get("document_id") is not None:
            row["document_id"] = task["document_id"]
        rows.append(row)
        next_row_id += 1

    if rows:
        write_jsonl_batch(output_file, rows)
    return next_row_id


@register
class QASimilarityMultihopPhase(Phase):
    name = "qa_similarity_multihop"
    role = PhaseRole.GENERATOR
    input_schema = None
    output_schema = StyledQARow

    async def estimate(self, **kwargs) -> Optional[int]:
        """
        Count similarity groups without LLM calls by running similarity_groups_iter
        in full.  This loads all embeddings into memory and performs the same O(n²)
        cosine similarity scan the real run does — no LLM calls are made on top.

        Per-job breakdown is printed.  Pairwise exclusion is applied exactly as in
        the real run, so the count is exact (not an approximation).

        WARNING: for large collections (100k+ chunks) this scan may take several
        minutes.  The real run takes the same time plus LLM calls.
        """
        cfg = QASimilarityMultihopConfig()
        num_rows = int(kwargs.get("num_rows", NUM_ROWS))
        exhaustive = num_rows < 0

        jobs: list = kwargs.get("similarity_jobs") or []
        if not jobs:
            jobs = [{
                "QA_SIMILARITY_MULTIHOP_MODE": os.environ.get("QA_SIMILARITY_MULTIHOP_MODE", "above"),
                "QA_SIMILARITY_MULTIHOP_THRESHOLD": float(os.environ.get("QA_SIMILARITY_MULTIHOP_THRESHOLD", 0.8)),
            }]
        for idx, job in enumerate(jobs):
            job["_job_index"] = idx

        fallback_styles = _default_styles()
        collection = get_collection_from_env()

        n_chunks = collection.count()
        print(
            f"  [qa_similarity_multihop] scanning {n_chunks:,} chunks "
            f"(doc_constraint={cfg.doc_constraint}, min_docs={cfg.min_docs}, "
            f"jobs={len(jobs)})...",
            flush=True,
        )
        if n_chunks > 50_000:
            print(
                f"  [qa_similarity_multihop] ⚠  large collection — similarity scan may take a while",
                flush=True,
            )

        used_pairs: set = set()
        job_counts: dict = {}
        total_rows = 0
        stop = False

        for group, job in similarity_groups_iter(
            collection,
            num_chunks=cfg.num_chunks,
            jobs=jobs,
            doc_constraint=cfg.doc_constraint,
            min_docs=cfg.min_docs,
            used_pairs=used_pairs,
            max_candidates=cfg.max_candidates,
        ):
            job_idx = job.get("_job_index", 0)
            styles = _job_styles(job, fallback_styles)
            n = len(styles)
            job_counts[job_idx] = job_counts.get(job_idx, 0) + n
            total_rows += n
            if not exhaustive and total_rows >= num_rows:
                total_rows = min(total_rows, num_rows)
                stop = True
                break

        for idx, job in enumerate(jobs):
            mode = job.get("QA_SIMILARITY_MULTIHOP_MODE", "above")
            threshold = job.get("QA_SIMILARITY_MULTIHOP_THRESHOLD")
            smin = job.get("QA_SIMILARITY_MULTIHOP_MIN")
            smax = job.get("QA_SIMILARITY_MULTIHOP_MAX")
            min_words = job.get("QA_SIMILARITY_MULTIHOP_MIN_MATCHING_WORDS")
            styles = _job_styles(job, fallback_styles)
            if mode == "above":
                crit = f"above {threshold}"
            elif mode == "below":
                crit = f"below {threshold}" + (f", ≥{min_words} shared words" if min_words else "")
            else:
                crit = f"range {smin}–{smax}"
            groups = job_counts.get(idx, 0) // max(len(styles), 1)
            print(
                f"    job {idx} [{crit}] {len(styles)} style(s): "
                f"{groups:,} groups → {job_counts.get(idx, 0):,} rows"
            )

        if not exhaustive and stop:
            print(f"  [qa_similarity_multihop] total: {total_rows:,} rows (capped at NUM_ROWS={num_rows:,})")
        else:
            print(f"  [qa_similarity_multihop] total: {total_rows:,} rows")
        return total_rows

    def describe_prompts(self):
        sample_style = (_default_styles() or ["general"])[0]
        style_instruction = QA_GEN_VAR_STYLE_INSTRUCTIONS.get(sample_style, f"[STYLE: {sample_style}]")
        doc_summary = (
            format_doc_summary("[DOCUMENT_SUMMARY]")
            if is_summary_enabled()
            else ""
        )
        return [
            (
                "qa_similarity_multihop / positive (user)",
                QA_SIMILARITY_MULTIHOP_PROMPT_POSITIVE.format(
                    style_instruction=style_instruction,
                    doc_summary=doc_summary,
                    sample_text="[SIMILAR_CHUNKS]",
                ),
            ),
            (
                "qa_similarity_multihop / negative (user)",
                QA_SIMILARITY_MULTIHOP_PROMPT_NEGATIVE.format(
                    style_instruction=style_instruction,
                    doc_summary=doc_summary,
                    sample_text="[DISSIMILAR_CHUNKS]",
                ),
            ),
        ]

    async def run(self, input_file: str, output_file: str, **kwargs) -> None:
        cfg = QASimilarityMultihopConfig()
        num_rows: int = kwargs.get("num_rows", NUM_ROWS)
        batch_size: int = kwargs.get("batch_size", cfg.batch_size)
        exhaustive = num_rows is not None and int(num_rows) < 0

        jobs: List[Dict[str, Any]] = kwargs.get("similarity_jobs") or []
        if not jobs:
            # Single implicit job from the QA_SIMILARITY_MULTIHOP_ env block.
            jobs = [{
                "QA_SIMILARITY_MULTIHOP_MODE": os.environ.get("QA_SIMILARITY_MULTIHOP_MODE", "above"),
                "QA_SIMILARITY_MULTIHOP_THRESHOLD": float(os.environ["QA_SIMILARITY_MULTIHOP_THRESHOLD"])
                if os.environ.get("QA_SIMILARITY_MULTIHOP_THRESHOLD") else 0.8,
            }]
        # Stamp a stable index onto each job for provenance.
        for idx, job in enumerate(jobs):
            job["_job_index"] = idx

        fallback_styles = _default_styles()
        print(
            f"[qa_similarity_multihop] num_chunks={cfg.num_chunks} "
            f"doc_constraint={cfg.doc_constraint} min_docs={cfg.min_docs} jobs={len(jobs)}"
        )

        collection = get_collection_from_env()

        client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
        models = await client.models.list()
        model_id = models.data[0].id
        print(f"[qa_similarity_multihop] model: {model_id}")

        summary_map: Dict[str, str] = {}
        if is_summary_enabled():
            print("[qa_similarity_multihop] loading document summaries...")
            summary_map = load_doc_summaries()
            print(f"[qa_similarity_multihop] loaded {len(summary_map):,} document summaries")

        last_id = get_last_processed_id(output_file)
        next_row_id = last_id + 1 if last_id >= 0 else 0
        used_pairs = _rebuild_used_pairs(output_file)
        emitted_keys: set = set()

        if exhaustive:
            est = await self.estimate(
                num_rows=num_rows,
                batch_size=batch_size,
                similarity_jobs=jobs,
            )
            total = est if est is not None else 0
        else:
            total = num_rows
        pbar = tqdm(desc="[qa_similarity_multihop] generating", initial=next_row_id, total=total)

        pending: List[Dict[str, Any]] = []

        async def _flush() -> None:
            nonlocal next_row_id, pending
            if not pending:
                return
            next_row_id = await _process_batch(client, cfg, model_id, pending, next_row_id, output_file)
            pending = []
            pbar.n = min(next_row_id, total)
            pbar.refresh()

        groups = similarity_groups_iter(
            collection,
            num_chunks=cfg.num_chunks,
            jobs=jobs,
            doc_constraint=cfg.doc_constraint,
            min_docs=cfg.min_docs,
            used_pairs=used_pairs,
            max_candidates=cfg.max_candidates,
        )

        stop = False
        for group, job in groups:
            sample_id = [c["hf_row_id"] for c in group]
            group_key = get_sample_group_key(sample_id)
            # Guard against re-emitting an identical group already on disk.
            if group_key in emitted_keys:
                continue
            emitted_keys.add(group_key)

            sample_text = sample_text_from_chunks(group)
            context_text = format_sample_text_for_prompt(sample_text)
            doc_ids = [c["doc_id"] for c in group]
            doc_summary = format_doc_summaries_for_docs(summary_map, doc_ids) if summary_map else ""

            non_seed = [c["similarity"] for c in group[1:]]
            group_sim = round(sum(non_seed) / len(non_seed), 4) if non_seed else 1.0

            mode = job.get("QA_SIMILARITY_MULTIHOP_MODE", "above")
            similarity_meta = {
                "similarity_job_index": job.get("_job_index"),
                "similarity_mode": mode,
                "similarity_threshold": job.get("QA_SIMILARITY_MULTIHOP_THRESHOLD"),
                "similarity_min": job.get("QA_SIMILARITY_MULTIHOP_MIN"),
                "similarity_max": job.get("QA_SIMILARITY_MULTIHOP_MAX"),
                "min_matching_words": job.get("QA_SIMILARITY_MULTIHOP_MIN_MATCHING_WORDS"),
                "chunk_group_similarity": group_sim,
            }
            positive = _is_positive(mode)

            for style in _job_styles(job, fallback_styles):
                pending.append({
                    "sample_id": sample_id,
                    "sample_text": sample_text,
                    "context_text": context_text,
                    "style": style,
                    "positive": positive,
                    "similarity_meta": similarity_meta,
                    "doc_summary": doc_summary,
                    "document_id": doc_ids[0] if doc_ids else None,
                })
                if len(pending) >= batch_size:
                    await _flush()
                    if not exhaustive and next_row_id >= num_rows:
                        stop = True
                        break
            if stop:
                break

        await _flush()
        pbar.close()
        print(f"[qa_similarity_multihop] done — {next_row_id} QA rows → {output_file}")
