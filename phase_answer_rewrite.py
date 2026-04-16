"""
Phase: answer_rewrite  (optional)

For each QA row, asks the model to rewrite the answer to be more
information-rich, using only information already present in the source text.

Role:   EDITOR
Input:  QARow
Output: QARow  (same schema; answer field replaced with rewritten version)
"""

import asyncio
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
from pydantic_settings import BaseSettings, SettingsConfigDict
from tqdm import tqdm

from SeedDataGen.base_phase import Phase, PhaseRole
from SeedDataGen.config import STOP_STRINGS, VLLM_API_KEY, VLLM_BASE_URL
from SeedDataGen.prompts import ANSWER_REWRITE_PROMPT
from SeedDataGen.registry import register
from SeedDataGen.schemas import QARow
from SeedDataGen.utils import (
    count_jsonl_lines,
    get_last_processed_id,
    get_max_int_field,
    iter_jsonl_batches,
    write_jsonl_batch,
)


class AnswerRewriteConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ANSWER_REWRITE_", env_file=".env", extra="ignore")

    temperature: float = 0.5
    top_p: float = 0.9
    max_tokens: int = 512
    batch_size: int = 32
    max_concurrent: int = 64


async def _rewrite_answer(
    client: AsyncOpenAI,
    cfg: AnswerRewriteConfig,
    model_id: str,
    sample_text: str,
    question: str,
    answer: str,
) -> Optional[str]:
    prompt = ANSWER_REWRITE_PROMPT.format(
        sample_text=sample_text,
        question=question,
        answer=answer,
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
        print(f"[answer_rewrite] LLM error: {e}")
        return None


async def _process_batch(
    client: AsyncOpenAI,
    cfg: AnswerRewriteConfig,
    model_id: str,
    batch: List[Dict[str, Any]],
    next_id: int,
    output_file: str,
) -> tuple[int, int]:
    sem = asyncio.Semaphore(cfg.max_concurrent)

    async def _one(item: Dict[str, Any]) -> Optional[str]:
        async with sem:
            return await _rewrite_answer(
                client, cfg, model_id,
                item["sample_text"], item["question"], item["answer"],
            )

    raw_outputs = await asyncio.gather(*[_one(it) for it in batch])

    rows: List[Dict[str, Any]] = []
    max_input_id = -1
    for item, rewritten in zip(batch, raw_outputs):
        if item["id"] > max_input_id:
            max_input_id = item["id"]
        new_answer = rewritten if rewritten else item["answer"]
        rows.append({**item, "id": next_id, "input_id": item["id"], "answer": new_answer})
        next_id += 1

    if rows:
        write_jsonl_batch(output_file, rows)
    return next_id, max_input_id


@register
class AnswerRewritePhase(Phase):
    name = "answer_rewrite"
    role = PhaseRole.EDITOR
    input_schema = QARow
    output_schema = QARow

    async def run(self, input_file: str, output_file: str, **kwargs) -> None:
        cfg = AnswerRewriteConfig()
        batch_size: int = kwargs.get("batch_size", cfg.batch_size)

        client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
        models = await client.models.list()
        model_id = models.data[0].id
        print(f"[answer_rewrite] model: {model_id}")

        total = count_jsonl_lines(input_file)
        print(f"[answer_rewrite] reading {total} rows from {input_file}")

        last_out_id = get_last_processed_id(output_file)
        next_id = last_out_id + 1 if last_out_id >= 0 else 0

        last_input_id = get_max_int_field(output_file, "input_id")
        resume_from = last_input_id + 1 if last_input_id >= 0 else 0

        pbar = tqdm(total=total, desc="[answer_rewrite]")

        for batch in iter_jsonl_batches(
            input_file,
            batch_size=batch_size,
            start_from_id=resume_from,
            required_fields=["sample_text", "question", "answer"],
        ):
            next_id, _ = await _process_batch(client, cfg, model_id, batch, next_id, output_file)
            pbar.update(len(batch))

        pbar.close()
        print(f"[answer_rewrite] done → {output_file}")
