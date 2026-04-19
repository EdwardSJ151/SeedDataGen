"""
Phase: judge

Scores each conversation using the LLM judge prompt (five dimensions),
computes the average score, and keeps only rows where avg_score > min_avg_score.

Role:   JUDGE
Input:  ConversationRow
Output: JudgedConversationRow  (adds scores dict and avg_score)
"""

import asyncio
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
from pydantic_settings import BaseSettings, SettingsConfigDict
from tqdm import tqdm

from SeedDataGen.base_phase import Phase, PhaseRole
from SeedDataGen.config import STOP_STRINGS, VLLM_API_KEY, VLLM_BASE_URL
from SeedDataGen.prompts import JUDGE_PROMPT
from SeedDataGen.registry import register
from SeedDataGen.schemas import ConversationRow, JudgedConversationRow
from SeedDataGen.utils import (
    count_jsonl_lines,
    format_conversation_for_judge,
    get_last_processed_id,
    get_max_int_field,
    iter_jsonl_batches,
    parse_judge_scores,
    write_jsonl_batch,
)


class JudgeConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="JUDGE_", env_file=".env", extra="ignore")

    temperature: float = 0.0
    top_p: float = 0.9
    max_tokens: int = 2048
    min_avg_score: float = 4.0
    batch_size: int = 32
    max_concurrent: int = 64


async def _judge_conversation(
    client: AsyncOpenAI,
    cfg: JudgeConfig,
    model_id: str,
    sample_text: str,
    messages: List[Dict[str, str]],
) -> Optional[str]:
    conv_str = format_conversation_for_judge(messages)
    prompt = JUDGE_PROMPT.format(sample_text=sample_text, conversation=conv_str)
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
        print(f"[judge] LLM error: {e}")
        return None


async def _process_batch(
    client: AsyncOpenAI,
    cfg: JudgeConfig,
    model_id: str,
    batch: List[Dict[str, Any]],
    next_id: int,
    output_file: str,
) -> tuple[int, int, int]:
    sem = asyncio.Semaphore(cfg.max_concurrent)

    async def _one(item: Dict[str, Any]):
        async with sem:
            return await _judge_conversation(client, cfg, model_id, item["sample_text"], item["messages"])

    raw_outputs = await asyncio.gather(*[_one(it) for it in batch])

    rows: List[Dict[str, Any]] = []
    max_input_id = -1
    for item, raw in zip(batch, raw_outputs):
        if item["id"] > max_input_id:
            max_input_id = item["id"]
        if raw is None:
            continue
        scores = parse_judge_scores(raw)
        if scores is None:
            continue
        avg = sum(scores) / len(scores)
        if avg <= cfg.min_avg_score:
            continue
        item["input_id"] = item["id"]
        item["id"] = next_id
        item["scores"] = {
            "fidelidade": scores[0],
            "correcao": scores[1],
            "clareza": scores[2],
            "coerencia": scores[3],
            "diversidade": scores[4],
        }
        item["avg_score"] = round(avg, 2)
        rows.append(item)
        next_id += 1

    n_kept = len(rows)
    if rows:
        write_jsonl_batch(output_file, rows)
    return next_id, max_input_id, n_kept


@register
class JudgePhase(Phase):
    name = "judge"
    role = PhaseRole.JUDGE
    input_schema = ConversationRow
    output_schema = JudgedConversationRow

    def describe_prompts(self):
        prompt = JUDGE_PROMPT.format(
            sample_text="[SAMPLE_TEXT]",
            conversation="[CONVERSATION]",
        )
        return [("judge / scoring prompt (user)", prompt)]

    async def run(self, input_file: str, output_file: str, **kwargs) -> None:
        cfg = JudgeConfig()
        batch_size: int = kwargs.get("batch_size", cfg.batch_size)

        client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
        models = await client.models.list()
        model_id = models.data[0].id
        print(f"[judge] model: {model_id}")

        last_out_id = get_last_processed_id(output_file)
        next_id = last_out_id + 1 if last_out_id >= 0 else 0

        last_input_id = get_max_int_field(output_file, "input_id")
        resume_from = last_input_id + 1 if last_input_id >= 0 else 0

        total = count_jsonl_lines(input_file)
        pbar = tqdm(total=total, desc="[judge]")

        total_seen = 0
        total_kept = 0
        for batch in iter_jsonl_batches(
            input_file,
            batch_size=batch_size,
            start_from_id=resume_from,
            required_fields=["sample_text", "messages"],
        ):
            total_seen += len(batch)
            next_id, _, n_kept = await _process_batch(client, cfg, model_id, batch, next_id, output_file)
            total_kept += n_kept
            pbar.update(len(batch))

        pbar.close()
        dropped = total_seen - total_kept
        print(
            f"[judge] done — kept {total_kept}, dropped {dropped} "
            f"(avg > {cfg.min_avg_score}) → {output_file}"
        )
