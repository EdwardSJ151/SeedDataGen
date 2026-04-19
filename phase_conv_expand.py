"""
Phase: conv_expand

Initialises a conversation from each QA pair, then expands it by generating
N additional user → assistant rounds via the LLM.

naive_gen=True (default): Each user turn is generated with only the current conversation history.

naive_gen=False:
    Each user turn is generated with the current history PLUS a list of seed
    questions already covered by earlier conversations from the same source
    document (same sample_id, lower id).  The model is instructed to explore
    aspects not yet covered.  Falls back to naive if only one QA pair exists
    for that sample (no prior conversations to reference).

Role:   EDITOR
Input:  QARow
Output: ConversationRow
"""

import asyncio
import random
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
from pydantic_settings import BaseSettings, SettingsConfigDict
from tqdm import tqdm

from SeedDataGen.base_phase import Phase, PhaseRole
from SeedDataGen.config import STOP_STRINGS, VLLM_API_KEY, VLLM_BASE_URL
from SeedDataGen.prompts import (
    ASSISTANT_TURN_PROMPT,
    USER_TURN_DIVERSITY_PROMPT,
    USER_TURN_PROMPT,
)
from SeedDataGen.registry import register
from SeedDataGen.schemas import ConversationRow, QARow
from SeedDataGen.utils import (
    count_jsonl_lines,
    format_conversation_history,
    format_user_history,
    get_last_processed_id,
    get_max_int_field,
    iter_jsonl_batches,
    write_jsonl_batch,
)


class ConvExpandConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CONV_EXPAND_", env_file=".env", extra="ignore")

    naive_gen: bool = True
    user_turn_temperature: float = 0.9
    user_turn_top_p: float = 0.95
    user_turn_max_tokens: int = 512
    assistant_turn_temperature: float = 0.7
    assistant_turn_top_p: float = 0.9
    assistant_turn_max_tokens: int = 2048
    n_user_turns_min: int = 3
    n_user_turns_max: int = 3
    batch_size: int = 32
    max_concurrent: int = 64


# LLM helpers
async def _llm_call(
    client: AsyncOpenAI,
    model_id: str,
    prompt_text: str,
    *,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> Optional[str]:
    try:
        resp = await client.chat.completions.create(
            model=model_id,
            messages=[{"role": "system", "content": prompt_text}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
                "stop": STOP_STRINGS,
            },
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"[conv_expand] LLM error: {e}")
        return None


# Conversation expansion
async def _expand_conversation(
    client: AsyncOpenAI,
    cfg: ConvExpandConfig,
    model_id: str,
    sample_text: str,
    qa: Dict[str, str],
    *,
    previous_questions: Optional[List[str]] = None,
) -> Optional[List[Dict[str, str]]]:
    """
    Expand a seed QA pair into a multi-turn conversation.

    previous_questions: seed questions from earlier conversations of the same
    sample, used only when naive_gen=False and there are prior conversations.
    """
    messages: List[Dict[str, str]] = [
        {"role": "user", "content": qa["question"]},
        {"role": "assistant", "content": qa["answer"]},
    ]
    n_turns = random.randint(cfg.n_user_turns_min, cfg.n_user_turns_max)
    use_diversity = not cfg.naive_gen and previous_questions

    for _ in range(n_turns):
        user_history_str = format_user_history(messages)

        if use_diversity:
            prev_q_str = "\n".join(f"- {q}" for q in previous_questions)
            prompt = USER_TURN_DIVERSITY_PROMPT.format(
                sample_text=sample_text,
                previous_questions=prev_q_str,
                conversation_history=user_history_str,
            )
        else:
            prompt = USER_TURN_PROMPT.format(
                sample_text=sample_text,
                conversation_history=user_history_str,
            )

        user_msg = await _llm_call(
            client, model_id, prompt,
            temperature=cfg.user_turn_temperature,
            top_p=cfg.user_turn_top_p,
            max_tokens=cfg.user_turn_max_tokens,
        )
        if not user_msg:
            break
        messages.append({"role": "user", "content": user_msg})

        full_history_str = format_conversation_history(messages)
        asst_msg = await _llm_call(
            client, model_id,
            ASSISTANT_TURN_PROMPT.format(sample_text=sample_text, conversation_history=full_history_str),
            temperature=cfg.assistant_turn_temperature,
            top_p=cfg.assistant_turn_top_p,
            max_tokens=cfg.assistant_turn_max_tokens,
        )
        if not asst_msg:
            break
        messages.append({"role": "assistant", "content": asst_msg})

    return messages


# Batch processing
async def _process_batch(
    client: AsyncOpenAI,
    cfg: ConvExpandConfig,
    model_id: str,
    batch: List[Dict[str, Any]],
    next_id: int,
    output_file: str,
) -> tuple[int, int, int]:
    sem = asyncio.Semaphore(cfg.max_concurrent)

    if cfg.naive_gen:
        # Simple path: no cross-conversation context needed.
        async def _one_naive(item: Dict[str, Any]):
            async with sem:
                return await _expand_conversation(
                    client, cfg, model_id,
                    item["sample_text"],
                    {"question": item["question"], "answer": item["answer"]},
                )

        results = await asyncio.gather(*[_one_naive(it) for it in batch])
        items_ordered = batch

    else:
        # Diversity path: group by sample_id, ordered by id so earlier pairs
        # are processed first and their questions become context for later ones.
        groups: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for item in batch:
            groups[item["sample_id"]].append(item)
        for g in groups.values():
            g.sort(key=lambda x: x["id"])

        # Build per-item previous_questions: questions from earlier items in
        # the same sample_id group (those with a lower position in sorted order).
        items_ordered = []
        previous_by_sample: Dict[int, List[str]] = defaultdict(list)
        for sid in sorted(groups):
            for item in groups[sid]:
                items_ordered.append((item, list(previous_by_sample[sid])))
                previous_by_sample[sid].append(item["question"])

        async def _one_div(item: Dict[str, Any], prev_qs: List[str]):
            async with sem:
                # Fall back to naive if this is the only QA pair for its sample.
                effective_prev = prev_qs if prev_qs else None
                return await _expand_conversation(
                    client, cfg, model_id,
                    item["sample_text"],
                    {"question": item["question"], "answer": item["answer"]},
                    previous_questions=effective_prev,
                )

        results = await asyncio.gather(*[_one_div(it, pqs) for it, pqs in items_ordered])
        items_ordered = [it for it, _ in items_ordered]

    rows: List[Dict[str, Any]] = []
    max_input_id = -1
    for item, msgs in zip(items_ordered, results):
        if item["id"] > max_input_id:
            max_input_id = item["id"]
        if msgs is None or len(msgs) < 4:
            continue
        rows.append({
            "id": next_id,
            "input_id": item["id"],
            "origin_id": item["origin_id"],
            "sample_id": item["sample_id"],
            "sample_text": item["sample_text"],
            "messages": msgs,
        })
        next_id += 1

    n_kept = len(rows)
    if rows:
        write_jsonl_batch(output_file, rows)
    return next_id, max_input_id, n_kept


# Phase class
@register
class ConvExpandPhase(Phase):
    name = "conv_expand"
    role = PhaseRole.EDITOR
    input_schema = QARow
    output_schema = ConversationRow

    def describe_prompts(self):
        cfg = ConvExpandConfig()
        results = []

        user_naive = USER_TURN_PROMPT.format(
            sample_text="[SAMPLE_TEXT]",
            conversation_history="[PREVIOUS_USER_QUESTIONS]",
        )
        results.append(("conv_expand / user turn — naive (system)", user_naive))

        if not cfg.naive_gen:
            user_div = USER_TURN_DIVERSITY_PROMPT.format(
                sample_text="[SAMPLE_TEXT]",
                previous_questions="[SEED_QUESTIONS_FROM_OTHER_CONVERSATIONS]",
                conversation_history="[PREVIOUS_USER_QUESTIONS]",
            )
            results.append(("conv_expand / user turn — diversity (system)", user_div))

        assistant = ASSISTANT_TURN_PROMPT.format(
            sample_text="[SAMPLE_TEXT]",
            conversation_history="[FULL_CONVERSATION_HISTORY]",
        )
        results.append(("conv_expand / assistant turn (system)", assistant))

        return results

    async def run(self, input_file: str, output_file: str, **kwargs) -> None:
        cfg = ConvExpandConfig()
        batch_size: int = kwargs.get("batch_size", cfg.batch_size)

        if not cfg.naive_gen and cfg.n_user_turns_max <= 1:
            warnings.warn(
                "[conv_expand] naive_gen=False requires n_user_turns_max >= 2 "
                "(need at least 2 QA pairs per sample for diversity context). "
                "Falling back to naive_gen=True.",
                stacklevel=2,
            )
            cfg = cfg.model_copy(update={"naive_gen": True})

        client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
        models = await client.models.list()
        model_id = models.data[0].id
        mode = "naive" if cfg.naive_gen else "diversity"
        print(f"[conv_expand] model: {model_id}  mode: {mode}")

        last_out_id = get_last_processed_id(output_file)
        next_id = last_out_id + 1 if last_out_id >= 0 else 0

        last_input_id = get_max_int_field(output_file, "input_id")
        resume_from = last_input_id + 1 if last_input_id >= 0 else 0

        total = count_jsonl_lines(input_file)
        pbar = tqdm(total=total, desc="[conv_expand]")

        total_seen = 0
        total_kept = 0
        for batch in iter_jsonl_batches(
            input_file,
            batch_size=batch_size,
            start_from_id=resume_from,
            required_fields=["sample_text", "question", "answer"],
        ):
            total_seen += len(batch)
            next_id, _, n_kept = await _process_batch(client, cfg, model_id, batch, next_id, output_file)
            total_kept += n_kept
            pbar.update(len(batch))

        pbar.close()
        dropped = total_seen - total_kept
        print(
            f"[conv_expand] done — kept {total_kept}, dropped {dropped} "
            f"(from {total_seen} QA rows) → {output_file}"
        )
