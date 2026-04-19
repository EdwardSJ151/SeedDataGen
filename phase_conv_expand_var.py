"""
Phase: conv_expand_var

Style-aware conversation expansion.  Requires input rows produced by qa_gen_var
(StyledQARow), which carry a `question_style` field identifying the style used
to generate the seed QA pair.

naive_gen=True (default): Each user turn is generated with only the current conversation history and 
the next style from the cycle.

naive_gen=False:
    Same as above but each user turn also receives a list of seed questions
    already covered by earlier conversations from the same source document
    (same sample_id, lower id), instructing the model to explore uncovered
    ground while still following the style.

The style sequence per conversation:
  - The seed question already has a style (from qa_gen_var).
  - Additional turns cycle through the configured styles in order, starting
    from the style after the seed style.  The assistant never sees the style.

Role:   EDITOR
Input:  StyledQARow  (requires question_style field)
Output: ConversationRow

Config env prefix: CONV_EXPAND_VAR_
"""

import asyncio
import random
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from tqdm import tqdm

from SeedDataGen.base_phase import Phase, PhaseRole
from SeedDataGen.config import STOP_STRINGS, VLLM_API_KEY, VLLM_BASE_URL
from SeedDataGen.prompts import (
    ASSISTANT_TURN_PROMPT,
    QA_GEN_VAR_STYLE_INSTRUCTIONS,
    USER_TURN_VAR_DIVERSITY_PROMPT,
    USER_TURN_VAR_PROMPT,
)
from SeedDataGen.registry import register
from SeedDataGen.schemas import ConversationRow, StyledQARow
from SeedDataGen.utils import (
    count_jsonl_lines,
    format_conversation_history,
    format_user_history,
    get_last_processed_id,
    get_max_int_field,
    iter_jsonl_batches,
    write_jsonl_batch,
)


class ConvExpandVarConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CONV_EXPAND_VAR_", env_file=".env", extra="ignore", enable_decoding=False)

    naive_gen: bool = True
    question_styles: List[str] = ["general", "specific", "compositional", "comparative"]
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

    @field_validator("question_styles", mode="before")
    @classmethod
    def _parse_styles(cls, v):
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v


# Style cycling helper
def _style_sequence(seed_style: str, styles: List[str], n_extra_turns: int) -> List[str]:
    """
    Return styles for n_extra_turns additional user turns, starting from the
    style immediately after seed_style in the cycle.
    """
    if seed_style in styles:
        start_idx = (styles.index(seed_style) + 1) % len(styles)
    else:
        start_idx = 0
    return [styles[(start_idx + i) % len(styles)] for i in range(n_extra_turns)]


# LLM helpers
async def _generate_user_turn(
    client: AsyncOpenAI,
    model_id: str,
    sample_text: str,
    history_str: str,
    style: str,
    *,
    previous_questions: Optional[List[str]],
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> Optional[str]:
    """
    Simulated user turn.  Style instruction is always injected.
    When previous_questions is provided (non-naive mode, not the first pair),
    the diversity prompt is used instead of the plain style prompt so the model
    is also steered away from already-covered ground.
    The assistant never sees this prompt.
    """
    style_instruction = QA_GEN_VAR_STYLE_INSTRUCTIONS.get(style, "")

    if previous_questions:
        prompt = USER_TURN_VAR_DIVERSITY_PROMPT.format(
            style_instruction=style_instruction,
            sample_text=sample_text,
            previous_questions="\n".join(f"- {q}" for q in previous_questions),
            conversation_history=history_str,
        )
    else:
        prompt = USER_TURN_VAR_PROMPT.format(
            style_instruction=style_instruction,
            sample_text=sample_text,
            conversation_history=history_str,
        )

    try:
        resp = await client.chat.completions.create(
            model=model_id,
            messages=[{"role": "system", "content": prompt}],
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
        print(f"[conv_expand_var] user turn LLM error: {e}")
        return None


async def _generate_assistant_turn(
    client: AsyncOpenAI,
    model_id: str,
    sample_text: str,
    history_str: str,
    *,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> Optional[str]:
    """Assistant turn — only sees the conversation history, never the style."""
    prompt = ASSISTANT_TURN_PROMPT.format(
        sample_text=sample_text,
        conversation_history=history_str,
    )
    try:
        resp = await client.chat.completions.create(
            model=model_id,
            messages=[{"role": "system", "content": prompt}],
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
        print(f"[conv_expand_var] assistant turn LLM error: {e}")
        return None


# Single conversation expansion
async def _expand_conversation(
    client: AsyncOpenAI,
    cfg: ConvExpandVarConfig,
    model_id: str,
    sample_text: str,
    qa: Dict[str, str],
    seed_style: str,
    *,
    previous_questions: Optional[List[str]] = None,
) -> Optional[List[Dict[str, str]]]:
    messages: List[Dict[str, str]] = [
        {"role": "user", "content": qa["question"]},
        {"role": "assistant", "content": qa["answer"]},
    ]

    n_turns = random.randint(cfg.n_user_turns_min, cfg.n_user_turns_max)
    styles = _style_sequence(seed_style, cfg.question_styles, n_turns)
    use_diversity = not cfg.naive_gen and previous_questions

    for style in styles:
        user_history_str = format_user_history(messages)
        user_msg = await _generate_user_turn(
            client, model_id,
            sample_text, user_history_str, style,
            previous_questions=previous_questions if use_diversity else None,
            temperature=cfg.user_turn_temperature,
            top_p=cfg.user_turn_top_p,
            max_tokens=cfg.user_turn_max_tokens,
        )
        if not user_msg:
            break
        messages.append({"role": "user", "content": user_msg})

        full_history_str = format_conversation_history(messages)
        asst_msg = await _generate_assistant_turn(
            client, model_id,
            sample_text, full_history_str,
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
    cfg: ConvExpandVarConfig,
    model_id: str,
    batch: List[Dict[str, Any]],
    next_id: int,
    output_file: str,
) -> tuple[int, int, int]:
    sem = asyncio.Semaphore(cfg.max_concurrent)

    if cfg.naive_gen:
        async def _one_naive(item: Dict[str, Any]):
            async with sem:
                return await _expand_conversation(
                    client, cfg, model_id,
                    item["sample_text"],
                    {"question": item["question"], "answer": item["answer"]},
                    item["question_style"],
                )

        results = await asyncio.gather(*[_one_naive(it) for it in batch])
        items_ordered = batch

    else:
        # Group by sample_id, sort by id so earlier pairs are processed first.
        groups: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for item in batch:
            groups[item["sample_id"]].append(item)
        for g in groups.values():
            g.sort(key=lambda x: x["id"])

        items_ordered = []
        previous_by_sample: Dict[int, List[str]] = defaultdict(list)
        for sid in sorted(groups):
            for item in groups[sid]:
                items_ordered.append((item, list(previous_by_sample[sid])))
                previous_by_sample[sid].append(item["question"])

        async def _one_div(item: Dict[str, Any], prev_qs: List[str]):
            async with sem:
                return await _expand_conversation(
                    client, cfg, model_id,
                    item["sample_text"],
                    {"question": item["question"], "answer": item["answer"]},
                    item["question_style"],
                    previous_questions=prev_qs if prev_qs else None,
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
class ConvExpandVarPhase(Phase):
    name = "conv_expand_var"
    role = PhaseRole.EDITOR
    input_schema = StyledQARow
    output_schema = ConversationRow

    def describe_prompts(self):
        cfg = ConvExpandVarConfig()
        results = []

        for style in cfg.question_styles:
            style_instruction = QA_GEN_VAR_STYLE_INSTRUCTIONS.get(style, f"[UNKNOWN STYLE: {style}]")

            user_naive = USER_TURN_VAR_PROMPT.format(
                style_instruction=style_instruction,
                sample_text="[SAMPLE_TEXT]",
                conversation_history="[PREVIOUS_USER_QUESTIONS]",
            )
            results.append((f"conv_expand_var / user turn — naive, style={style} (system)", user_naive))

            if not cfg.naive_gen:
                user_div = USER_TURN_VAR_DIVERSITY_PROMPT.format(
                    style_instruction=style_instruction,
                    sample_text="[SAMPLE_TEXT]",
                    previous_questions="[SEED_QUESTIONS_FROM_OTHER_CONVERSATIONS]",
                    conversation_history="[PREVIOUS_USER_QUESTIONS]",
                )
                results.append((f"conv_expand_var / user turn — diversity, style={style} (system)", user_div))

        assistant = ASSISTANT_TURN_PROMPT.format(
            sample_text="[SAMPLE_TEXT]",
            conversation_history="[FULL_CONVERSATION_HISTORY]",
        )
        results.append(("conv_expand_var / assistant turn (system)", assistant))

        return results

    async def run(self, input_file: str, output_file: str, **kwargs) -> None:
        cfg = ConvExpandVarConfig()
        batch_size: int = kwargs.get("batch_size", cfg.batch_size)

        unknown = [s for s in cfg.question_styles if s not in QA_GEN_VAR_STYLE_INSTRUCTIONS]
        if unknown:
            raise ValueError(
                f"[conv_expand_var] Unknown styles: {unknown}. "
                f"Available: {list(QA_GEN_VAR_STYLE_INSTRUCTIONS)}"
            )

        if not cfg.naive_gen and cfg.n_user_turns_max <= 1:
            warnings.warn(
                "[conv_expand_var] naive_gen=False requires n_user_turns_max >= 2 "
                "(need at least 2 QA pairs per sample for diversity context). "
                "Falling back to naive_gen=True.",
                stacklevel=2,
            )
            cfg = cfg.model_copy(update={"naive_gen": True})

        client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
        models = await client.models.list()
        model_id = models.data[0].id
        mode = "naive" if cfg.naive_gen else "diversity"
        print(f"[conv_expand_var] model: {model_id}  mode: {mode}")
        print(f"[conv_expand_var] style cycle: {cfg.question_styles}")

        last_out_id = get_last_processed_id(output_file)
        next_id = last_out_id + 1 if last_out_id >= 0 else 0

        last_input_id = get_max_int_field(output_file, "input_id")
        resume_from = last_input_id + 1 if last_input_id >= 0 else 0

        total = count_jsonl_lines(input_file)
        pbar = tqdm(total=total, desc="[conv_expand_var]")

        total_seen = 0
        total_kept = 0
        for batch in iter_jsonl_batches(
            input_file,
            batch_size=batch_size,
            start_from_id=resume_from,
            required_fields=["sample_text", "question", "answer", "question_style"],
        ):
            total_seen += len(batch)
            next_id, _, n_kept = await _process_batch(client, cfg, model_id, batch, next_id, output_file)
            total_kept += n_kept
            pbar.update(len(batch))

        pbar.close()
        dropped = total_seen - total_kept
        print(
            f"[conv_expand_var] done — kept {total_kept}, dropped {dropped} "
            f"(from {total_seen} StyledQA rows) → {output_file}"
        )
