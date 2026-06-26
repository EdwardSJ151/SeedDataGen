"""
Phase: conv_filter

Pure-Python heuristic filter applied to conversations.

Drops a conversation if:
  - It has fewer messages than min_messages.
  - Any assistant turn is shorter than assistant_min_len characters.
  - Any two user messages are near-duplicates (Levenshtein ≤ user_levenshtein_threshold).
  - Any two adjacent messages are near-duplicates (Levenshtein ≤ adjacent_levenshtein_threshold).

Role:   FILTER
Input:  ConversationRow
Output: ConversationRow  (re-numbered ids; input_id preserved)
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict
from tqdm import tqdm

from SeedDataGen.base_phase import Phase, PhaseRole
from SeedDataGen.registry import register
from SeedDataGen.schemas import ConversationRow
from SeedDataGen.utils import (
    count_jsonl_lines,
    get_last_processed_id,
    get_max_int_field,
    is_refusal,
    is_single_turn,
    iter_jsonl_batches,
    levenshtein,
    write_jsonl_batch,
)

# Worker threads for the (CPU-bound) Levenshtein filtering.  rapidfuzz releases
# the GIL during the C edit-distance computation, so threads parallelize the
# dominant cost.  Fixed at 16 (8 cores / 16 threads); not configurable.
_FILTER_WORKERS = 16


class ConvFilterConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CONV_FILTER_", env_file=".env", extra="ignore")

    min_messages: int = 4
    assistant_min_len: int = 10
    user_levenshtein_threshold: int = 20
    adjacent_levenshtein_threshold: int = 20
    batch_size: int = 32


def _truncate_at_refusal(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Return the conversation prefix up to (but not including) the first refusing
    pair.  When an assistant turn is the deterministic refusal string, drop that
    (user, assistant) pair and everything after it, leaving a valid prefix that
    ends on an assistant turn.  Returns the input unchanged when no refusal.
    """
    for i, m in enumerate(messages):
        if m["role"] == "assistant" and is_refusal(m["content"]):
            # The user that prompted this refusal is at i-1; drop from there on.
            return messages[: max(i - 1, 0)]
    return messages


def _filter_conversation(
    messages: List[Dict[str, str]], cfg: ConvFilterConfig
) -> Optional[List[Dict[str, str]]]:
    """
    Return the (possibly truncated) messages to keep, or None to drop the
    conversation.

    The length floor adapts to the conversation's intent, read from the
    generator's raw output **before** truncation: a single-turn conversation
    (one user turn → 2 messages, e.g. dog_instruct / rewrite_gen) needs one
    pair; a multi-turn one uses ``cfg.min_messages`` (default 4). Measuring
    before truncation is what disambiguates an intended single-turn (kept) from
    a multi-turn that an early refusal collapses to one pair (dropped — it was
    multi-turn in the input, so its floor stays at ``min_messages``).
    """
    single_turn = is_single_turn(messages)
    messages = _truncate_at_refusal(messages)

    min_len = 2 if single_turn else cfg.min_messages
    if len(messages) < min_len:
        return None

    for m in messages:
        if m["role"] == "assistant" and len(m["content"].strip()) < cfg.assistant_min_len:
            return None

    user_msgs = [m["content"] for m in messages if m["role"] == "user"]
    for i in range(len(user_msgs)):
        for j in range(i + 1, len(user_msgs)):
            if levenshtein(user_msgs[i], user_msgs[j]) <= cfg.user_levenshtein_threshold:
                return None

    for i in range(len(messages) - 1):
        if levenshtein(messages[i]["content"], messages[i + 1]["content"]) <= cfg.adjacent_levenshtein_threshold:
            return None

    return messages


@register
class ConvFilterPhase(Phase):
    name = "conv_filter"
    role = PhaseRole.FILTER
    input_schema = ConversationRow
    output_schema = ConversationRow

    async def run(self, input_file: str, output_file: str, **kwargs) -> None:
        cfg = ConvFilterConfig()
        batch_size: int = kwargs.get("batch_size", cfg.batch_size)

        total = count_jsonl_lines(input_file)
        print(f"[conv_filter] reading {total} conversations from {input_file}")

        last_out_id = get_last_processed_id(output_file)
        next_id = last_out_id + 1 if last_out_id >= 0 else 0

        last_input_id = get_max_int_field(output_file, "input_id")
        resume_from = last_input_id + 1 if last_input_id >= 0 else 0

        kept = 0
        dropped = 0
        out_buf: List[Dict] = []
        pbar = tqdm(total=total, desc="[conv_filter]")

        with ThreadPoolExecutor(max_workers=_FILTER_WORKERS) as pool:
            for batch in iter_jsonl_batches(input_file, batch_size=batch_size, start_from_id=resume_from):
                # One conversation per task; pool.map preserves input order, so id
                # assignment below stays deterministic.
                kept_results = list(
                    pool.map(lambda it: _filter_conversation(it.get("messages", []), cfg), batch)
                )
                for item, kept_msgs in zip(batch, kept_results):
                    if kept_msgs is not None:
                        item["messages"] = kept_msgs
                        item["input_id"] = item["id"]
                        item["id"] = next_id
                        next_id += 1
                        out_buf.append(item)
                        kept += 1
                    else:
                        dropped += 1
                if len(out_buf) >= batch_size:
                    write_jsonl_batch(output_file, out_buf)
                    out_buf = []
                pbar.update(len(batch))

        if out_buf:
            write_jsonl_batch(output_file, out_buf)

        pbar.close()
        print(f"[conv_filter] done — kept {kept}, dropped {dropped} → {output_file}")
