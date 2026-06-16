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

from typing import Dict, List

from pydantic_settings import BaseSettings, SettingsConfigDict
from tqdm import tqdm

from SeedDataGen.base_phase import Phase, PhaseRole
from SeedDataGen.registry import register
from SeedDataGen.schemas import ConversationRow
from SeedDataGen.utils import (
    count_jsonl_lines,
    get_last_processed_id,
    get_max_int_field,
    iter_jsonl_batches,
    levenshtein,
    write_jsonl_batch,
)


class ConvFilterConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CONV_FILTER_", env_file=".env", extra="ignore")

    min_messages: int = 4
    assistant_min_len: int = 10
    user_levenshtein_threshold: int = 20
    adjacent_levenshtein_threshold: int = 20
    batch_size: int = 32


def _filter_conversation(messages: List[Dict[str, str]], cfg: ConvFilterConfig) -> bool:
    if len(messages) < cfg.min_messages:
        return False

    for m in messages:
        if m["role"] == "assistant" and len(m["content"].strip()) < cfg.assistant_min_len:
            return False

    user_msgs = [m["content"] for m in messages if m["role"] == "user"]
    for i in range(len(user_msgs)):
        for j in range(i + 1, len(user_msgs)):
            if levenshtein(user_msgs[i], user_msgs[j]) <= cfg.user_levenshtein_threshold:
                return False

    for i in range(len(messages) - 1):
        if levenshtein(messages[i]["content"], messages[i + 1]["content"]) <= cfg.adjacent_levenshtein_threshold:
            return False

    return True


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

        for batch in iter_jsonl_batches(input_file, batch_size=batch_size, start_from_id=resume_from):
            for item in batch:
                msgs = item.get("messages", [])
                if _filter_conversation(msgs, cfg):
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
