"""
Phase 4: Conversation Heuristic Filter

Applies heuristics:
  - Minimum number of messages (CONV_MIN_MESSAGES).
  - Every assistant turn has content ≥ CONV_ASSISTANT_MIN_LEN.
  - No near-duplicate user messages (Levenshtein ≤ threshold).
  - No near-duplicate adjacent messages (Levenshtein ≤ threshold).

Rows carry origin_id unchanged from Phase 1.
Resume: tracks last processed input_id so a restart skips already-done rows.

Input:  Phase 3 JSONL  (id, input_id, origin_id, sample_id, sample_text, messages)
Output: Phase 4 JSONL  (id re-numbered, input_id saved, origin_id preserved)
"""

import argparse
from typing import Dict, List

from tqdm import tqdm

from SeedDataGen.config import (
    CONV_MIN_MESSAGES,
    CONV_ASSISTANT_MIN_LEN,
    CONV_USER_LEVENSHTEIN_THRESHOLD,
    CONV_ADJACENT_LEVENSHTEIN_THRESHOLD,
    BATCH_SIZE,
    PHASE3_OUTPUT,
    PHASE4_OUTPUT,
)
from SeedDataGen.utils import (
    iter_jsonl_batches,
    write_jsonl_batch,
    get_last_processed_id,
    get_max_int_field,
    count_jsonl_lines,
    levenshtein,
)


def filter_conversation(messages: List[Dict[str, str]]) -> bool:
    if len(messages) < CONV_MIN_MESSAGES:
        return False

    for m in messages:
        if m["role"] == "assistant" and len(m["content"].strip()) < CONV_ASSISTANT_MIN_LEN:
            return False

    user_msgs = [m["content"] for m in messages if m["role"] == "user"]
    for i in range(len(user_msgs)):
        for j in range(i + 1, len(user_msgs)):
            if levenshtein(user_msgs[i], user_msgs[j]) <= CONV_USER_LEVENSHTEIN_THRESHOLD:
                return False

    for i in range(len(messages) - 1):
        if levenshtein(messages[i]["content"], messages[i + 1]["content"]) <= CONV_ADJACENT_LEVENSHTEIN_THRESHOLD:
            return False

    return True


def main(
    input_file: str = PHASE3_OUTPUT,
    output_file: str = PHASE4_OUTPUT,
    batch_size: int = BATCH_SIZE,
):
    total = count_jsonl_lines(input_file)
    print(f"Phase 4 — reading {total} conversations from {input_file}")

    last_out_id = get_last_processed_id(output_file)
    next_id = last_out_id + 1 if last_out_id >= 0 else 0

    last_input_id = get_max_int_field(output_file, "input_id")
    resume_from_input = last_input_id + 1 if last_input_id >= 0 else 0

    kept = 0
    dropped = 0
    out_buf: List[Dict] = []
    pbar = tqdm(total=total, desc="Phase 4: conversation filter")

    for batch in iter_jsonl_batches(
        input_file,
        batch_size=batch_size,
        start_from_id=resume_from_input,
    ):
        for item in batch:
            msgs = item.get("messages", [])
            if filter_conversation(msgs):
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
    print(
        f"Phase 4 done — kept {kept}, dropped {dropped} "
        f"(from {kept + dropped} input conversations) → {output_file}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4: conversation heuristic filter")
    parser.add_argument("--input", default=PHASE3_OUTPUT)
    parser.add_argument("--output", default=PHASE4_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    main(args.input, args.output, args.batch_size)
