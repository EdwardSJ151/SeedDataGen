"""
Phase 2: QA Pair Heuristic Filter

Pure Python — no LLM calls.

For each sample_id group:
  - Drop QA pairs whose answer is shorter than QA_MIN_ANSWER_LEN.
  - Drop near-duplicate questions (Levenshtein ≤ QA_LEVENSHTEIN_THRESHOLD).

Rows carry origin_id unchanged from Phase 1.
Resume: tracks last processed input_id so a restart skips already-done groups.

Input:  Phase 1 JSONL  (id, origin_id, sample_id, sample_text, question, answer)
Output: Phase 2 JSONL  (id re-numbered, input_id saved, origin_id preserved)
"""

import argparse
from collections import defaultdict
from typing import Dict, List

from tqdm import tqdm

from SeedDataGen.config import (
    QA_MIN_ANSWER_LEN,
    QA_LEVENSHTEIN_THRESHOLD,
    BATCH_SIZE,
    PHASE1_OUTPUT,
    PHASE2_OUTPUT,
)
from SeedDataGen.utils import (
    iter_jsonl_batches,
    write_jsonl_batch,
    count_jsonl_lines,
    get_last_processed_id,
    get_max_int_field,
    levenshtein,
)


def filter_qa_group(qa_rows: List[Dict]) -> List[Dict]:
    """Apply plan Step 2 heuristics to QA rows sharing the same sample_id."""
    filtered: List[Dict] = []
    for qa in qa_rows:
        if len(qa["answer"].strip()) < QA_MIN_ANSWER_LEN:
            continue
        is_dup = False
        for other in filtered:
            if levenshtein(qa["question"], other["question"]) <= QA_LEVENSHTEIN_THRESHOLD:
                is_dup = True
                break
        if not is_dup:
            filtered.append(qa)
    return filtered


def main(
    input_file: str = PHASE1_OUTPUT,
    output_file: str = PHASE2_OUTPUT,
    batch_size: int = BATCH_SIZE,
):
    total = count_jsonl_lines(input_file)
    print(f"Phase 2 — reading {total} rows from {input_file}")

    last_out_id = get_last_processed_id(output_file)
    next_id = last_out_id + 1 if last_out_id >= 0 else 0

    last_input_id = get_max_int_field(output_file, "input_id")
    resume_from_input = last_input_id + 1 if last_input_id >= 0 else 0

    # Group by sample_id, only loading rows not yet processed
    groups: Dict[int, List[Dict]] = defaultdict(list)
    for batch in iter_jsonl_batches(
        input_file,
        batch_size=batch_size,
        start_from_id=resume_from_input,
    ):
        for row in batch:
            groups[row["sample_id"]].append(row)

    written = 0
    pbar = tqdm(total=len(groups), desc="Phase 2: QA filter")
    out_buf: List[Dict] = []

    for sample_id in sorted(groups.keys()):
        kept = filter_qa_group(groups[sample_id])
        for row in kept:
            row["input_id"] = row["id"]
            row["id"] = next_id
            next_id += 1
            out_buf.append(row)
        if len(out_buf) >= batch_size:
            write_jsonl_batch(output_file, out_buf)
            written += len(out_buf)
            out_buf = []
        pbar.update(1)

    if out_buf:
        write_jsonl_batch(output_file, out_buf)
        written += len(out_buf)

    pbar.close()
    print(f"Phase 2 done — {written} QA rows kept → {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2: QA heuristic filter")
    parser.add_argument("--input", default=PHASE1_OUTPUT)
    parser.add_argument("--output", default=PHASE2_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    main(args.input, args.output, args.batch_size)
