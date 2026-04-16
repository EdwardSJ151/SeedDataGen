"""
Phase: qa_filter

Pure-Python heuristic filter applied to QA pairs.

For each sample_id group:
  - Drop QA pairs whose answer is shorter than min_answer_len.
  - Drop near-duplicate questions (Levenshtein distance ≤ levenshtein_threshold).

Role:   FILTER
Input:  QARow
Output: QARow  (re-numbered ids; input_id preserved)
"""

from collections import defaultdict
from typing import Dict, List

from pydantic_settings import BaseSettings, SettingsConfigDict
from tqdm import tqdm

from SeedDataGen.base_phase import Phase, PhaseRole
from SeedDataGen.registry import register
from SeedDataGen.schemas import QARow
from SeedDataGen.utils import (
    count_jsonl_lines,
    get_last_processed_id,
    get_max_int_field,
    iter_jsonl_batches,
    levenshtein,
    write_jsonl_batch,
)


class QAFilterConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="QA_FILTER_", env_file=".env", extra="ignore")

    min_answer_len: int = 10
    levenshtein_threshold: int = 20
    batch_size: int = 32


def _filter_qa_group(qa_rows: List[Dict], cfg: QAFilterConfig) -> List[Dict]:
    filtered: List[Dict] = []
    for qa in qa_rows:
        if len(qa["answer"].strip()) < cfg.min_answer_len:
            continue
        is_dup = False
        for other in filtered:
            if levenshtein(qa["question"], other["question"]) <= cfg.levenshtein_threshold:
                is_dup = True
                break
        if not is_dup:
            filtered.append(qa)
    return filtered


@register
class QAFilterPhase(Phase):
    name = "qa_filter"
    role = PhaseRole.FILTER
    input_schema = QARow
    output_schema = QARow

    async def run(self, input_file: str, output_file: str, **kwargs) -> None:
        cfg = QAFilterConfig()
        batch_size: int = kwargs.get("batch_size", cfg.batch_size)

        total = count_jsonl_lines(input_file)
        print(f"[qa_filter] reading {total} rows from {input_file}")

        last_out_id = get_last_processed_id(output_file)
        next_id = last_out_id + 1 if last_out_id >= 0 else 0

        last_input_id = get_max_int_field(output_file, "input_id")
        resume_from = last_input_id + 1 if last_input_id >= 0 else 0

        groups: Dict[int, List[Dict]] = defaultdict(list)
        for batch in iter_jsonl_batches(input_file, batch_size=batch_size, start_from_id=resume_from):
            for row in batch:
                groups[row["sample_id"]].append(row)

        written = 0
        pbar = tqdm(total=len(groups), desc="[qa_filter]")
        out_buf: List[Dict] = []

        for sample_id in sorted(groups.keys()):
            kept = _filter_qa_group(groups[sample_id], cfg)
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
        print(f"[qa_filter] done — {written} rows kept → {output_file}")
