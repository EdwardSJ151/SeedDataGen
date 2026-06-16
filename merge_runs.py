#!/usr/bin/env python3
"""
merge_runs — concatenate the final outputs of several multi-run pipelines into a
single JSONL, renumbering the monotonic `id` field so it stays unique across the
combined file.

Each row already carries a `GEN_TYPE` field identifying which generator produced
it, so provenance is preserved without any extra bookkeeping.

Usage:
    python -m SeedDataGen.merge_runs out/qa_gen_var/final.jsonl \\
        out/qa_local_multihop/final.jsonl \\
        out/qa_similarity_multihop/final.jsonl \\
        -o combined/final.jsonl

    # Globs are expanded by the shell, so this also works:
    python -m SeedDataGen.merge_runs out/*/final.jsonl -o combined/final.jsonl
"""

import argparse
import json
import os
from typing import List


def merge(inputs: List[str], output: str) -> int:
    """Concatenate *inputs* into *output*, renumbering `id`. Returns row count."""
    out_dir = os.path.dirname(output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    next_id = 0
    with open(output, "w", encoding="utf-8") as out_fh:
        for path in inputs:
            if not os.path.exists(path):
                print(f"[merge_runs] WARNING: missing input, skipping: {path}")
                continue
            n = 0
            with open(path, "r", encoding="utf-8") as in_fh:
                for line in in_fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    obj["id"] = next_id
                    next_id += 1
                    n += 1
                    out_fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
            print(f"[merge_runs] {path}: {n} rows")
    return next_id


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Concatenate and renumber multiple pipeline final outputs.",
    )
    parser.add_argument("inputs", nargs="+", help="Input JSONL files to merge (in order)")
    parser.add_argument("-o", "--output", required=True, help="Output JSONL path")
    args = parser.parse_args()

    total = merge(args.inputs, args.output)
    print(f"[merge_runs] done — {total} rows → {args.output}")


if __name__ == "__main__":
    main()
