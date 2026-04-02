#!/usr/bin/env python3
"""
SeedDataGen Pipeline Runner

Orchestrates all six phases:
  Phase 1: QA generation          (async vLLM)
  Phase 2: QA heuristic filter    (pure Python)
  Phase 3: Conversation expansion (async vLLM)
  Phase 4: Conversation filter    (pure Python)
  Phase 5: LLM judge + score gate (async vLLM)
  Phase 6: Embedding dedup        (sentence-transformers, different model)

Usage:
    # Full pipeline
    python -m SeedDataGen.run_pipeline --num-rows 10000

    # Single phase
    python -m SeedDataGen.run_pipeline --phase 3 --input seed_phase2_qa_filtered.jsonl

    # Resume from phase 4 onwards
    python -m SeedDataGen.run_pipeline --start-phase 4
"""

import argparse
import asyncio
import os
import sys

from SeedDataGen.config import (
    BATCH_SIZE,
    NUM_ROWS,
    PHASE1_OUTPUT,
    PHASE2_OUTPUT,
    PHASE3_OUTPUT,
    PHASE4_OUTPUT,
    PHASE5_OUTPUT,
    PHASE6_OUTPUT,
)


# -----------------------------------------------------------------------
# Phase runners
# -----------------------------------------------------------------------
async def run_phase1(num_rows: int, batch_size: int, output: str):
    print("\n" + "=" * 60)
    print("PHASE 1: QA Pair Generation")
    print("=" * 60)
    from SeedDataGen.phase1_qa_gen import main as p1

    await p1(output_file=output, num_rows=num_rows, batch_size=batch_size)


def run_phase2(input_file: str, batch_size: int, output: str):
    print("\n" + "=" * 60)
    print("PHASE 2: QA Heuristic Filter")
    print("=" * 60)
    from SeedDataGen.phase2_qa_filter import main as p2

    p2(input_file=input_file, output_file=output, batch_size=batch_size)


async def run_phase3(input_file: str, batch_size: int, output: str):
    print("\n" + "=" * 60)
    print("PHASE 3: Conversation Expansion")
    print("=" * 60)
    from SeedDataGen.phase3_expand import main as p3

    await p3(input_file=input_file, output_file=output, batch_size=batch_size)


def run_phase4(input_file: str, batch_size: int, output: str):
    print("\n" + "=" * 60)
    print("PHASE 4: Conversation Heuristic Filter")
    print("=" * 60)
    from SeedDataGen.phase4_conv_filter import main as p4

    p4(input_file=input_file, output_file=output, batch_size=batch_size)


async def run_phase5(input_file: str, batch_size: int, output: str):
    print("\n" + "=" * 60)
    print("PHASE 5: LLM Judge Scoring")
    print("=" * 60)
    from SeedDataGen.phase5_judge import main as p5

    await p5(input_file=input_file, output_file=output, batch_size=batch_size)


def run_phase6(input_file: str, batch_size: int, output: str):
    print("\n" + "=" * 60)
    print("PHASE 6: Embedding Similarity Filter")
    print("=" * 60)
    from SeedDataGen.phase6_embed_filter import main as p6

    p6(input_file=input_file, output_file=output, batch_size=batch_size)


# -----------------------------------------------------------------------
# Full pipeline
# -----------------------------------------------------------------------
async def run_full_pipeline(
    num_rows: int,
    batch_size: int,
    start_phase: int = 1,
    p1_out: str = PHASE1_OUTPUT,
    p2_out: str = PHASE2_OUTPUT,
    p3_out: str = PHASE3_OUTPUT,
    p4_out: str = PHASE4_OUTPUT,
    p5_out: str = PHASE5_OUTPUT,
    p6_out: str = PHASE6_OUTPUT,
):
    if start_phase <= 1:
        await run_phase1(num_rows, batch_size, p1_out)

    if start_phase <= 2:
        run_phase2(p1_out, batch_size, p2_out)

    if start_phase <= 3:
        await run_phase3(p2_out, batch_size, p3_out)

    if start_phase <= 4:
        run_phase4(p3_out, batch_size, p4_out)

    if start_phase <= 5:
        await run_phase5(p4_out, batch_size, p5_out)

    if start_phase <= 6:
        run_phase6(p5_out, batch_size, p6_out)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Final output: {p6_out}")


# -----------------------------------------------------------------------
# Single phase
# -----------------------------------------------------------------------
async def run_single_phase(
    phase: int,
    input_file: str | None,
    output_file: str | None,
    batch_size: int,
    num_rows: int,
):
    if phase == 1:
        await run_phase1(num_rows, batch_size, output_file or PHASE1_OUTPUT)
    elif phase == 2:
        run_phase2(input_file or PHASE1_OUTPUT, batch_size, output_file or PHASE2_OUTPUT)
    elif phase == 3:
        await run_phase3(input_file or PHASE2_OUTPUT, batch_size, output_file or PHASE3_OUTPUT)
    elif phase == 4:
        run_phase4(input_file or PHASE3_OUTPUT, batch_size, output_file or PHASE4_OUTPUT)
    elif phase == 5:
        await run_phase5(input_file or PHASE4_OUTPUT, batch_size, output_file or PHASE5_OUTPUT)
    elif phase == 6:
        run_phase6(input_file or PHASE5_OUTPUT, batch_size, output_file or PHASE6_OUTPUT)
    else:
        print(f"Invalid phase: {phase}. Must be 1-6.")
        sys.exit(1)


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="SeedDataGen Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
    python -m SeedDataGen.run_pipeline --num-rows 10000
    python -m SeedDataGen.run_pipeline --phase 3 --input seed_phase2_qa_filtered.jsonl
    python -m SeedDataGen.run_pipeline --start-phase 5
""",
    )
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--start-phase", type=int, default=1, choices=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--input", type=str, help="Override input file for the phase")
    parser.add_argument("--output", type=str, help="Override output file for the phase")
    parser.add_argument("--num-rows", type=int, default=NUM_ROWS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)

    args = parser.parse_args()

    print("=" * 60)
    print("SeedDataGen Pipeline")
    print("=" * 60)
    print(f"vLLM URL: {os.environ.get('VLLM_BASE_URL', 'http://localhost:8020/v1')}")
    print(f"Batch size: {args.batch_size}")
    print(f"Num rows: {args.num_rows}")

    if args.phase:
        print(f"Running: Phase {args.phase} only")
        asyncio.run(run_single_phase(
            phase=args.phase,
            input_file=args.input,
            output_file=args.output,
            batch_size=args.batch_size,
            num_rows=args.num_rows,
        ))
    else:
        print(f"Running: Phase {args.start_phase} onwards")
        asyncio.run(run_full_pipeline(
            num_rows=args.num_rows,
            batch_size=args.batch_size,
            start_phase=args.start_phase,
        ))


if __name__ == "__main__":
    main()
