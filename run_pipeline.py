#!/usr/bin/env python3
"""
SeedDataGen Pipeline Runner

Loads the pipeline definition from pipeline.yaml, validates phase wiring
(role transitions + schema compatibility), then runs each phase in order.
Phase numbers are assigned dynamically from the order they appear in the YAML.

Usage:
    # Full pipeline (pipeline.yaml default)
    python -m SeedDataGen.run_pipeline --num-rows 10000

    # Full pipeline with a custom YAML
    python -m SeedDataGen.run_pipeline --pipeline my_pipeline.yaml

    # Start from a specific phase by name
    python -m SeedDataGen.run_pipeline --start-from qa_filter

    # Run a single phase only
    python -m SeedDataGen.run_pipeline --only judge --input conv_filtered.jsonl

    # List all registered phases
    python -m SeedDataGen.run_pipeline --list-phases

    # Dump all prompts to a text file (no LLM calls)
    python -m SeedDataGen.run_pipeline --dump-prompts prompts.txt
    python -m SeedDataGen.run_pipeline --pipeline pipeline_myver.yaml --dump-prompts prompts_myver.txt
"""

import argparse
import asyncio
import contextlib
import importlib
import os
import sys
from pathlib import Path
from typing import Any, Iterator, Optional

import yaml

from SeedDataGen.base_phase import Phase
from SeedDataGen.config import BATCH_SIZE, NUM_ROWS, VLLM_BASE_URL
from SeedDataGen.registry import get_phase, list_phases

# Auto-discover and import all phase_*.py modules so their @register
# decorators fire before we look anything up in the registry.
_PHASE_PACKAGE = "SeedDataGen"
_PHASE_DIR = Path(__file__).resolve().parent


def _import_all_phases() -> None:
    for path in sorted(_PHASE_DIR.glob("phase_*.py")):
        module_name = f"{_PHASE_PACKAGE}.{path.stem}"
        if module_name not in sys.modules:
            importlib.import_module(module_name)


# YAML loading
def _load_pipeline_yaml(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    entries = data.get("pipeline")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"pipeline.yaml must contain a non-empty 'pipeline' list. Got: {data!r}")
    for i, entry in enumerate(entries):
        if "phase" not in entry:
            raise ValueError(f"Pipeline entry {i} is missing the 'phase' key: {entry!r}")
        if "output" not in entry:
            raise ValueError(f"Pipeline entry {i} (phase='{entry['phase']}') is missing the 'output' key.")
    return entries


# Per-phase config override (YAML config: block → temporary env vars)
@contextlib.contextmanager
def _phase_env(overrides: dict[str, Any]) -> Iterator[None]:
    """
    Temporarily set environment variables from a YAML config: block, then
    restore the previous values on exit.  Keys are uppercased automatically
    so the YAML can use lowercase for readability.
    """
    if not overrides:
        yield
        return

    prev: dict[str, str | None] = {}
    for key, val in overrides.items():
        env_key = str(key).upper()
        prev[env_key] = os.environ.get(env_key)
        os.environ[env_key] = str(val)
    try:
        yield
    finally:
        for env_key, old_val in prev.items():
            if old_val is None:
                os.environ.pop(env_key, None)
            else:
                os.environ[env_key] = old_val


# Wiring validation
def _build_and_validate(entries: list[dict[str, Any]]) -> list[Phase]:
    phases: list[Phase] = []
    for entry in entries:
        cls = get_phase(entry["phase"])
        phases.append(cls())

    for i in range(1, len(phases)):
        prev = phases[i - 1]
        curr = phases[i]
        force = entries[i].get("force", False)
        curr.check_compatible_with(prev, force=force)

    return phases


# Prompt dump
def _dump_prompts(
    entries: list[dict[str, Any]],
    phases: list[Phase],
    output_file: str,
) -> None:
    """
    For every phase in the pipeline, apply its YAML config overrides, call
    describe_prompts(), and write all rendered prompts to *output_file*.
    No LLM calls or file I/O is performed.
    """
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("SEEDATAGEN — PIPELINE PROMPT DUMP")
    lines.append("=" * 70)

    for step_num, (entry, phase) in enumerate(zip(entries, phases), start=1):
        cfg_overrides: dict[str, Any] = entry.get("config", {}) or {}

        with _phase_env(cfg_overrides):
            prompts = phase.describe_prompts()

        header = f"Phase {step_num}: {phase.name}  (role={phase.role.value})"
        if cfg_overrides:
            header += f"  config={cfg_overrides}"
        lines.append("")
        lines.append("=" * 70)
        lines.append(header)
        lines.append("=" * 70)

        if not prompts:
            lines.append("  [no LLM prompts — heuristic/embedding phase]")
        else:
            for label, prompt_text in prompts:
                lines.append("")
                lines.append(f"--- {label} ---")
                lines.append(prompt_text)

    lines.append("")
    lines.append("=" * 70)
    lines.append(f"Total phases: {len(phases)}")
    lines.append("=" * 70)

    output = "\n".join(lines)
    with open(output_file, "w", encoding="utf-8") as fh:
        fh.write(output)

    print(f"Prompts written to: {output_file}")


# Runner
async def run_pipeline(
    entries: list[dict[str, Any]],
    phases: list[Phase],
    *,
    start_from: Optional[str] = None,
    only: Optional[str] = None,
    input_override: Optional[str] = None,
    output_override: Optional[str] = None,
    num_rows: int = NUM_ROWS,
    batch_size: int = BATCH_SIZE,
) -> None:

    if only is not None:
        # Single-phase mode
        target_idx = next((i for i, p in enumerate(phases) if p.name == only), None)
        if target_idx is None:
            names = [p.name for p in phases]
            raise ValueError(f"Phase '{only}' not found in pipeline. Available: {names}")

        entry = entries[target_idx]
        phase = phases[target_idx]
        step_num = target_idx + 1
        in_file = input_override or (entries[target_idx - 1]["output"] if target_idx > 0 else "")
        out_file = output_override or entry["output"]

        cfg_overrides: dict[str, Any] = entry.get("config", {}) or {}

        print(f"\n{'=' * 60}")
        print(f"Phase {step_num}: {phase.name}  [single-phase mode]")
        print(f"  input : {in_file}")
        print(f"  output: {out_file}")
        if cfg_overrides:
            print(f"  config: {cfg_overrides}")
        print(f"{'=' * 60}")

        with _phase_env(cfg_overrides):
            await phase.run(
                input_file=in_file,
                output_file=out_file,
                num_rows=num_rows,
                batch_size=batch_size,
            )
        return

    # Full / partial pipeline
    start_idx = 0
    if start_from is not None:
        start_idx = next((i for i, p in enumerate(phases) if p.name == start_from), None)
        if start_idx is None:
            names = [p.name for p in phases]
            raise ValueError(f"Phase '{start_from}' not found in pipeline. Available: {names}")

    for step_num, (entry, phase) in enumerate(zip(entries, phases), start=1):
        if step_num - 1 < start_idx:
            continue

        # Input is the previous step's output, except for the very first step
        # or when the user overrides.
        if step_num - 1 == start_idx and input_override:
            in_file = input_override
        elif step_num > 1:
            in_file = entries[step_num - 2]["output"]
        else:
            in_file = ""  # generators don't use input_file

        out_file = entry["output"]

        cfg_overrides = entry.get("config", {}) or {}

        print(f"\n{'=' * 60}")
        print(f"Phase {step_num}: {phase.name}")
        if in_file:
            print(f"  input : {in_file}")
        print(f"  output: {out_file}")
        if cfg_overrides:
            print(f"  config: {cfg_overrides}")
        print(f"{'=' * 60}")

        with _phase_env(cfg_overrides):
            await phase.run(
                input_file=in_file,
                output_file=out_file,
                num_rows=num_rows,
                batch_size=batch_size,
            )

    print(f"\n{'=' * 60}")
    print("PIPELINE COMPLETE")
    print(f"{'=' * 60}")
    final_output = entries[-1]["output"]
    print(f"Final output: {final_output}")


# CLI
def main() -> None:
    _import_all_phases()

    parser = argparse.ArgumentParser(
        description="SeedDataGen Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python -m SeedDataGen.run_pipeline --num-rows 10000
  python -m SeedDataGen.run_pipeline --start-from qa_filter
  python -m SeedDataGen.run_pipeline --only judge --input conv_filtered.jsonl
  python -m SeedDataGen.run_pipeline --pipeline custom.yaml
  python -m SeedDataGen.run_pipeline --list-phases
  python -m SeedDataGen.run_pipeline --dump-prompts prompts.txt
  python -m SeedDataGen.run_pipeline --pipeline pipeline_myver.yaml --dump-prompts prompts_myver.txt
""",
    )
    parser.add_argument(
        "--pipeline",
        default=str(_PHASE_DIR / "pipeline.yaml"),
        help="Path to pipeline YAML (default: SeedDataGen/pipeline.yaml)",
    )
    parser.add_argument(
        "--start-from",
        metavar="PHASE_NAME",
        help="Resume from this phase (by name) onwards",
    )
    parser.add_argument(
        "--only",
        metavar="PHASE_NAME",
        help="Run only this single phase",
    )
    parser.add_argument("--input", metavar="FILE", help="Override input file for the first/only phase")
    parser.add_argument("--output", metavar="FILE", help="Override output file (only phase mode)")
    parser.add_argument("--num-rows", type=int, default=NUM_ROWS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--list-phases",
        action="store_true",
        help="Print all registered phase names and exit",
    )
    parser.add_argument(
        "--dump-prompts",
        metavar="OUTPUT_TXT",
        help="Render all pipeline prompts with placeholder values and write to a .txt file (no LLM calls)",
    )

    args = parser.parse_args()

    if args.list_phases:
        print("Registered phases:")
        for name in list_phases():
            print(f"  {name}")
        return

    print("=" * 60)
    print("SeedDataGen Pipeline")
    print("=" * 60)
    print(f"vLLM URL  : {os.environ.get('VLLM_BASE_URL', VLLM_BASE_URL)}")
    print(f"Pipeline  : {args.pipeline}")
    print(f"Batch size: {args.batch_size}")
    print(f"Num rows  : {args.num_rows}")
    if args.start_from:
        print(f"Start from: {args.start_from}")
    if args.only:
        print(f"Only phase: {args.only}")

    entries = _load_pipeline_yaml(args.pipeline)
    phases = _build_and_validate(entries)

    if args.dump_prompts:
        _dump_prompts(entries, phases, args.dump_prompts)
        return

    asyncio.run(
        run_pipeline(
            entries,
            phases,
            start_from=args.start_from,
            only=args.only,
            input_override=args.input,
            output_override=args.output,
            num_rows=args.num_rows,
            batch_size=args.batch_size,
        )
    )


if __name__ == "__main__":
    main()
