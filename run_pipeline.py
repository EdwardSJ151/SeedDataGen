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

from SeedDataGen.base_phase import Phase, PhaseRole
from SeedDataGen.config import BATCH_SIZE, NUM_ROWS, VLLM_BASE_URL
from SeedDataGen.registry import get_phase, list_phases

# Auto-discover and import all phase_*.py modules so their @register
# decorators fire before we look anything up in the registry.
_PHASE_PACKAGE = "SeedDataGen"
_PHASE_DIR = Path(__file__).resolve().parent


def _import_all_phases() -> None:
    # Recursive discovery so phases living in role subpackages (generator/,
    # editor/, filter/, judge/, dedup/, preprocess/) are also registered.
    for path in sorted(_PHASE_DIR.glob("**/phase_*.py")):
        rel = path.relative_to(_PHASE_DIR).with_suffix("")
        module_name = f"{_PHASE_PACKAGE}." + ".".join(rel.parts)
        if module_name not in sys.modules:
            importlib.import_module(module_name)


# YAML loading
def _load_pipeline_yaml(path: str) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """
    Returns (entries, global_env) where:
      entries    — the 'pipeline:' list
      global_env — the optional top-level 'env:' dict (empty if absent)
    """
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
    global_env = {str(k): str(v) for k, v in (data.get("env") or {}).items()}
    return entries, global_env


def _is_multi_run_yaml(path: str) -> bool:
    """True if *path* uses the multi-run orchestrator format (a 'runs:' key)."""
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return isinstance(data, dict) and isinstance(data.get("runs"), list)


def _load_multihop_yaml(
    path: str,
) -> tuple[Optional[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    """
    Parse the multi-run YAML format.

    Returns (preprocess_entry, tail_entries, runs, global_env):
      preprocess_entry — the optional 'preprocess:' mapping (or None)
      tail_entries     — the shared 'tail:' list (phases after each generator)
      runs             — the 'runs:' list (one full pipeline per entry)
      global_env       — the optional top-level 'env:' dict
    """
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    runs = data.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError(f"Multi-run YAML must contain a non-empty 'runs' list. Got: {data!r}")

    tail = data.get("tail") or []
    if not isinstance(tail, list):
        raise ValueError("'tail' must be a list of phase entries.")
    for i, entry in enumerate(tail):
        if "phase" not in entry or "output" not in entry:
            raise ValueError(f"tail entry {i} must contain both 'phase' and 'output': {entry!r}")

    for i, run in enumerate(runs):
        if "generator" not in run:
            raise ValueError(f"runs entry {i} is missing the 'generator' key: {run!r}")
        if "output_dir" not in run:
            raise ValueError(f"runs entry {i} (generator='{run['generator']}') is missing 'output_dir'.")

    preprocess_entry = data.get("preprocess")
    if preprocess_entry is not None and "phase" not in preprocess_entry:
        raise ValueError(f"'preprocess' must contain a 'phase' key: {preprocess_entry!r}")

    global_env = {str(k): str(v) for k, v in (data.get("env") or {}).items()}
    return preprocess_entry, tail, runs, global_env


def _apply_global_env(global_env: dict[str, str]) -> None:
    """Apply top-level env: block to os.environ before phases are imported."""
    for key, val in global_env.items():
        os.environ[key] = val


# Per-phase config override (YAML config: block → temporary env vars)
@contextlib.contextmanager
def _phase_env(overrides: dict[str, Any]) -> Iterator[None]:
    """
    Temporarily set environment variables from a YAML config: block, then
    restore the previous values on exit.  Keys are used verbatim (no case
    folding) so they map exactly onto the Pydantic BaseSettings env var names.
    Non-scalar values (e.g. lists/dicts) are skipped — those are passed to
    phases as kwargs instead.
    """
    if not overrides:
        yield
        return

    prev: dict[str, str | None] = {}
    for key, val in overrides.items():
        if isinstance(val, (list, dict)):
            continue
        env_key = str(key)
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


def _dump_prompts_multi(
    preprocess_entry: Optional[dict[str, Any]],
    tail_entries: list[dict[str, Any]],
    runs: list[dict[str, Any]],
    output_file: str,
) -> None:
    """Render prompts for the preprocess phase and every run (generator + tail)."""
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("SEEDATAGEN — MULTI-RUN PROMPT DUMP")
    lines.append("=" * 70)

    def _emit(entry: dict[str, Any], phase: Phase, header: str) -> None:
        cfg_overrides: dict[str, Any] = entry.get("config", {}) or {}
        with _phase_env(cfg_overrides):
            prompts = phase.describe_prompts()
        lines.append("")
        lines.append("=" * 70)
        lines.append(header)
        if cfg_overrides:
            lines.append(f"  config={cfg_overrides}")
        lines.append("=" * 70)
        if not prompts:
            lines.append("  [no LLM prompts — heuristic/embedding/preprocess phase]")
        else:
            for label, prompt_text in prompts:
                lines.append("")
                lines.append(f"--- {label} ---")
                lines.append(prompt_text)

    if preprocess_entry:
        cls = get_phase(preprocess_entry["phase"])
        _emit(preprocess_entry, cls(), f"PREPROCESS: {preprocess_entry['phase']}")

    for run_idx, run in enumerate(runs, start=1):
        entries, run_config = _build_run_entries(run, tail_entries)
        with _phase_env(run_config):
            for entry in entries:
                cls = get_phase(entry["phase"])
                phase = cls()
                _emit(entry, phase, f"RUN {run_idx}: {entry['phase']} (role={phase.role.value})")

    lines.append("")
    lines.append("=" * 70)
    lines.append(f"Total runs: {len(runs)}")
    lines.append("=" * 70)

    with open(output_file, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
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

        extra_kwargs: dict[str, Any] = entry.get("kwargs", {}) or {}
        with _phase_env(cfg_overrides):
            await phase.run(
                input_file=in_file,
                output_file=out_file,
                num_rows=num_rows,
                batch_size=batch_size,
                **extra_kwargs,
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

        extra_kwargs = entry.get("kwargs", {}) or {}
        with _phase_env(cfg_overrides):
            await phase.run(
                input_file=in_file,
                output_file=out_file,
                num_rows=num_rows,
                batch_size=batch_size,
                **extra_kwargs,
            )

    print(f"\n{'=' * 60}")
    print("PIPELINE COMPLETE")
    print(f"{'=' * 60}")
    final_output = entries[-1]["output"]
    print(f"Final output: {final_output}")


# Multi-run orchestrator
async def _run_preprocess(entry: dict[str, Any], batch_size: int) -> None:
    cls = get_phase(entry["phase"])
    phase = cls()
    cfg_overrides: dict[str, Any] = entry.get("config", {}) or {}
    print(f"\n{'=' * 60}")
    print(f"PREPROCESS: {phase.name}")
    if cfg_overrides:
        print(f"  config: {cfg_overrides}")
    print(f"{'=' * 60}")
    with _phase_env(cfg_overrides):
        await phase.run(input_file="", output_file="", num_rows=0, batch_size=batch_size)


def _build_run_entries(
    run: dict[str, Any],
    tail_entries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Build the per-run entry list ([generator] + tail with prefixed outputs) and
    return it together with the run-level env overrides (config minus any
    non-scalar keys, which are routed to kwargs instead).
    """
    gen_name = run["generator"]
    output_dir = run["output_dir"]

    run_config = dict(run.get("config", {}) or {})
    similarity_jobs = run_config.get("QA_SIMILARITY_MULTIHOP_JOBS")

    gen_entry: dict[str, Any] = {
        "phase": gen_name,
        "output": os.path.join(output_dir, f"{gen_name}.jsonl"),
    }
    if similarity_jobs is not None:
        gen_entry["kwargs"] = {"similarity_jobs": similarity_jobs}

    entries: list[dict[str, Any]] = [gen_entry]
    for t in tail_entries:
        te = dict(t)
        te["output"] = os.path.join(output_dir, t["output"])
        entries.append(te)

    return entries, run_config


async def _run_multi(
    preprocess_entry: Optional[dict[str, Any]],
    tail_entries: list[dict[str, Any]],
    runs: list[dict[str, Any]],
    *,
    num_rows_cli: Optional[int],
    batch_size: int,
) -> None:
    if preprocess_entry:
        await _run_preprocess(preprocess_entry, batch_size)

    for run_idx, run in enumerate(runs, start=1):
        entries, run_config = _build_run_entries(run, tail_entries)
        os.makedirs(run["output_dir"], exist_ok=True)

        print(f"\n{'#' * 60}")
        print(f"RUN {run_idx}/{len(runs)}: generator='{run['generator']}' → {run['output_dir']}")
        print(f"{'#' * 60}")

        with _phase_env(run_config):
            num_rows = (
                num_rows_cli
                if num_rows_cli is not None
                else int(os.environ.get("NUM_ROWS", str(NUM_ROWS)))
            )
            phases = _build_and_validate(entries)
            await run_pipeline(entries, phases, num_rows=num_rows, batch_size=batch_size)

    print(f"\n{'=' * 60}")
    print("ALL RUNS COMPLETE")
    print(f"{'=' * 60}")


# Dry-run
async def _dry_run_multi(
    preprocess_entry: Optional[dict[str, Any]],
    tail_entries: list[dict[str, Any]],
    runs: list[dict[str, Any]],
    *,
    batch_size: int,
) -> None:
    print("=" * 60)
    print("DRY RUN — no LLM calls; counting generator output rows")
    print("=" * 60)

    if preprocess_entry:
        await _run_preprocess(preprocess_entry, batch_size)

    total = 0
    rows_by_run: list[tuple[str, Optional[int]]] = []

    for run_idx, run in enumerate(runs, start=1):
        entries, run_config = _build_run_entries(run, tail_entries)
        gen_entry = entries[0]
        cls = get_phase(gen_entry["phase"])
        phase = cls()

        print(f"\nRUN {run_idx}/{len(runs)}: {run['generator']} → {run['output_dir']}")

        with _phase_env(run_config):
            num_rows = int(os.environ.get("NUM_ROWS", str(NUM_ROWS)))
            extra_kwargs: dict[str, Any] = gen_entry.get("kwargs", {}) or {}
            est = await phase.estimate(num_rows=num_rows, batch_size=batch_size, **extra_kwargs)

        rows_by_run.append((run["generator"], est))
        if est is not None:
            total += est

    print(f"\n{'=' * 60}")
    print("DRY RUN SUMMARY")
    print(f"{'=' * 60}")
    for gen_name, est in rows_by_run:
        if est is None:
            print(f"  {gen_name:<35} unknown")
        else:
            print(f"  {gen_name:<35} {est:>12,}  QA rows")
    if any(e is not None for _, e in rows_by_run):
        print(f"  {'TOTAL (before tail filtering)':<35} {total:>12,}  QA rows")
    print(f"\n  Tail phases (qa_filter → conv_expand_var → conv_filter → judge → embed_filter)")
    print(f"  typically keep 20–50% of QA rows as final conversations.")
    print(f"{'=' * 60}")


async def _dry_run_legacy(
    entries: list[dict[str, Any]],
    phases: list[Phase],
    *,
    num_rows: int,
    batch_size: int,
) -> None:
    print("=" * 60)
    print("DRY RUN — no LLM calls; counting generator output rows")
    print("=" * 60)

    for entry, phase in zip(entries, phases):
        if phase.role != PhaseRole.GENERATOR:
            continue
        cfg_overrides: dict[str, Any] = entry.get("config", {}) or {}
        extra_kwargs: dict[str, Any] = entry.get("kwargs", {}) or {}
        with _phase_env(cfg_overrides):
            est = await phase.estimate(num_rows=num_rows, batch_size=batch_size, **extra_kwargs)
        print(f"\n{'=' * 60}")
        if est is None:
            print(f"Generator '{phase.name}': no estimate available")
        else:
            exhaustive = num_rows < 0
            cap = "exhaustive" if exhaustive else f"capped at NUM_ROWS={num_rows:,}"
            print(f"Generator '{phase.name}': {est:,} QA rows ({cap})")
        print(f"{'=' * 60}")
        return

    print("No GENERATOR phase found in pipeline.")


# CLI
def main() -> None:
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
    parser.add_argument("--num-rows", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Run preprocess (vectorstore build) but skip all LLM calls; "
            "count and print the number of QA rows each generator would emit."
        ),
    )

    args = parser.parse_args()

    if args.list_phases:
        _import_all_phases()
        print("Registered phases:")
        for name in list_phases():
            print(f"  {name}")
        return

    # Multi-run orchestrator format (preprocess / tail / runs)
    if _is_multi_run_yaml(args.pipeline):
        preprocess_entry, tail_entries, runs, global_env = _load_multihop_yaml(args.pipeline)
        if global_env:
            _apply_global_env(global_env)
        _import_all_phases()

        batch_size = (
            args.batch_size if args.batch_size is not None
            else int(os.environ.get("BATCH_SIZE", str(BATCH_SIZE)))
        )

        print("=" * 60)
        print("SeedDataGen Pipeline (multi-run)")
        print("=" * 60)
        print(f"vLLM URL  : {os.environ.get('VLLM_BASE_URL', VLLM_BASE_URL)}")
        print(f"Pipeline  : {args.pipeline}")
        print(f"Batch size: {batch_size}")
        print(f"Runs      : {len(runs)}")
        if global_env:
            print(f"Global env: {global_env}")

        if args.dump_prompts:
            _dump_prompts_multi(preprocess_entry, tail_entries, runs, args.dump_prompts)
            return

        if args.dry_run:
            asyncio.run(
                _dry_run_multi(
                    preprocess_entry,
                    tail_entries,
                    runs,
                    batch_size=batch_size,
                )
            )
            return

        asyncio.run(
            _run_multi(
                preprocess_entry,
                tail_entries,
                runs,
                num_rows_cli=args.num_rows,
                batch_size=batch_size,
            )
        )
        return

    # Legacy single-pipeline format
    entries, global_env = _load_pipeline_yaml(args.pipeline)
    if global_env:
        _apply_global_env(global_env)

    _import_all_phases()

    # Resolve num_rows and batch_size after global env is applied
    num_rows = args.num_rows if args.num_rows is not None else int(os.environ.get("NUM_ROWS", str(NUM_ROWS)))
    batch_size = args.batch_size if args.batch_size is not None else int(os.environ.get("BATCH_SIZE", str(BATCH_SIZE)))

    print("=" * 60)
    print("SeedDataGen Pipeline")
    print("=" * 60)
    print(f"vLLM URL  : {os.environ.get('VLLM_BASE_URL', VLLM_BASE_URL)}")
    print(f"Pipeline  : {args.pipeline}")
    print(f"Batch size: {batch_size}")
    print(f"Num rows  : {num_rows}")
    if args.start_from:
        print(f"Start from: {args.start_from}")
    if args.only:
        print(f"Only phase: {args.only}")
    if global_env:
        print(f"Global env: {global_env}")

    phases = _build_and_validate(entries)

    if args.dump_prompts:
        _dump_prompts(entries, phases, args.dump_prompts)
        return

    if args.dry_run:
        asyncio.run(
            _dry_run_legacy(
                entries,
                phases,
                num_rows=num_rows,
                batch_size=batch_size,
            )
        )
        return

    asyncio.run(
        run_pipeline(
            entries,
            phases,
            start_from=args.start_from,
            only=args.only,
            input_override=args.input,
            output_override=args.output,
            num_rows=num_rows,
            batch_size=batch_size,
        )
    )


if __name__ == "__main__":
    main()
