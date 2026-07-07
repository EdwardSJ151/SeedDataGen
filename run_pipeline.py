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

_PHASE_PACKAGE = "SeedDataGen"
_PHASE_DIR = Path(__file__).resolve().parent


def _resolve_pipeline_path(pipeline_arg: str) -> Path:
    """Resolve --pipeline the same way as the main CLI (cwd, then package dir)."""
    p = Path(pipeline_arg)
    if p.is_file():
        return p.resolve()
    for base in (Path.cwd(), _PHASE_DIR):
        candidate = (base / pipeline_arg).resolve()
        if candidate.is_file():
            return candidate
    return (_PHASE_DIR / pipeline_arg).resolve()


def _load_yaml_global_env(path: Path) -> dict[str, str]:
    """Return the top-level ``env:`` mapping from a pipeline YAML file."""
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return {str(k): str(v) for k, v in (data.get("env") or {}).items()}


def _apply_global_env(global_env: dict[str, str]) -> None:
    """Apply top-level env: block to os.environ."""
    for key, val in global_env.items():
        os.environ[key] = str(val)


def _bootstrap_pipeline_env(argv: list[str] | None = None) -> None:
    """
    Apply the YAML ``env:`` block before SeedDataGen.config is imported.

    Pipeline YAML is applied again in ``main()``; this early pass mirrors the
    8bbba82 / 4b5c5fd pattern — settings belong in os.environ at runtime, not import time.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--pipeline",
        default=str(_PHASE_DIR / "pipeline.yaml"),
    )
    args, _ = parser.parse_known_args(argv if argv is not None else sys.argv[1:])
    path = _resolve_pipeline_path(args.pipeline)
    if not path.is_file():
        return
    _apply_global_env(_load_yaml_global_env(path))


def _apply_pipeline_env_from_arg(pipeline_arg: str) -> None:
    """Apply ``env:`` from the resolved pipeline file (authoritative in main())."""
    path = _resolve_pipeline_path(pipeline_arg)
    if not path.is_file():
        raise FileNotFoundError(
            f"Pipeline file not found: {pipeline_arg!r} "
            f"(resolved to {path}). Set --pipeline or run from the repo directory."
        )
    _apply_global_env(_load_yaml_global_env(path))


_bootstrap_pipeline_env()

from SeedDataGen.base_phase import Phase, PhaseRole
from SeedDataGen.config import BATCH_SIZE, NUM_ROWS, VLLM_BASE_URL, validate_pipeline_env
from SeedDataGen.registry import get_phase, list_phases
from SeedDataGen.utils import (
    count_jsonl_lines,
    get_failed_pairs,
    get_last_processed_id,
    get_max_int_field,
    get_sample_group_key,
    stamp_statuses,
    write_jsonl_batch,
)

# Auto-discover and import all phase_*.py modules so their @register
# decorators fire before we look anything up in the registry.

def _import_all_phases() -> None:
    # Recursive discovery so phases living in role subpackages (generator/,
    # editor/, filter/, judge/, dedup/, preprocess/) are also registered.
    for path in sorted(_PHASE_DIR.glob("**/phase_*.py")):
        rel = path.relative_to(_PHASE_DIR).with_suffix("")
        module_name = f"{_PHASE_PACKAGE}." + ".".join(rel.parts)
        if module_name not in sys.modules:
            importlib.import_module(module_name)


# YAML loading
def _load_pipeline_yaml(path: str) -> tuple[list[dict[str, Any]], dict[str, str], int]:
    """
    Returns (entries, global_env, retry_max_attempts) where:
      entries             — the 'pipeline:' list
      global_env          — the optional top-level 'env:' dict (empty if absent)
      retry_max_attempts  — 0 = no retry (default), -1 = infinite, N = max retries
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
    retry_max_attempts = int(data.get("retry_max_attempts", 0))
    return entries, global_env, retry_max_attempts


def _is_multi_run_yaml(path: str) -> bool:
    """True if *path* uses the multi-run orchestrator format (a 'runs:' key)."""
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return isinstance(data, dict) and isinstance(data.get("runs"), list)


def _load_multihop_yaml(
    path: str,
) -> tuple[Optional[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, str], int]:
    """
    Parse the multi-run YAML format.

    Returns (preprocess_entry, tail_entries, runs, global_env, retry_max_attempts):
      preprocess_entry   — the optional 'preprocess:' mapping (or None)
      tail_entries       — the shared 'tail:' list (phases after each generator)
      runs               — the 'runs:' list (one full pipeline per entry)
      global_env         — the optional top-level 'env:' dict
      retry_max_attempts — 0 = no retry (default), -1 = infinite, N = max retries
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
    retry_max_attempts = int(data.get("retry_max_attempts", 0))
    return preprocess_entry, tail, runs, global_env, retry_max_attempts


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
def _finalize_phase_output(
    phase: Phase, out_file: str, empties: list[tuple[str, str]]
) -> None:
    """
    Ensure *out_file* exists after a phase runs.  A phase that keeps 0 rows writes
    no file, which would crash the next phase on open().  Create the empty file,
    warn, and record it for the end-of-run summary.  PREPROCESS phases (which
    produce a vectorstore, not a JSONL) are skipped.
    """
    if phase.role == PhaseRole.PREPROCESS or not out_file:
        return
    if count_jsonl_lines(out_file) > 0:
        return
    parent = os.path.dirname(out_file)
    if parent:
        os.makedirs(parent, exist_ok=True)
    open(out_file, "a").close()
    print(f"[warning] phase '{phase.name}' produced 0 rows — wrote empty file {out_file}")
    empties.append((phase.name, out_file))


def _print_empty_summary(
    empties: list[tuple[str, str]], header: str = "EMPTY OUTPUTS (0 rows)"
) -> None:
    if not empties:
        return
    print(f"\n{'=' * 60}")
    print(header)
    print(f"{'=' * 60}")
    for name, path in empties:
        print(f"  - {name} → {path}")


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
) -> list[tuple[str, str]]:
    empties: list[tuple[str, str]] = []

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
        _finalize_phase_output(phase, out_file, empties)
        _print_empty_summary(empties)
        return empties

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
        _finalize_phase_output(phase, out_file, empties)

    print(f"\n{'=' * 60}")
    print("PIPELINE COMPLETE")
    print(f"{'=' * 60}")
    final_output = entries[-1]["output"]
    print(f"Final output: {final_output}")
    _print_empty_summary(empties)
    return empties


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
    retry_max_attempts: int = 0,
) -> None:
    if preprocess_entry:
        await _run_preprocess(preprocess_entry, batch_size)

    all_empties: list[tuple[str, str]] = []
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
            run_empties = await run_pipeline(
                entries, phases, num_rows=num_rows, batch_size=batch_size
            )
            all_empties.extend(
                (f"run {run_idx}/{run['generator']}: {name}", path)
                for name, path in run_empties
            )
            if retry_max_attempts != 0:
                await _retry_pipeline(entries, phases, batch_size, retry_max_attempts)

    print(f"\n{'=' * 60}")
    print("ALL RUNS COMPLETE")
    print(f"{'=' * 60}")
    _print_empty_summary(all_empties, header="EMPTY OUTPUTS ACROSS ALL RUNS (0 rows)")


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
    print("\n  Tail phases (qa_filter → conv_expand_var → conv_filter → judge → embed_filter)")
    print("  typically keep 20–50% of QA rows as final conversations.")
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


def _require_dataset_env() -> None:
    """Fail fast when the pipeline YAML (or shell) omitted required dataset keys."""
    validate_pipeline_env()


# ---------------------------------------------------------------------------
# Retry orchestrator
# ---------------------------------------------------------------------------

def _build_gen_pairs(gen_output: str) -> dict[tuple[str, str], list[int]]:
    """
    Return {(group_key, question_style): [row_ids]} for rows that are NOT
    already stamped as "failed" (i.e. rows from the latest retry attempt).
    """
    import json
    pairs: dict[tuple[str, str], list[int]] = {}
    if not os.path.exists(gen_output):
        return pairs
    with open(gen_output, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("status") == "failed":
                continue
            gk = get_sample_group_key(obj.get("sample_id", ""))
            qs = obj.get("question_style") or ""
            row_id = obj.get("id")
            if row_id is None:
                continue
            key = (gk, qs)
            pairs.setdefault(key, []).append(row_id)
    return pairs


def _build_passed_pairs(embed_output: str) -> set[tuple[str, str]]:
    """Return {(group_key, question_style)} for rows in the embed_filter output."""
    import json
    passed: set[tuple[str, str]] = set()
    if not os.path.exists(embed_output):
        return passed
    with open(embed_output, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            gk = get_sample_group_key(obj.get("sample_id", ""))
            qs = obj.get("question_style") or ""
            passed.add((gk, qs))
    return passed


async def _retry_embed_filter(
    embed_output: str,
    pre_embed_output: str,
    failed_pairs: set[tuple[str, str]],
    embed_phase: Phase,
    batch_size: int,
) -> None:
    """
    Run embed_filter on a combined input of existing survivors for affected
    groups + new retry rows from the pre-embed-filter phase output.

    New survivors (rows not already in embed_output) are appended to embed_output.
    """
    import json
    import tempfile

    affected_groups = {gk for (gk, _) in failed_pairs}

    # Load existing survivors for affected groups.
    existing: list[dict] = []
    if os.path.exists(embed_output):
        with open(embed_output, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if get_sample_group_key(obj.get("sample_id", "")) in affected_groups:
                    existing.append(obj)

    # Find new retry rows from the pre-embed input (judge output).
    last_embed_input_id = get_max_int_field(embed_output, "input_id")
    new_rows: list[dict] = []
    if os.path.exists(pre_embed_output):
        with open(pre_embed_output, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("id", -1) <= last_embed_input_id:
                    continue
                if get_sample_group_key(obj.get("sample_id", "")) not in affected_groups:
                    continue
                new_rows.append(obj)

    if not new_rows:
        print("[retry] no new rows for affected groups; skipping embed_filter step")
        return

    # Build combined temp input: existing survivors (context) first, new rows after.
    # Renumber ids sequentially so embed_filter's id-based skip logic works cleanly.
    tmp_dir = os.path.dirname(embed_output)
    tmp_input = os.path.join(tmp_dir, "_retry_embed_input.jsonl")
    tmp_output = os.path.join(tmp_dir, "_retry_embed_output.jsonl")

    combined: list[dict] = []
    for i, row in enumerate(existing):
        r = dict(row)
        r["_context"] = True
        r["id"] = i
        combined.append(r)
    offset = len(existing)
    for j, row in enumerate(new_rows):
        r = dict(row)
        r["_original_id"] = r["id"]
        r["id"] = offset + j
        combined.append(r)

    write_jsonl_batch(tmp_input, combined)

    # Run embed_filter fresh on the temp input (no prior output → resume_from=0).
    try:
        await embed_phase.run(
            input_file=tmp_input,
            output_file=tmp_output,
            batch_size=batch_size,
        )
    except Exception as e:
        print(f"[retry] embed_filter failed during retry: {e}")
        for f in (tmp_input, tmp_output):
            if os.path.exists(f):
                os.remove(f)
        return

    # Extract new survivors: rows without _context.
    new_survivors: list[dict] = []
    if os.path.exists(tmp_output):
        with open(tmp_output, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("_context"):
                    continue
                new_survivors.append(obj)

    if new_survivors:
        next_id = get_last_processed_id(embed_output) + 1
        for row in new_survivors:
            row["input_id"] = row.pop("_original_id", row.get("input_id"))
            row.pop("_context", None)
            row["id"] = next_id
            next_id += 1
        write_jsonl_batch(embed_output, new_survivors)
        print(f"[retry] embed_filter appended {len(new_survivors)} new survivor(s) → {embed_output}")
    else:
        print("[retry] embed_filter: no new survivors from retry rows")

    for f in (tmp_input, tmp_output):
        if os.path.exists(f):
            os.remove(f)


async def _retry_pipeline(
    entries: list[dict[str, Any]],
    phases: list[Phase],
    batch_size: int,
    max_attempts: int,
) -> None:
    """
    After the main pipeline completes, identify failed (group_key, style) pairs
    and iterate: re-run generator → tail phases resume naturally → combined
    embed_filter step.  Repeats up to max_attempts times (-1 = infinite).
    """
    if phases[-1].role != PhaseRole.DEDUP:
        print(
            "[retry] last phase is not a DEDUP phase — retry requires embed_filter as the "
            "final phase; skipping retry."
        )
        return
    if len(entries) < 2:
        print("[retry] pipeline has fewer than 2 phases; skipping retry.")
        return

    gen_entry = entries[0]
    embed_entry = entries[-1]
    pre_embed_entry = entries[-2]
    gen_phase = phases[0]
    embed_phase = phases[-1]
    tail_phases = list(zip(entries[1:-1], phases[1:-1]))

    gen_output = gen_entry["output"]
    embed_output = embed_entry["output"]
    pre_embed_output = pre_embed_entry["output"]

    attempt = 0
    while True:
        attempt += 1
        print(f"\n{'=' * 60}")
        print(f"RETRY attempt {attempt}" + (f"/{max_attempts}" if max_attempts > 0 else ""))
        print(f"{'=' * 60}")

        # 1. Identify failures.
        all_gen_pairs = _build_gen_pairs(gen_output)
        passed_pairs = _build_passed_pairs(embed_output)
        failed_pairs = {k for k in all_gen_pairs if k not in passed_pairs}

        if not failed_pairs:
            print("[retry] all pairs passed — converged.")
            break

        print(f"[retry] {len(failed_pairs)} pair(s) still failed; {len(passed_pairs)} passed.")

        # 2. Stamp statuses in the generator output.
        passed_ids: set[int] = {
            max(all_gen_pairs[k]) for k in all_gen_pairs if k in passed_pairs
        }
        failed_ids: set[int] = {
            max(all_gen_pairs[k]) for k in all_gen_pairs if k in failed_pairs
        }
        stamp_statuses(gen_output, passed_ids, failed_ids)

        # 3. Re-run generator with retry_pairs.
        gen_cfg = gen_entry.get("config", {}) or {}
        gen_kwargs: dict[str, Any] = gen_entry.get("kwargs", {}) or {}
        with _phase_env(gen_cfg):
            await gen_phase.run(
                input_file="",
                output_file=gen_output,
                retry_pairs=failed_pairs,
                batch_size=batch_size,
                num_rows=-1,
                **gen_kwargs,
            )

        # 4. Re-run tail phases (all except embed_filter); resume handles new rows.
        for tail_entry, tail_phase in tail_phases:
            tail_idx = entries.index(tail_entry)
            in_file = entries[tail_idx - 1]["output"]
            out_file = tail_entry["output"]
            tail_cfg = tail_entry.get("config", {}) or {}
            tail_kwargs: dict[str, Any] = tail_entry.get("kwargs", {}) or {}
            print(f"\n[retry] re-running {tail_phase.name} ...")
            with _phase_env(tail_cfg):
                await tail_phase.run(
                    input_file=in_file,
                    output_file=out_file,
                    batch_size=batch_size,
                    num_rows=-1,
                    **tail_kwargs,
                )
            _finalize_phase_output(tail_phase, out_file, [])

        # 5. embed_filter combined-input step.
        print(f"\n[retry] running embed_filter for {len(failed_pairs)} affected group(s) ...")
        embed_cfg = embed_entry.get("config", {}) or {}
        with _phase_env(embed_cfg):
            await _retry_embed_filter(
                embed_output, pre_embed_output, failed_pairs, embed_phase, batch_size
            )

        if max_attempts > 0 and attempt >= max_attempts:
            # Re-check failures after this last attempt.
            all_gen_pairs = _build_gen_pairs(gen_output)
            passed_pairs = _build_passed_pairs(embed_output)
            still_failed = {k for k in all_gen_pairs if k not in passed_pairs}
            if still_failed:
                print(
                    f"[retry] max_attempts={max_attempts} reached; "
                    f"{len(still_failed)} pair(s) still failed."
                )
            else:
                print("[retry] all pairs passed after final attempt.")
            break


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
        _apply_pipeline_env_from_arg(args.pipeline)
        preprocess_entry, tail_entries, runs, global_env, retry_max_attempts = _load_multihop_yaml(args.pipeline)
        _require_dataset_env()
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
        if retry_max_attempts != 0:
            print(f"Retry     : {retry_max_attempts if retry_max_attempts > 0 else 'until convergence'} (per run)")
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
                retry_max_attempts=retry_max_attempts,
            )
        )
        return

    # Legacy single-pipeline format
    _apply_pipeline_env_from_arg(args.pipeline)
    entries, global_env, retry_max_attempts = _load_pipeline_yaml(args.pipeline)
    _require_dataset_env()
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

    async def _run_with_retry() -> None:
        await run_pipeline(
            entries,
            phases,
            start_from=args.start_from,
            only=args.only,
            input_override=args.input,
            output_override=args.output,
            num_rows=num_rows,
            batch_size=batch_size,
        )
        if retry_max_attempts != 0 and not args.start_from and not args.only:
            await _retry_pipeline(entries, phases, batch_size, retry_max_attempts)

    asyncio.run(_run_with_retry())


if __name__ == "__main__":
    main()
