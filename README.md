# SeedDataGen

YAML-driven pipeline: HuggingFace corpus → filtered multi-turn instruction JSONL.
Generation and judging call a vLLM server; dedup uses `sentence-transformers`.

## Quick start

Run from the directory that **contains** `SeedDataGen/` (imports are `SeedDataGen.*`):

```bash
pip install -r SeedDataGen/requirements.txt

python -m SeedDataGen.run_pipeline --pipeline SeedDataGen/pipeline_myver.yaml
python -m SeedDataGen.run_pipeline --pipeline SeedDataGen/pipeline_multihop.yaml
python -m SeedDataGen.run_pipeline --list-phases
python -m SeedDataGen.run_pipeline --dump-prompts prompts_preview.txt
python -m SeedDataGen.run_pipeline --pipeline SeedDataGen/pipeline_multihop.yaml --dry-run
```

Set `VLLM_BASE_URL`, `DATASET_DOC_NAME_FIELD`, and other `DATASET_*` keys in the YAML
`env:` block or `SeedDataGen/.env` before running.

## Pipeline YAML files

| File | Description |
|------|-------------|
| `pipeline.yaml` | Legacy single-generator (`qa_gen`) |
| `pipeline_myver.yaml` | Single-generator (`qa_gen_var`) |
| `pipeline_multihop.yaml` | Multi-run example (three generators + shared tail) |
| `pipeline_multihop_myver.yaml` | Multi-run tuned for the CEMIG norms corpus |

## Merge multi-run outputs

```bash
python -m SeedDataGen.merge_runs out/*/final.jsonl -o combined/final.jsonl
```

## Documentation

| Doc | Use when |
|-----|----------|
| [`CODEBASE_MAP.md`](CODEBASE_MAP.md) | Finding which file implements a feature |
| [`CLAUDE.md`](CLAUDE.md) | Agent/dev essentials: architecture, gotchas, how to add phases |
| [`docs/PHASES.md`](docs/PHASES.md) | Per-phase settings, schemas, and behavior |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Data contracts, resume, YAML formats, extension guide |

Each role folder also has a short `CLAUDE.md` (`generator/`, `editor/`, etc.).
