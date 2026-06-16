# SeedDataGen

Turns a HuggingFace text dataset into filtered multi-turn instruction-tuning
conversations in JSONL.  Uses a vLLM server for generation/judging and
`sentence-transformers` for embedding-based deduplication.

## Quick start

```bash
# Install dependencies (run from the parent directory that contains SeedDataGen/)
pip install -r SeedDataGen/requirements.txt

# Full pipeline using the example YAML
python -m SeedDataGen.run_pipeline --pipeline SeedDataGen/pipeline_myver.yaml

# Multi-generator run (qa_gen_var + qa_local_multihop + qa_similarity_multihop)
python -m SeedDataGen.run_pipeline --pipeline SeedDataGen/pipeline_multihop.yaml

# List all registered phases
python -m SeedDataGen.run_pipeline --list-phases

# Preview all prompts before running (no LLM calls)
python -m SeedDataGen.run_pipeline --dump-prompts prompts_preview.txt
```

Set `VLLM_BASE_URL` (and optionally `VLLM_API_KEY`) in the YAML `env:` block or
in `SeedDataGen/.env`.

## Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** — full architecture reference, data
  contracts, how to add phases/generators, YAML format, and design rationale.
  This is the right starting point for any extension work.
- **[PHASES.md](PHASES.md)** — concise per-phase reference card (role, config
  prefix, key settings, input/output schema).

## Pipeline files

| File | Description |
|---|---|
| `pipeline.yaml` | Legacy single-generator example (`qa_gen`) |
| `pipeline_myver.yaml` | Production single-generator example (`qa_gen_var`) |
| `pipeline_multihop.yaml` | Multi-generator example (all three generators) |

## Merging multiple runs

```bash
python -m SeedDataGen.merge_runs out/*/final.jsonl -o combined/final.jsonl
```
