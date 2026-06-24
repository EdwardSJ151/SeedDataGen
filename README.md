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

## Components

A pipeline is a chain of **phases**, each with a **role**. Generators start the chain (read the
dataset, no input file); the rest read the previous phase's JSONL.

**Generators** (`None → row`) — produce rows from dataset chunks:

| Phase | Output | What it does |
|-------|--------|--------------|
| `qa_gen` | `QARow` | Up to 5 QA pairs per chunk |
| `qa_gen_var` | `StyledQARow` | One QA pair per **style** per chunk |
| `qa_local_multihop` | `StyledQARow` | QA over a window of **adjacent** chunks (needs `chroma_preprocess`) |
| `qa_similarity_multihop` | `StyledQARow` | QA over **vector-similar** chunk groups (needs `chroma_preprocess`) |
| `rewrite_gen` | `ConversationRow` | Single-turn rewrite/summary per chunk × style (`summary`/`simplify`/`focus`) |

**Editors** (`row → row`) — transform rows:

| Phase | In → Out | What it does |
|-------|----------|--------------|
| `conv_expand_var` | `StyledQARow → ConversationRow` | Expand a QA pair into a multi-turn conversation |
| `answer_rewrite` *(optional)* | `QARow → QARow` | Rewrite answers to be richer, source-only |

**Filters** (`row → row`, pure Python, no LLM) — drop/renumber rows:

| Phase | In/Out | What it does |
|-------|--------|--------------|
| `qa_filter` | `QARow → StyledQARow` | Drop short answers / near-duplicate questions |
| `conv_filter` | `ConversationRow` | Truncate at refusals; drop too-short / duplicate conversations |

**Judge** (`judge`, JUDGE) — scores conversations on 5 dimensions, keeps those above `MIN_AVG_SCORE`
(`ConversationRow → JudgedConversationRow`).

**Dedup** (`embed_filter`, DEDUP) — embedding-based dedup within each `sample_id` group
(`JudgedConversationRow`).

**Preprocess** (`chroma_preprocess`, run once) — builds the Chroma vectorstore the multihop generators
need.

Per-phase settings, env prefixes, and schemas: [`docs/PHASES.md`](docs/PHASES.md).

## Adding a phase to a pipeline YAML

A single-pipeline YAML is a top-level `env:` block plus an ordered `pipeline:` list. Each entry names a
`phase`, the `output` JSONL it writes, and an optional `config:` block of env-var overrides (verbatim
keys, e.g. the phase's `{PREFIX}_*` settings):

```yaml
env:
  VLLM_BASE_URL: "http://localhost:8000/v1"
  DATASET_ID: "org/my-corpus"
  DATASET_DOC_NAME_FIELD: "document_name"

pipeline:
  - phase: qa_gen_var
    output: qa.jsonl
    config:
      QA_GEN_VAR_QUESTION_STYLES: "easy,medium,hard"
  - phase: qa_filter
    output: qa_filtered.jsonl
  - phase: conv_expand_var
    output: conv.jsonl
  - phase: conv_filter
    output: conv_filtered.jsonl
  - phase: judge
    output: judged.jsonl
  - phase: embed_filter
    output: final.jsonl
```

**Ordering rules** (validated at startup): consecutive phases must pass the role transition
(GENERATOR → EDITOR/FILTER/JUDGE/DEDUP; JUDGE → FILTER/DEDUP; DEDUP → EDITOR/FILTER) and the schema
check (`prev.output_schema` must be a subclass of `curr.input_schema`). A phase emitting a finished
`ConversationRow` (e.g. `rewrite_gen`) skips `qa_filter`/`conv_expand_var` and feeds `conv_filter`
directly.

**Multi-run YAML** (`runs:` key) runs several generators, each followed by a shared `tail:` list, with an
optional one-time `preprocess:`. See `pipeline_multihop.yaml`.

## Pipeline YAML files

| File | Description |
|------|-------------|
| `pipeline.yaml` | Legacy single-generator (`qa_gen`) |
| `pipeline_myver.yaml` | Single-generator (`qa_gen_var`) |
| `pipeline_multihop.yaml` | Multi-run example (three generators + shared tail) |
| `pipeline_multihop_myver.yaml` | Multi-run tuned for the CEMIG norms corpus |
| `pipeline_rewrite_myver.yaml` | Single-turn rewrite/summary (`rewrite_gen`), separate from Q&A |

## Merge multi-run outputs

```bash
python -m SeedDataGen.merge_runs out/*/final.jsonl -o combined/final.jsonl
```
