# SeedDataGen — Architecture & Extension Guide

This document describes the full architecture of SeedDataGen — how data flows,
how every component is structured, what contracts phases must honour, and a
step-by-step guide for extending the system with new phases, new generators, or
new pipeline patterns.  It is the primary reference for any developer (or LLM)
that needs to understand or modify this codebase.

---

## 1. What SeedDataGen does

SeedDataGen turns a HuggingFace text dataset into filtered, multi-turn
instruction-tuning conversations.  The pipeline is:

```
HuggingFace dataset
        │
        ▼
  [chroma_preprocess]  ← one-time vectorstore build (multihop only)
        │
        ├──► GENERATOR (qa_gen_var / qa_local_multihop / qa_similarity_multihop)
        │         │
        │         ▼
        │    qa_filter          ← heuristic QA dedup
        │         │
        │         ▼
        │    conv_expand_var    ← seed QA → multi-turn conversation
        │         │
        │         ▼
        │    conv_filter        ← heuristic conversation dedup
        │         │
        │         ▼
        │    judge              ← LLM scoring (keeps avg > threshold)
        │         │
        │         ▼
        │    embed_filter       ← embedding-based dedup per source group
        │         │
        │         ▼
        │    final.jsonl
        │
        └──► (same tail for every generator run)
                │
                ▼
         merge_runs.py  ← optional: combine multiple runs
```

All intermediate artifacts are JSONL files.  Every row carries an `id` (global
monotonic counter), `origin_id` (the `id` at generation time, never touched),
`sample_id` (list of HF row ids used), `sample_text` (`{hf_row_id: text}` dict),
and provenance metadata (`GEN_TYPE`, `num_chunks`, etc.).

---

## 2. Repository layout

```
SeedDataGen/
│
├── base_phase.py          PhaseRole enum, Phase ABC, COMPATIBLE_TRANSITIONS
├── registry.py            @register decorator, get_phase(), list_phases()
├── schemas.py             Pydantic row models (BaseRow → QARow → ConversationRow → …)
├── config.py              Global env-backed config (dataset, vLLM, execution)
├── utils.py               JSONL I/O, resume helpers, text formatting, score parsing
├── run_pipeline.py        CLI runner — loads YAML, discovers phases, orchestrates runs
├── merge_runs.py          CLI util — concatenate + renumber multiple run outputs
├── prompts.py             Re-export shim (kept for backwards-compat; do not add prompts here)
│
├── preprocess/
│   ├── prompts.py         (empty — preprocess phases make no LLM calls)
│   ├── chunk_index.py     Chroma client helpers (get_collection, collection_exists, …)
│   ├── chunk_retrieval.py local_window(), similarity_groups_iter(), doc_chunk_map()
│   └── phase_chroma_preprocess.py
│
├── generator/
│   ├── prompts.py         QA generation prompts + style instructions dict
│   ├── phase_qa_gen.py
│   ├── phase_qa_gen_var.py
│   ├── phase_qa_local_multihop.py
│   └── phase_qa_similarity_multihop.py
│
├── editor/
│   ├── prompts.py         Conversation expansion prompts + GEN_TYPE routing dicts
│   ├── phase_answer_rewrite.py
│   ├── phase_conv_expand.py    (deprecated)
│   └── phase_conv_expand_var.py
│
├── filter/
│   ├── phase_qa_filter.py
│   └── phase_conv_filter.py
│
├── judge/
│   ├── prompts.py         JUDGE_PROMPT
│   └── phase_judge.py
│
└── dedup/
    └── phase_embed_filter.py
```

Every role directory has an `__init__.py`.  Phases are discovered recursively
(`**/phase_*.py`) at runner startup — no manual import list to maintain.

---

## 3. Core abstractions

### 3.1 PhaseRole

```python
class PhaseRole(str, Enum):
    PREPROCESS = "preprocess"
    GENERATOR  = "generator"
    EDITOR     = "editor"
    FILTER     = "filter"
    JUDGE      = "judge"
    DEDUP      = "dedup"
```

`PREPROCESS` is special: it runs once before all generator runs (in multi-run
YAML) and is never part of the generator → tail chain.  It is not listed in
`COMPATIBLE_TRANSITIONS`.

### 3.2 Phase ABC

Every phase is a subclass of `Phase` (from `base_phase.py`).  It must declare:

```python
class MyPhase(Phase):
    name          = "my_phase"           # unique key used in YAML and registry
    role          = PhaseRole.FILTER     # governs allowed transitions
    input_schema  = QARow                # Pydantic model for input rows (None for GENERATORs)
    output_schema = QARow                # Pydantic model for output rows

    async def run(self, input_file: str, output_file: str, **kwargs) -> None:
        ...

    def describe_prompts(self) -> List[Tuple[str, str]]:
        # Optional. Returns [(label, rendered_prompt_string), ...]
        # Used by --dump-prompts, no LLM calls made here.
        return []
```

`kwargs` the runner always passes: `num_rows` (int, -1 = exhaustive), `batch_size` (int).
Phases may also receive additional kwargs from the YAML (see §8.3).

### 3.3 Pydantic row schemas (`schemas.py`)

```
BaseRow
├── QARow            ← generators, qa_filter, answer_rewrite
│   └── StyledQARow  ← qa_gen_var, qa_local_multihop, qa_similarity_multihop
└── ConversationRow  ← conv_expand_var, conv_filter
    └── JudgedConversationRow  ← judge, embed_filter
```

All models have `model_config = ConfigDict(extra="ignore")`, so downstream
phases never break when upstream phases add new fields.

`BaseRow` fields:

| Field         | Type                            | Notes |
|---------------|---------------------------------|-------|
| `id`          | `int`                           | Monotonic, re-assigned by each phase |
| `origin_id`   | `int`                           | Fixed at generation; never touched again |
| `sample_id`   | `Union[int, List[int]]`         | HF row id(s) of the source chunk(s) |
| `sample_text` | `Union[str, Dict[str, str]]`    | `{hf_row_id: text}` for multihop; plain `str` for legacy |

`QARow` adds: `question`, `answer`, `input_id` (optional), plus provenance
fields: `GEN_TYPE`, `num_chunks`, `doc_constraint`, and similarity metadata
(`similarity_job_index`, `similarity_mode`, `similarity_threshold`,
`similarity_min`, `similarity_max`, `min_matching_words`,
`chunk_group_similarity`) — all `Optional`, default `None`.

`StyledQARow` adds `question_style: str`.

### 3.4 Registry (`registry.py`)

```python
from SeedDataGen.registry import register
from SeedDataGen.base_phase import Phase

@register
class MyPhase(Phase):
    name = "my_phase"
    ...
```

`@register` inserts the class into a module-level dict keyed by `name`.
`get_phase("my_phase")` returns the class.  Duplicate names raise immediately.
The runner imports all `phase_*.py` files recursively before looking anything up.

### 3.5 Phase configuration (`pydantic-settings`)

Every phase owns its runtime config via a `BaseSettings` subclass:

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class MyPhaseConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MY_PHASE_",
        env_file=".env",
        extra="ignore",
    )

    temperature: float = 0.7
    batch_size: int = 32
```

Values are read from environment variables (`MY_PHASE_TEMPERATURE`, etc.) at
the moment the config object is instantiated — which means YAML `config:` blocks
take effect correctly as long as they set env vars *before* `run()` is called
(the runner handles this via `_phase_env()`).

Config keys in YAML must match the env var names exactly (no case folding).

---

## 4. Data contracts between phases

### 4.1 Row shape emitted by generators

All generators now emit rows in this shape:

```json
{
  "id": 0,
  "origin_id": 0,
  "sample_id": ["42"],
  "sample_text": {"42": "chunk text here"},
  "question": "What is X?",
  "answer": "X is Y.",
  "question_style": "hard",
  "GEN_TYPE": "qa_gen_var",
  "num_chunks": 1,
  "doc_constraint": null
}
```

Multihop generators emit the same shape with more entries in `sample_id` and
`sample_text`, and populate the similarity fields.

### 4.2 Grouping key

Phases that must group rows by source document (qa_filter, conv_expand_var
diversity mode, embed_filter) must **not** use `sample_id` directly as a dict
key because it is a list.  Use the helper:

```python
from SeedDataGen.utils import get_sample_group_key
key = get_sample_group_key(row["sample_id"])
# → "42" for single-chunk, or "42|87|91" for multihop (sorted, pipe-delimited)
```

### 4.3 Rendering multi-chunk context for prompts

When passing `sample_text` to a prompt, always render it through:

```python
from SeedDataGen.utils import format_sample_text
context = format_sample_text(row["sample_text"])
```

This handles both the legacy plain-string shape and the multihop dict shape
(`{hf_row_id: text}` → `[Chunk 42]\ntext\n\n[Chunk 87]\ntext\n\n...`).

### 4.4 Field passthrough

Editors and filters must preserve all upstream fields.  Use dict spread:

```python
rows.append({
    **item,                 # everything from upstream
    "id": next_id,          # override
    "input_id": item["id"], # track provenance
    "messages": msgs,       # add new field
})
```

Never hand-pick only the fields you know about — new upstream fields will
otherwise be silently dropped and downstream phases will lose provenance.

---

## 5. Resume semantics

Every phase that writes to a JSONL output file must support resume (idempotent
re-entry after a crash).

**Standard pattern for non-generator phases:**

```python
last_out_id  = get_last_processed_id(output_file)   # highest `id` already written
next_id      = last_out_id + 1 if last_out_id >= 0 else 0
last_input_id = get_max_int_field(output_file, "input_id")
resume_from  = last_input_id + 1 if last_input_id >= 0 else 0
# Then pass resume_from to iter_jsonl_batches(…, start_from_id=resume_from)
```

**For generators:**

```python
skip_ids = get_processed_sample_ids(output_file)
# Returns the set of all hf_row_id strings already in the output.
# Skip records whose hf_row_id is in skip_ids while streaming the dataset.
```

**For multihop generators:**

- `qa_local_multihop`: tracks `(get_sample_group_key(sample_id), question_style)` pairs.
- `qa_similarity_multihop`: rebuilds the pairwise-exclusion set from all `sample_id`
  lists in the output file so no chunk pair is reused.

---

## 6. Prompts

Each role directory has a `prompts.py` that owns all LLM prompt templates for
that role.

```
generator/prompts.py  → QA_GENERATION_PROMPT, QA_GEN_VAR_SYSTEM_PROMPT,
                         QA_GEN_VAR_STYLE_INSTRUCTIONS (dict),
                         QA_LOCAL_MULTIHOP_PROMPT,
                         QA_SIMILARITY_MULTIHOP_PROMPT_POSITIVE,
                         QA_SIMILARITY_MULTIHOP_PROMPT_NEGATIVE

editor/prompts.py     → USER_TURN_VAR_PROMPT, USER_TURN_VAR_DIVERSITY_PROMPT,
                         ASSISTANT_TURN_PROMPT, ANSWER_REWRITE_PROMPT,
                         CONV_EXPAND_USER_TURN_BY_GEN_TYPE (dict),
                         CONV_EXPAND_ASSISTANT_TURN_BY_GEN_TYPE (dict)

judge/prompts.py      → JUDGE_PROMPT
preprocess/prompts.py → (empty)
```

The root `prompts.py` re-exports everything for backwards-compatible imports.
**Add new prompts to the role-specific file, not the root shim.**

### 6.1 GEN_TYPE prompt routing in conv_expand_var

`conv_expand_var` selects the user-turn and assistant-turn templates per row:

```python
# editor/prompts.py
CONV_EXPAND_USER_TURN_BY_GEN_TYPE: dict[str, str] = {
    "qa_gen_var":              USER_TURN_VAR_PROMPT,
    "qa_local_multihop":       USER_TURN_VAR_PROMPT,
    "qa_similarity_multihop":  USER_TURN_VAR_PROMPT,
}
```

If a new generator emits a new `GEN_TYPE` and needs a different user-turn
prompt, add an entry here.  If the key is absent the phase falls back to
`USER_TURN_VAR_PROMPT`.

### 6.2 Style instructions

All QA generators that use styles look up per-style Portuguese instructions in:

```python
# generator/prompts.py
QA_GEN_VAR_STYLE_INSTRUCTIONS: dict[str, str] = {
    "easy": "...", "medium": "...", "hard": "...", "extra_hard": "...",
    "general": "...", "specific": "...", "compositional": "...", "comparative": "...",
}
```

To add a new style, add an entry here.  The style name is then usable in
`QA_GEN_VAR_QUESTION_STYLES`.

---

## 7. Role compatibility rules

The runner validates consecutive phase pairs at startup.  The allowed next roles
for each role are:

| From role | Allowed next roles |
|-----------|--------------------|
| GENERATOR | EDITOR, FILTER, JUDGE, DEDUP |
| EDITOR    | EDITOR, FILTER, JUDGE, DEDUP |
| FILTER    | EDITOR, FILTER, JUDGE, DEDUP |
| JUDGE     | FILTER, DEDUP |
| DEDUP     | EDITOR, FILTER |

`PREPROCESS` is not in this table.  It runs before the generator and is never a
consecutive-transition participant.

Add `force: true` to a YAML entry to bypass the role check (with a warning).
The schema subclass check (`output_schema` of phase N must be a subclass of
`input_schema` of phase N+1) can never be bypassed.

---

## 8. How to add a new phase

### 8.1 Choose the right role and directory

| What the phase does | Role | Directory |
|---|---|---|
| Reads dataset, emits first rows | GENERATOR | `generator/` |
| Transforms rows (LLM call, field rewrite) | EDITOR | `editor/` |
| Drops rows based on heuristics | FILTER | `filter/` |
| Scores rows with LLM, drops low scorers | JUDGE | `judge/` |
| Drops near-duplicate rows | DEDUP | `dedup/` |
| One-time index/vectorstore build | PREPROCESS | `preprocess/` |

### 8.2 Minimal template

```python
# SeedDataGen/<role>/phase_my_phase.py

from pydantic_settings import BaseSettings, SettingsConfigDict

from SeedDataGen.base_phase import Phase, PhaseRole
from SeedDataGen.registry import register
from SeedDataGen.schemas import QARow  # or whichever schema fits
from SeedDataGen.utils import (
    count_jsonl_lines, get_last_processed_id, get_max_int_field,
    iter_jsonl_batches, write_jsonl_batch,
)


class MyPhaseConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MY_PHASE_", env_file=".env", extra="ignore")

    batch_size: int = 32
    # ... your settings


@register
class MyPhase(Phase):
    name = "my_phase"
    role = PhaseRole.FILTER           # or EDITOR, JUDGE, DEDUP, GENERATOR
    input_schema  = QARow             # None if role == GENERATOR
    output_schema = QARow

    def describe_prompts(self):
        return []                     # return [(label, rendered_str)] if LLM-based

    async def run(self, input_file: str, output_file: str, **kwargs) -> None:
        cfg = MyPhaseConfig()
        batch_size = kwargs.get("batch_size", cfg.batch_size)

        # --- resume bookkeeping ---
        last_out_id   = get_last_processed_id(output_file)
        next_id       = last_out_id + 1 if last_out_id >= 0 else 0
        last_input_id = get_max_int_field(output_file, "input_id")
        resume_from   = last_input_id + 1 if last_input_id >= 0 else 0

        # --- process ---
        out_buf = []
        for batch in iter_jsonl_batches(
            input_file, batch_size=batch_size, start_from_id=resume_from,
            required_fields=["question", "answer"],
        ):
            for item in batch:
                # do your work; preserve all upstream fields
                out_buf.append({
                    **item,
                    "id": next_id,
                    "input_id": item["id"],
                })
                next_id += 1

            write_jsonl_batch(output_file, out_buf)
            out_buf = []

        print(f"[my_phase] done → {output_file}")
```

No other registration step is required.  The runner discovers the file
automatically when it recurses through `**/phase_*.py`.

### 8.3 Passing non-scalar config (lists / dicts)

Environment variables only carry strings, so complex config (e.g. a list of
jobs) cannot go through `BaseSettings`.  Pass it as a YAML `kwargs:` block
instead:

```yaml
# In a pipeline YAML
- phase: my_phase
  output: my_output.jsonl
  kwargs:
    my_jobs:
      - mode: "above"
        threshold: 0.8
```

The runner sets `entry["kwargs"]` and unpacks it into `phase.run(**extra_kwargs)`.
Read it inside `run()` via `kwargs.get("my_jobs")`.

In the multi-run YAML, only the generator receives kwargs from the runner (via
`QA_SIMILARITY_MULTIHOP_JOBS` → `similarity_jobs`).  Tail-phase kwargs can be
set in the `tail:` entries using the same `kwargs:` key.

### 8.4 Adding a new generator

Generators are the most complex phase type.  Beyond the base template:

1. **No `input_file`** — generators stream from HuggingFace (or Chroma for
   multihop), so `input_file` is ignored.
2. **Emit the full row shape** — always write `sample_id`, `sample_text`,
   `GEN_TYPE`, `num_chunks`, `doc_constraint`, and `question_style` (if styled).
3. **Use `hf_row_id`** — obtain the row id from the HF column named in
   `DATASET_ID_FIELD` (falls back to the streaming index).  Write it as a
   single-element list: `sample_id = [hf_row_id]`, `sample_text = {hf_row_id: text}`.
4. **Resume** — use `get_processed_sample_ids(output_file)` to build `skip_ids`
   and skip already-processed source records.
5. **Support `NUM_ROWS=-1`** — check `int(num_rows) < 0` and run until the
   dataset is exhausted.
6. **Register and `GEN_TYPE`** — decorate with `@register` and set `GEN_TYPE`
   to the phase name on every emitted row.

If the new generator needs Chroma, call `get_collection_from_env()` from
`SeedDataGen.preprocess.chunk_index`.  It reads the `CHROMA_*` env block
(same keys as `chroma_preprocess`).

### 8.5 Adding a new conv_expand variant

If the new generator needs a different conversation expansion prompt, add the
new `GEN_TYPE` key to the routing dicts in `editor/prompts.py`:

```python
CONV_EXPAND_USER_TURN_BY_GEN_TYPE["my_new_gen_type"] = MY_NEW_USER_PROMPT
CONV_EXPAND_ASSISTANT_TURN_BY_GEN_TYPE["my_new_gen_type"] = MY_NEW_ASST_PROMPT
```

Alternatively, create a dedicated editor phase (subclass `ConvExpandVarPhase`
or write a new one from scratch).

---

## 9. The Chroma vectorstore layer

Used by multihop generators.  Built once by `chroma_preprocess`.

### Collection schema

Each Chroma document represents one chunk:

| Chroma field | Content |
|---|---|
| `id` | `str(hf_row_id)` — the HF column value from `CHROMA_METADATA_HF_ROW_ID` |
| `document` | chunk text |
| `embedding` | computed by `CHROMA_EMBED_MODEL` |
| `metadata.hf_row_id` | same as id |
| `metadata.doc_id` | value of `CHROMA_METADATA_DOC_ID` column |
| `metadata.chunk_index` | value of `CHROMA_METADATA_CHUNK_INDEX` column |

### Key retrieval helpers (`preprocess/chunk_retrieval.py`)

```python
doc_chunk_map(collection)
# → {doc_id: [sorted chunk_indices]}

get_doc_chunks(collection, doc_id)
# → [chunk_dict, ...] sorted by chunk_index

local_window(collection, doc_id, start_chunk_index, num_chunks)
# → ordered slice of chunks for the given window

similarity_groups_iter(collection, num_chunks, jobs, doc_constraint,
                       min_docs, used_pairs, max_candidates=200)
# → Iterator[(chunk_group, job_dict)]
# Pairwise exclusion: used_pairs is mutated in-place.
```

Chunk dicts have keys: `id`, `hf_row_id`, `doc_id`, `chunk_index`, `text`,
`similarity`.

### Opening the collection from a phase

```python
from SeedDataGen.preprocess.chunk_index import get_collection_from_env
collection = get_collection_from_env()
```

This reads `CHROMA_VECTORSTORE_NAME`, `CHROMA_PERSIST_DIR`, `CHROMA_EMBED_MODEL`
from the environment — the same keys set in the YAML `preprocess:` config block.

---

## 10. Utility functions reference (`utils.py`)

| Function | Purpose |
|---|---|
| `get_last_processed_id(filepath)` | Highest `id` in a JSONL, or -1 |
| `get_max_int_field(filepath, field)` | Highest value of any int field, or -1 |
| `get_processed_sample_ids(filepath)` | Set of all `hf_row_id` strings already written (flattens lists) |
| `iter_jsonl_batches(path, batch_size, start_from_id, required_fields)` | Batch iterator with resume and field validation |
| `write_jsonl_batch(filepath, batch)` | Append a list of dicts as JSONL lines |
| `write_jsonl_line(filepath, obj)` | Append a single dict |
| `count_jsonl_lines(filepath)` | Line count (for progress bars) |
| `format_sample_text(sample_text)` | Render str or `{id: text}` dict for prompt injection |
| `get_sample_group_key(sample_id)` | Stable dict key for `Union[int, List[int]]` |
| `parse_qa_pairs(text)` | Extract `Pergunta: … Resposta: …` pairs from LLM output |
| `parse_judge_scores(text)` | Extract five 1–5 scores from judge LLM output |
| `format_conversation_history(messages)` | Full user+assistant history string |
| `format_user_history(messages)` | User-only history string |
| `levenshtein(a, b)` | Edit distance (used by heuristic filters) |

---

## 11. Running a pipeline

### 11.1 Prerequisites

```bash
# From the parent of the SeedDataGen directory (e.g. ~/Code/CEMIG/)
pip install -r SeedDataGen/requirements.txt
```

A running vLLM server is required for all phases that call an LLM
(`qa_gen*`, `conv_expand*`, `judge`).  Point `VLLM_BASE_URL` at it.

```bash
# Optional: put secrets in a .env file next to run_pipeline.py
echo 'VLLM_BASE_URL=http://my-server:8026/v1' > SeedDataGen/.env
```

### 11.2 CLI reference

```bash
# Full pipeline (legacy pipeline: format)
python -m SeedDataGen.run_pipeline

# Custom YAML
python -m SeedDataGen.run_pipeline --pipeline my_pipeline.yaml

# Limit rows and batch size
python -m SeedDataGen.run_pipeline --num-rows 10000 --batch-size 64

# Resume from a specific phase
python -m SeedDataGen.run_pipeline --start-from conv_expand_var

# Run only one phase (with explicit input file)
python -m SeedDataGen.run_pipeline --only judge --input conv_filtered.jsonl

# List all registered phases
python -m SeedDataGen.run_pipeline --list-phases

# Render all prompts (no LLM calls) — useful for review before a run
python -m SeedDataGen.run_pipeline --dump-prompts prompts_preview.txt
```

### 11.3 Legacy pipeline YAML (`pipeline:` key)

Used by `pipeline.yaml` and `pipeline_myver.yaml`.  Suitable for a single
linear generator → tail pipeline.

```yaml
env:
  VLLM_BASE_URL: "http://my-server:8026/v1"
  DATASET_ID: "org/my-dataset"
  DATASET_SPLIT: "train"
  NUM_ROWS: "10000"

pipeline:
  - phase: qa_gen_var
    output: qa_gen_var.jsonl
    config:
      QA_GEN_VAR_QUESTION_STYLES: "easy,medium,hard,extra_hard"

  - phase: qa_filter
    output: qa_filtered.jsonl

  - phase: conv_expand_var
    output: conv_expanded.jsonl
    config:
      CONV_EXPAND_VAR_QUESTION_STYLES: "easy,medium,hard,extra_hard"
      CONV_EXPAND_VAR_N_USER_TURNS_MIN: 3
      CONV_EXPAND_VAR_N_USER_TURNS_MAX: 3
      CONV_EXPAND_VAR_NAIVE_GEN: false

  - phase: conv_filter
    output: conv_filtered.jsonl

  - phase: judge
    output: judged.jsonl

  - phase: embed_filter
    output: final.jsonl
```

Rules:
- Every entry needs both `phase` (registered name) and `output` (path).
- `config:` keys are applied as env vars verbatim before the phase runs and
  restored afterwards.  Keys are **not** uppercased — use exact env var names.
- `force: true` skips the role-transition check (with a warning).
- `kwargs:` passes non-scalar values directly to `phase.run(**kwargs)`.

### 11.4 Multi-run YAML (`runs:` key)

Used by `pipeline_multihop.yaml`.  Supports running multiple generators against
a shared tail, with an optional one-time preprocess step.

```yaml
env:
  VLLM_BASE_URL: "http://my-server:8026/v1"
  DATASET_ID: "org/my-dataset"
  DATASET_SPLIT: "train"
  DATASET_ID_FIELD: "id"           # HF column used as hf_row_id

# Optional: one-time vectorstore build (required for multihop generators)
preprocess:
  phase: chroma_preprocess
  config:
    CHROMA_VECTORSTORE_NAME: "my-collection"
    CHROMA_PERSIST_DIR: ".cache/chroma"
    CHROMA_METADATA_HF_ROW_ID: "id"
    CHROMA_METADATA_DOC_ID: "document_id"
    CHROMA_METADATA_CHUNK_INDEX: "chunk_index"
    CHROMA_METADATA_TEXT: "text"
    CHROMA_EMBED_MODEL: "sentence-transformers/all-MiniLM-L6-v2"
    CHROMA_FORCE_REBUILD: false

# Shared phases that run after every generator
tail:
  - phase: qa_filter
    output: qa_filtered.jsonl
  - phase: conv_expand_var
    output: conv_expanded.jsonl
    config:
      CONV_EXPAND_VAR_QUESTION_STYLES: "easy,medium,hard,extra_hard"
      CONV_EXPAND_VAR_NAIVE_GEN: false
  - phase: conv_filter
    output: conv_filtered.jsonl
  - phase: judge
    output: judged.jsonl
  - phase: embed_filter
    output: final.jsonl

# One entry = one full independent pipeline (generator + tail)
runs:
  - generator: qa_gen_var
    output_dir: out/qa_gen_var/
    config:
      NUM_ROWS: "50000"
      QA_GEN_VAR_QUESTION_STYLES: "easy,medium,hard,extra_hard"

  - generator: qa_local_multihop
    output_dir: out/qa_local_multihop/
    config:
      NUM_ROWS: "-1"                         # -1 = exhaust all windows
      CHROMA_VECTORSTORE_NAME: "my-collection"
      QA_LOCAL_MULTIHOP_NUM_CHUNKS: 3
      QA_LOCAL_MULTIHOP_WINDOW_STRIDE: 1
      QA_GEN_VAR_QUESTION_STYLES: "easy,medium,hard,extra_hard"

  - generator: qa_similarity_multihop
    output_dir: out/qa_similarity_multihop/
    config:
      NUM_ROWS: "-1"
      CHROMA_VECTORSTORE_NAME: "my-collection"
      QA_SIMILARITY_MULTIHOP_NUM_CHUNKS: 3
      QA_SIMILARITY_MULTIHOP_DOC_CONSTRAINT: "different"
      QA_SIMILARITY_MULTIHOP_MIN_DOCS: 2
      QA_GEN_VAR_QUESTION_STYLES: "easy,medium,hard,extra_hard"
      # JOBS is a list → passed as kwargs["similarity_jobs"], not an env var
      QA_SIMILARITY_MULTIHOP_JOBS:
        - QA_SIMILARITY_MULTIHOP_MODE: "above"
          QA_SIMILARITY_MULTIHOP_THRESHOLD: 0.80
          QA_GEN_VAR_QUESTION_STYLES: "easy,medium,hard"
        - QA_SIMILARITY_MULTIHOP_MODE: "range"
          QA_SIMILARITY_MULTIHOP_MIN: 0.30
          QA_SIMILARITY_MULTIHOP_MAX: 0.50
          QA_GEN_VAR_QUESTION_STYLES: "easy,medium"
        - QA_SIMILARITY_MULTIHOP_MODE: "below"
          QA_SIMILARITY_MULTIHOP_THRESHOLD: 0.30
          QA_SIMILARITY_MULTIHOP_MIN_MATCHING_WORDS: 10
          QA_GEN_VAR_QUESTION_STYLES: "hard"
```

Execution order:
1. `preprocess` runs once (skipped if collection already populated).
2. Each entry under `runs` executes sequentially: generator + all tail phases,
   writing into `output_dir`.

### 11.5 Merging multi-run outputs

```bash
python -m SeedDataGen.merge_runs \
    out/qa_gen_var/final.jsonl \
    out/qa_local_multihop/final.jsonl \
    out/qa_similarity_multihop/final.jsonl \
    -o combined/final.jsonl
```

`id` is renumbered monotonically.  `GEN_TYPE` on every row identifies which
generator produced it.

---

## 12. Key design decisions

**Why JSONL and not a database?** Every phase is a simple streaming
transformation; JSONL makes it trivial to inspect intermediate outputs, resume
from any point, and re-run a single phase in isolation (`--only`).

**Why `sample_id` is a list.** Single-chunk rows use `[hf_row_id]`; multihop
rows use all chunk ids.  This keeps the type uniform across all phases — no
special-casing needed downstream.

**Why `extra="ignore"` on all schemas.** New upstream fields propagate through
unmodified without breaking validation.  Downstream schemas only declare what
they actively use.

**Why env vars (not constructor args) for phase config.** YAML `config:` blocks
write to `os.environ` temporarily via a context manager.  `pydantic-settings`
reads them at instantiation time inside `run()`.  This means the same phase
class can be configured differently in every run entry without any code change.

**Why similarity jobs are kwargs, not env vars.** A list of dicts cannot be
serialised as a single env var string cleanly.  The runner detects the
`QA_SIMILARITY_MULTIHOP_JOBS` key in the run config, keeps it out of the env
context, and passes it as `kwargs["similarity_jobs"]` directly to `phase.run()`.
Any other list/dict-valued config should follow the same pattern using the
`kwargs:` YAML key.
