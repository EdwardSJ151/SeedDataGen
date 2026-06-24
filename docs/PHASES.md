# SeedDataGen — Phase Reference

Per-phase settings and behavior. For the file index see [`CODEBASE_MAP.md`](../CODEBASE_MAP.md);
for data contracts, resume, YAML formats, and extension see [`ARCHITECTURE.md`](ARCHITECTURE.md).

## Config conventions

- Each phase reads settings from a `pydantic-settings` class with an env **prefix** (e.g. `QA_GEN_VAR_`).
  Set keys in the YAML `env:` block or a phase's `config:` block using the full env var name
  (`QA_GEN_VAR_TEMPERATURE`, not nested YAML keys).
- **`BATCH_SIZE`** is global: the runner passes `batch_size` to every phase's `run()`.
- **Concurrency is per-phase**, not global. LLM phases cap in-flight requests with
  `{PREFIX}_MAX_CONCURRENT` (default 64). The global `MAX_CONCURRENT` in `config.py` is unused.
- List/dict config (e.g. `QA_SIMILARITY_MULTIHOP_JOBS`) goes in YAML `kwargs:` and is passed to
  `phase.run(**kwargs)`, not through env vars.

## Row shape (generators)

```json
{
  "id": 0,
  "origin_id": 0,
  "sample_id": ["42"],
  "sample_text": {"42": {"text": "...", "document_name": "ND-9-3"}},
  "question": "...",
  "answer": "...",
  "question_style": "hard",
  "GEN_TYPE": "qa_gen_var",
  "num_chunks": 1,
  "doc_constraint": null
}
```

Multihop rows use multiple entries in `sample_id` / `sample_text`. Group rows with
`get_sample_group_key(sample_id)` — never use a raw list as a dict key.

Required dataset env vars at run time: `DATASET_DOC_NAME_FIELD` plus the usual `DATASET_*` keys.
`DATASET_ID_FIELD` must match a column in the HF dataset (no silent fallback to the stream index).

---

## Phases

### `chroma_preprocess`
**Role:** PREPROCESS · **Output:** Chroma collection (no JSONL)

Streams the HF dataset and upserts every chunk into a persistent Chroma collection for multihop
generators. No-op if the collection exists unless `CHROMA_FORCE_REBUILD=true`.

**Prefix:** `CHROMA_` — `VECTORSTORE_NAME`, `PERSIST_DIR`, `EMBED_MODEL`, `FORCE_REBUILD`,
`METADATA_HF_ROW_ID`, `METADATA_DOC_ID`, `METADATA_CHUNK_INDEX`, `METADATA_TEXT`, `UPSERT_BATCH_SIZE`.

---

### `qa_gen`
**Role:** GENERATOR · **Output:** `QARow`

One LLM call per chunk; parses up to 5 `Pergunta: … Resposta: …` pairs. Resume skips chunks whose
`hf_row_id` is already in the output (`get_processed_sample_ids`).

**Prefix:** `QA_GEN_` — `TEMPERATURE`, `TOP_P`, `MAX_TOKENS`, `MAX_CONCURRENT`.

---

### `qa_gen_var`
**Role:** GENERATOR · **Output:** `StyledQARow`

One LLM call per **style** per chunk. Styles (in `generator/prompts.py` → `QA_GEN_VAR_STYLE_INSTRUCTIONS`):

| Style | Intent |
|-------|--------|
| `general`, `specific`, `compositional`, `comparative` | Classic difficulty/structure axes |
| `easy`, `medium`, `hard`, `extra_hard` | Graded difficulty |
| `applied_context` | Practical user-style question grounded in the text |

**Prefix:** `QA_GEN_VAR_` — `QUESTION_STYLES` (comma-separated), `TEMPERATURE`, `TOP_P`,
`MAX_TOKENS`, `MAX_CONCURRENT`.

With 4 styles and default `BATCH_SIZE=32`, each batch creates 128 tasks but only
`QA_GEN_VAR_MAX_CONCURRENT` (default 64) run at once.

Use `conv_expand_var` downstream to preserve `question_style`.

---

### `qa_local_multihop`
**Role:** GENERATOR · **Output:** `StyledQARow` · **Requires:** `chroma_preprocess`

Slides a window of `NUM_CHUNKS` adjacent chunks per document (`WINDOW_STRIDE` step); one LLM call
per window per style. Resume skips `(sample_group_key, question_style)` pairs already on disk.

**Prefix:** `QA_LOCAL_MULTIHOP_` — `NUM_CHUNKS`, `WINDOW_STRIDE`, `TEMPERATURE`, `TOP_P`,
`MAX_TOKENS`, `MAX_CONCURRENT`. Styles via `QA_GEN_VAR_QUESTION_STYLES`. Chroma via `CHROMA_*`.

---

### `qa_similarity_multihop`
**Role:** GENERATOR · **Output:** `StyledQARow` · **Requires:** `chroma_preprocess`

Builds chunk groups by vector similarity per **job** (`above` | `below` | `range`). Adds similarity
provenance fields (`similarity_mode`, `similarity_threshold`, `chunk_group_similarity`, …).
Pairwise exclusion prevents reusing the same chunk pair across groups (rebuilt from output on resume).

`QA_SIMILARITY_MULTIHOP_JOBS` is a YAML list → passed as `kwargs["similarity_jobs"]`.

**Prefix:** `QA_SIMILARITY_MULTIHOP_` — `NUM_CHUNKS`, `DOC_CONSTRAINT`, `MIN_DOCS`, `MAX_CANDIDATES`,
`TEMPERATURE`, `TOP_P`, `MAX_TOKENS`, `MAX_CONCURRENT`.

---

### `rewrite_gen`
**Role:** GENERATOR · **Output:** `ConversationRow` (single-turn)

Single-turn rewrite/summarization per chunk × style. Two LLM calls per pair — a user turn that writes
ONLY the request, then an assistant turn that answers it as internal knowledge (no meta-references,
emits `REFUSAL_STRING` when unfounded) — assembled into `messages=[user, assistant]`. Because it emits
a finished `ConversationRow`, it **skips** `qa_filter`/`conv_expand_var` and goes straight to
`conv_filter → judge → embed_filter` (see `pipeline_rewrite_myver.yaml`, a separate pipeline). Styles
(`generator/prompts.py` → `REWRITE_GEN_STYLE_INSTRUCTIONS`): `summary`, `simplify`, `focus`.

**Prefix:** `REWRITE_GEN_` — `QUESTION_STYLES`, separate `USER_TURN_*` / `ASSISTANT_TURN_*`
temperature/top_p/max_tokens, `BATCH_SIZE`, `MAX_CONCURRENT`.

---

### `answer_rewrite` *(optional)*
**Role:** EDITOR · **Input/Output:** `QARow`

Rewrites each answer using only the source text; question unchanged. Falls back to the original
answer on LLM failure.

**Prefix:** `ANSWER_REWRITE_` — `TEMPERATURE`, `TOP_P`, `MAX_TOKENS`, `MAX_CONCURRENT`.

---

### `qa_filter`
**Role:** FILTER · **Input:** `QARow` · **Output:** `StyledQARow`

Heuristic only. Per `sample_group_key`: drop short answers and near-duplicate questions (Levenshtein).
Declared output is `StyledQARow` so `qa_gen_var → qa_filter → conv_expand_var` validates; the phase
is a dict passthrough that preserves all upstream fields including `question_style`.

**Prefix:** `QA_FILTER_` — `MIN_ANSWER_LEN`, `LEVENSHTEIN_THRESHOLD`.

---

### `conv_expand` *(deprecated)*
**Role:** EDITOR · **Input:** `QARow` · **Output:** `ConversationRow`

Legacy single-style expansion. Prefer `conv_expand_var`.

**Prefix:** `CONV_EXPAND_` — `NAIVE_GEN`, `N_USER_TURNS_MIN/MAX`, turn temperatures, `MAX_CONCURRENT`.

---

### `conv_expand_var`
**Role:** EDITOR · **Input:** `StyledQARow` · **Output:** `ConversationRow`

Seeds from each QA pair, then adds N user→assistant rounds. Extra turns cycle styles; diversity mode
(`NAIVE_GEN=false`) steers away from prior seed questions on the same source. Each turn is an immutable
system persona + ONE user message carrying the `<documento>` chunk; user-turn messages are routed by
`GEN_TYPE` via the `CONV_EXPAND_USER_TURN_*_USER_MSG_BY_GEN_TYPE` maps in `editor/prompts.py`. The
assistant emits `REFUSAL_STRING` when the chunk lacks the basis.

Each turn's LLM call acquires the semaphore individually (~2 × N sequential calls per row).

**Prefix:** `CONV_EXPAND_VAR_` — same as `conv_expand` plus `QUESTION_STYLES` and
`FULL_HISTORY_FOR_USER_TURN` (user turn sees full history vs. prior user questions only).

---

### `conv_filter`
**Role:** FILTER · **Input/Output:** `ConversationRow`

First **truncates** at the first assistant refusal (`utils.is_refusal` / `REFUSAL_STRING`) — dropping
that pair and everything after — then drops conversations with too few messages, short assistant turns,
duplicate user turns, or duplicate adjacent messages. So an early refusal collapses to the seed pair
and is dropped by `MIN_MESSAGES`.

**Prefix:** `CONV_FILTER_` — `MIN_MESSAGES`, `ASSISTANT_MIN_LEN`, `USER_LEVENSHTEIN_THRESHOLD`,
`ADJACENT_LEVENSHTEIN_THRESHOLD`.

---

### `judge`
**Role:** JUDGE · **Input:** `ConversationRow` · **Output:** `JudgedConversationRow`

LLM scores five dimensions (1–5): fidelidade, correção, clareza, coerência, diversidade. Keeps rows
with `avg_score > min_avg_score`. Unparseable judge output is dropped.

**Prefix:** `JUDGE_` — `MIN_AVG_SCORE` (default 4.0), `TEMPERATURE` (default 0.0), `TOP_P`,
`MAX_TOKENS`, `MAX_CONCURRENT`.

---

### `embed_filter`
**Role:** DEDUP · **Input/Output:** `JudgedConversationRow`

Greedy embedding dedup **within each `sample_group_key`**. Default model runs on CPU.

**Prefix:** `EMBED_FILTER_` — `MODEL_NAME`, `DEVICE`, `SIMILARITY_THRESHOLD`.

---

## Role compatibility

Checked at startup (`base_phase.py` → `COMPATIBLE_TRANSITIONS`). `force: true` in YAML bypasses
the role check only; schema check always runs.

| From | Allowed next |
|------|----------------|
| GENERATOR | EDITOR, FILTER, JUDGE, DEDUP |
| EDITOR | EDITOR, FILTER, JUDGE, DEDUP |
| FILTER | EDITOR, FILTER, JUDGE, DEDUP |
| JUDGE | FILTER, DEDUP |
| DEDUP | EDITOR, FILTER |

`PREPROCESS` runs once outside this chain. Schema rule: `issubclass(prev.output_schema, curr.input_schema)`.
