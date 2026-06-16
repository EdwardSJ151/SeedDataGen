# SeedDataGen — Phase Reference

## Code layout

Phases live in role-based subpackages and self-register; the runner discovers
every `phase_*.py` recursively:

```
preprocess/  chroma_preprocess, chunk_index, chunk_retrieval
generator/   qa_gen, qa_gen_var, qa_local_multihop, qa_similarity_multihop
editor/      answer_rewrite, conv_expand (deprecated), conv_expand_var
filter/      qa_filter, conv_filter
judge/       judge
dedup/       embed_filter
```

Each role dir has its own `prompts.py`; the root `prompts.py` re-exports them
for backwards compatibility.

## Row shape

Generators emit a provenance-rich row.  `sample_id` is a **list of HF row ids**
and `sample_text` is a **`{hf_row_id: text}` dict** (single-element for
single-chunk generators):

```json
{ "sample_id": [42], "sample_text": {"42": "..."}, "GEN_TYPE": "qa_gen_var",
  "num_chunks": 1, "doc_constraint": null, "question_style": "hard" }
```

Phases that group rows derive a stable key with
`get_sample_group_key(sample_id)` (`"|".join(sorted(...))`); it is never stored.

## Phases

### `chroma_preprocess`
**Role:** PREPROCESS · **Output:** none (builds a Chroma vectorstore)

One-time preparation.  Streams the dataset and upserts every chunk into a
persistent Chroma collection (id = `hf_row_id`, document = chunk text, metadata
= `{hf_row_id, doc_id, chunk_index}`) so the multihop generators can retrieve
adjacent or similar chunks.  No-op if the collection already exists unless
`CHROMA_FORCE_REBUILD` is set.  Runs once, outside the generator → tail chain.

Config prefix: `CHROMA_` — key settings: `VECTORSTORE_NAME`, `PERSIST_DIR`,
`METADATA_HF_ROW_ID`, `METADATA_DOC_ID`, `METADATA_CHUNK_INDEX`,
`METADATA_TEXT`, `EMBED_MODEL`, `FORCE_REBUILD`.

---

### `qa_gen`
**Role:** GENERATOR · **Output:** `QARow`

Streams documents from the configured HuggingFace dataset.  For each document,
sends one LLM call with `QA_GENERATION_PROMPT` and parses up to 5
`Pergunta: ... Resposta: ...` pairs from the response.  Emits
`GEN_TYPE="qa_gen"`, `num_chunks=1`, `doc_constraint=null`.

- Supports resume: skips documents (by `hf_row_id`) already in the output.
- Config prefix: `QA_GEN_` — key settings: `TEMPERATURE`, `TOP_P`,
  `MAX_TOKENS`, `BATCH_SIZE`, `MAX_CONCURRENT`.
- Dataset settings live in global `config.py`: `DATASET_ID`, `DATASET_SPLIT`,
  `DATASET_TEXT_FIELD`, `DATASET_ID_FIELD` (HF row-id column; falls back to the
  streaming index), `DATASET_MAX_CHARS`, `DATASET_MIN_CHARS`.

---

### `qa_gen_var`
**Role:** GENERATOR · **Output:** `StyledQARow`

Like `qa_gen` but makes **one LLM call per configured style per document**.
Each call generates exactly one QA pair constrained to that style.  Output
rows carry a `question_style` field that downstream phases can use.

Available styles (defined in `prompts.QA_GEN_VAR_STYLE_INSTRUCTIONS`):

| Style | What it produces |
|-------|-----------------|
| `general` | Broad question about the main topic |
| `specific` | Focused on a concrete detail or fact |
| `compositional` | Requires combining multiple pieces of information |
| `comparative` | Compares two concepts, entities, or periods |
| `easy` | Direct question, answer requires little elaboration |
| `medium` | Light organisation or a single inference step |
| `hard` | Dense reasoning across multiple passages |
| `extra_hard` | Maximum complexity the text can still support |

Config prefix: `QA_GEN_VAR_` — key settings: `QUESTION_STYLES` (comma-separated),
`TEMPERATURE`, `TOP_P`, `MAX_TOKENS`, `BATCH_SIZE`, `MAX_CONCURRENT`.

> Use `conv_expand_var` (not `conv_expand`) after this phase to preserve
> style information in conversations.

---

### `qa_local_multihop`
**Role:** GENERATOR · **Output:** `StyledQARow` · **Requires:** `chroma_preprocess`

Multihop QA over **adjacent** chunks.  For each document in the Chroma
collection, slides a window of `NUM_CHUNKS` consecutive chunks (step
`WINDOW_STRIDE`) and makes one LLM call per window per style, emitting one QA
pair each.  Output rows carry `sample_id` = the window's HF row ids,
`sample_text` = `{hf_row_id: chunk_text}`, `GEN_TYPE="qa_local_multihop"`,
`num_chunks`, `doc_constraint=null`, `question_style`.

- Styles are shared with `qa_gen_var` via `QA_GEN_VAR_QUESTION_STYLES`.
- Resume: skips `(sample_group_key, question_style)` combos already present.
- Config prefix: `QA_LOCAL_MULTIHOP_` — `NUM_CHUNKS`, `WINDOW_STRIDE`,
  `TEMPERATURE`, `TOP_P`, `MAX_TOKENS`, `BATCH_SIZE`, `MAX_CONCURRENT`.
- Reads the collection via the `CHROMA_*` env block (`CHROMA_VECTORSTORE_NAME`,
  `CHROMA_PERSIST_DIR`, `CHROMA_EMBED_MODEL`).

---

### `qa_similarity_multihop`
**Role:** GENERATOR · **Output:** `StyledQARow` · **Requires:** `chroma_preprocess`

Multihop QA over **similarity-selected** chunk groups.  Groups are built by
vector similarity according to one or more *jobs*; for each group, one LLM call
per job-style is made.  Output rows add the similarity provenance fields
(`similarity_job_index`, `similarity_mode`, `similarity_threshold`,
`similarity_min`, `similarity_max`, `min_matching_words`,
`chunk_group_similarity`) plus the standard multihop fields.

Each job (`QA_SIMILARITY_MULTIHOP_MODE` = `above` | `below` | `range`) may set
`QA_SIMILARITY_MULTIHOP_THRESHOLD` / `_MIN` / `_MAX`, an optional lexical
`QA_SIMILARITY_MULTIHOP_MIN_MATCHING_WORDS` (lowercased, whitespace-split word
overlap per pair), and its own `QA_GEN_VAR_QUESTION_STYLES`.  `above`/`range`
use the positive prompt; `below` uses the negative prompt.

- **Pairwise exclusion:** once a group `{A,B,C}` is emitted, the pairs
  `(A,B),(A,C),(B,C)` are banned so no pair is reused across groups/jobs.  A
  resuming run rebuilds the exclusion set from the prior output.
- `doc_constraint` (`same` | `different`) and `min_docs` constrain group docs.
- `QA_SIMILARITY_MULTIHOP_JOBS` is a list, so it is passed to the generator as
  `kwargs["similarity_jobs"]` (not as an env var).
- Config prefix: `QA_SIMILARITY_MULTIHOP_` — `NUM_CHUNKS`, `DOC_CONSTRAINT`,
  `MIN_DOCS`, `MAX_CANDIDATES`, `TEMPERATURE`, `TOP_P`, `MAX_TOKENS`,
  `BATCH_SIZE`, `MAX_CONCURRENT`.

---

### `answer_rewrite` *(optional)*
**Role:** EDITOR · **Input:** `QARow` · **Output:** `QARow`

Rewrites each answer to be more information-rich using only content already
present in the source text.  The question is unchanged.  Falls back to the
original answer if the LLM call fails.

Config prefix: `ANSWER_REWRITE_` — key settings: `TEMPERATURE`, `TOP_P`,
`MAX_TOKENS`, `BATCH_SIZE`, `MAX_CONCURRENT`.

---

### `qa_filter`
**Role:** FILTER · **Input:** `QARow` · **Output:** `QARow`

Pure-Python heuristic filter, no LLM calls.  Groups rows by
`get_sample_group_key(sample_id)` and for each group:

- Drops pairs whose answer is shorter than `min_answer_len` characters.
- Drops near-duplicate questions (Levenshtein distance ≤ `levenshtein_threshold`).

Config prefix: `QA_FILTER_` — key settings: `MIN_ANSWER_LEN` (default 10),
`LEVENSHTEIN_THRESHOLD` (default 20), `BATCH_SIZE`.

---

### `conv_expand` *(deprecated)*
**Role:** EDITOR · **Input:** `QARow` · **Output:** `ConversationRow`

> Deprecated — prefer `conv_expand_var`, which adds style cycling, GEN_TYPE
> prompt routing, and multi-chunk (multihop) support.  Retained for legacy
> single-style pipelines only.

Turns each QA pair into a multi-turn conversation.  The seed pair becomes the
first user/assistant exchange; then N extra rounds are added via two LLM calls
per round (simulated user, then assistant).

**naive_gen True** (default): user turns see current history only.

**naive_gen False** (diversity mode): within each batch, items are grouped by
`sample_id` and sorted by `id`.  Each user turn for a non-first item also
receives the seed questions of earlier items from the same sample, instructing
the model to explore aspects not already covered.  First item per sample and
`n_user_turns_max <= 1` fall back to naive automatically.

Config prefix: `CONV_EXPAND_` — key settings: `NAIVE_GEN`, `N_USER_TURNS_MIN`,
`N_USER_TURNS_MAX`, `USER_TURN_TEMPERATURE`, `ASSISTANT_TURN_TEMPERATURE`,
`BATCH_SIZE`, `MAX_CONCURRENT`.

---

### `conv_expand_var`
**Role:** EDITOR · **Input:** `StyledQARow` · **Output:** `ConversationRow`

Style-aware variant of `conv_expand`.  Accepts the output of `qa_gen_var` and
both multihop generators.  Extra user turns cycle through the configured styles
in order, starting after the seed style.  Style instructions are injected only
into the user-turn prompt — the assistant never sees them.

- **Full passthrough:** every upstream field (`GEN_TYPE`, `num_chunks`,
  `similarity_*`, …) is preserved on the emitted conversation row.
- **GEN_TYPE routing:** the naive user-turn and assistant-turn prompts are
  selected per `GEN_TYPE` via `CONV_EXPAND_USER_TURN_BY_GEN_TYPE` /
  `CONV_EXPAND_ASSISTANT_TURN_BY_GEN_TYPE` (fallback to the var prompts).
- **Multi-chunk context:** `sample_text` (string or `{id: text}` dict) is
  rendered for the prompt via `format_sample_text`.
- Grouping (diversity mode) is by `get_sample_group_key(sample_id)`.

**naive_gen False** additionally uses prior-seed-question context (same logic
as `conv_expand` diversity mode).

Config prefix: `CONV_EXPAND_VAR_` — same settings as `conv_expand` plus
`QUESTION_STYLES` (comma-separated, must overlap with `QA_GEN_VAR_STYLE_INSTRUCTIONS`).

---

### `conv_filter`
**Role:** FILTER · **Input:** `ConversationRow` · **Output:** `ConversationRow`

Pure-Python heuristic filter.  Drops a conversation if any of the following:

- Fewer than `min_messages` total messages.
- Any assistant turn shorter than `assistant_min_len` characters.
- Any two user messages are near-duplicates (Levenshtein ≤ `user_levenshtein_threshold`).
- Any two adjacent messages are near-duplicates (Levenshtein ≤ `adjacent_levenshtein_threshold`).

Config prefix: `CONV_FILTER_` — key settings: `MIN_MESSAGES` (default 4),
`ASSISTANT_MIN_LEN` (default 10), `USER_LEVENSHTEIN_THRESHOLD` (default 20),
`ADJACENT_LEVENSHTEIN_THRESHOLD` (default 20), `BATCH_SIZE`.

---

### `judge`
**Role:** JUDGE · **Input:** `ConversationRow` · **Output:** `JudgedConversationRow`

Calls the LLM with `JUDGE_PROMPT` for each conversation.  The model scores
five dimensions (1–5): fidelidade, correção, clareza, coerência, diversidade.
Only rows whose average score is above `min_avg_score` are kept.  Adds
`scores` (dict) and `avg_score` (float) to each output row.

Config prefix: `JUDGE_` — key settings: `TEMPERATURE` (default 0.0),
`MIN_AVG_SCORE` (default 4.0), `TOP_P`, `MAX_TOKENS`, `BATCH_SIZE`,
`MAX_CONCURRENT`.

---

### `embed_filter`
**Role:** DEDUP · **Input:** `JudgedConversationRow` · **Output:** `JudgedConversationRow`

Embedding-based deduplication using `sentence-transformers`.  Groups
conversations by `get_sample_group_key(sample_id)`, embeds each as flat text, then greedily keeps
only conversations whose cosine similarity to every already-kept conversation
in the group is ≤ `similarity_threshold`.

Config prefix: `EMBED_FILTER_` — key settings: `MODEL_NAME` (default
`sentence-transformers/all-MiniLM-L6-v2`), `DEVICE` (`cpu` or `cuda`),
`SIMILARITY_THRESHOLD` (default 0.95), `BATCH_SIZE`.

---

## Adding a new phase

See [ARCHITECTURE.md §8](ARCHITECTURE.md) for a full walkthrough including a
code template, resume semantics, generator-specific rules, and how to handle
non-scalar config via kwargs.

---

## Running multiple pipelines from one YAML

The multi-run format (`pipeline_multihop.yaml`) runs several full pipelines
sequentially:

- `preprocess:` — a single PREPROCESS phase (e.g. `chroma_preprocess`) run once.
- `tail:` — the shared downstream phases appended after every run's generator.
- `runs:` — a list of `{generator, output_dir, config}`; each is a full pipeline
  (`generator` → `tail`) writing into its own `output_dir`.  The generator
  output is `<output_dir>/<generator>.jsonl`; tail outputs are prefixed with
  `output_dir`.

Config keys are passed verbatim as env vars (no case folding), so they must
match the phase BaseSettings env names.  Non-scalar config values (e.g.
`QA_SIMILARITY_MULTIHOP_JOBS`) are routed to the phase as kwargs instead of env.

The runner auto-detects the format: a `runs:` key selects multi-run, otherwise
the legacy `pipeline:` format is used (both supported).

Combine the per-run results afterwards:

```bash
python -m SeedDataGen.merge_runs out/*/final.jsonl -o combined/final.jsonl
```

`merge_runs` concatenates the inputs and renumbers `id` monotonically;
`GEN_TYPE` on each row preserves provenance.

---

## Compatibility rules

The runner validates phase wiring at startup before any LLM call:

| From role | Allowed next roles |
|-----------|--------------------|
| GENERATOR | EDITOR, FILTER, JUDGE, DEDUP |
| EDITOR | EDITOR, FILTER, JUDGE, DEDUP |
| FILTER | EDITOR, FILTER, JUDGE, DEDUP |
| JUDGE | FILTER, DEDUP |
| DEDUP | EDITOR, FILTER |

`PREPROCESS` is intentionally absent from this table: it runs once outside the
generator → tail chain and is not validated as a consecutive transition.

Schema compatibility is also checked: the output schema of phase N must be a
subclass of the input schema of phase N+1.  Add `force: true` to a YAML entry
to bypass the role-transition check (schema check always runs).
