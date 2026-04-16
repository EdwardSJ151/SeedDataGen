# SeedDataGen — Phase Reference

## Phases

### `qa_gen`
**Role:** GENERATOR · **Output:** `QARow`

Streams documents from the configured HuggingFace dataset.  For each document,
sends one LLM call with `QA_GENERATION_PROMPT` and parses up to 5
`Pergunta: ... Resposta: ...` pairs from the response.

- Supports resume: skips documents already written to the output file.
- Config prefix: `QA_GEN_` — key settings: `TEMPERATURE`, `TOP_P`,
  `MAX_TOKENS`, `BATCH_SIZE`, `MAX_CONCURRENT`.
- Dataset settings live in global `config.py`: `DATASET_ID`, `DATASET_SPLIT`,
  `DATASET_TEXT_FIELD`, `DATASET_MAX_CHARS`, `DATASET_MIN_CHARS`.

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

Pure-Python heuristic filter, no LLM calls.  Groups rows by `sample_id` and
for each group:

- Drops pairs whose answer is shorter than `min_answer_len` characters.
- Drops near-duplicate questions (Levenshtein distance ≤ `levenshtein_threshold`).

Config prefix: `QA_FILTER_` — key settings: `MIN_ANSWER_LEN` (default 10),
`LEVENSHTEIN_THRESHOLD` (default 20), `BATCH_SIZE`.

---

### `conv_expand`
**Role:** EDITOR · **Input:** `QARow` · **Output:** `ConversationRow`

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

Style-aware variant of `conv_expand`.  Requires `qa_gen_var` output.  Extra
user turns cycle through the configured styles in order, starting after the
seed style.  Style instructions are injected only into the user-turn prompt —
the assistant never sees them.

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
conversations by `sample_id`, embeds each as flat text, then greedily keeps
only conversations whose cosine similarity to every already-kept conversation
in the group is ≤ `similarity_threshold`.

Config prefix: `EMBED_FILTER_` — key settings: `MODEL_NAME` (default
`sentence-transformers/all-MiniLM-L6-v2`), `DEVICE` (`cpu` or `cuda`),
`SIMILARITY_THRESHOLD` (default 0.95), `BATCH_SIZE`.

---

## Adding a new phase

1. Create `SeedDataGen/phase_<name>.py`.
2. Define a `*Config(BaseSettings)` with your env prefix.
3. Write a class that inherits `Phase`, sets `name`, `role`, `input_schema`,
   `output_schema`, and implements `async def run(self, input_file, output_file, **kwargs)`.
4. Decorate it with `@register`.
5. Add it to `pipeline.yaml`.

The runner auto-discovers all `phase_*.py` files on startup.

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

Schema compatibility is also checked: the output schema of phase N must be a
subclass of the input schema of phase N+1.  Add `force: true` to a YAML entry
to bypass the role-transition check (schema check always runs).
