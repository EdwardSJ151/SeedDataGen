"""
Shared utilities for the SeedDataGen pipeline.

JSONL I/O with resume support, Levenshtein distance, QA parsing,
conversation formatting, score parsing, and Pydantic schema helpers.
"""

import json
import os
import re
from typing import Any, Dict, Iterator, List, Optional, Type, Union

from pydantic import BaseModel


# JSONL helpers (same patterns as PtPersonaIFGen)
def get_last_processed_id(filepath: str) -> int:
    """Return the highest `id` written to *filepath*, or -1 if empty/missing."""
    return get_max_int_field(filepath, "id")


def get_max_int_field(filepath: str, field: str) -> int:
    """Return the highest value of an integer *field* in a JSONL, or -1."""
    if not os.path.exists(filepath):
        return -1
    best = -1
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                val = obj.get(field)
                if isinstance(val, int) and val > best:
                    best = val
            except (json.JSONDecodeError, AttributeError):
                continue
    return best


def iter_jsonl_batches(
    path: str,
    batch_size: int,
    start_from_id: int = 0,
    required_fields: Optional[List[str]] = None,
) -> Iterator[List[Dict]]:
    """Yield batches of dicts from a JSONL, skipping rows with id < start_from_id."""
    required_fields = required_fields or []
    with open(path, "r", encoding="utf-8") as f:
        batch: List[Dict] = []
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not all(k in obj for k in required_fields):
                continue
            if not isinstance(obj.get("id"), int):
                continue
            if obj["id"] < start_from_id:
                continue
            batch.append(obj)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


def write_jsonl_batch(filepath: str, batch: List[Dict]) -> None:
    with open(filepath, "a", encoding="utf-8") as f:
        for obj in batch:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def write_jsonl_line(filepath: str, obj: Dict) -> None:
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def count_jsonl_lines(filepath: str) -> int:
    if not os.path.exists(filepath):
        return 0
    with open(filepath, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


# HuggingFace dataset field validation
def require_hf_field(rec: Dict, field: str, *, row_label: str = "row") -> Any:
    """
    Return ``rec[field]`` or raise ValueError if the column is absent or None.

    Used instead of silent fallbacks (e.g. streaming index) so misconfigured
    dataset column mappings fail fast.
    """
    if field not in rec or rec[field] is None:
        keys = sorted(rec.keys())
        raise ValueError(
            f"Required HF dataset column '{field}' is missing on {row_label}. "
            f"Available columns: {keys}. "
            f"Set DATASET_ID_FIELD / CHROMA_METADATA_* in the pipeline YAML to match "
            f"the dataset schema."
        )
    return rec[field]


def require_hf_int_field(rec: Dict, field: str, *, row_label: str = "row") -> int:
    """Like require_hf_field but coerces the value to int."""
    raw = require_hf_field(rec, field, row_label=row_label)
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"HF dataset column '{field}' must be an integer on {row_label}, got {raw!r}"
        ) from exc


def assert_hf_dataset_has_fields(
    ds_iter: Iterator,
    fields: List[str],
    *,
    dataset_id: str,
) -> Iterator:
    """
    Validate that the first streamed row contains all *fields*, then re-yield the
    full iterator.  Raises ValueError on an empty dataset or missing columns.
    """
    iterator = iter(ds_iter)
    try:
        first = next(iterator)
    except StopIteration:
        raise ValueError(f"HF dataset {dataset_id!r} is empty — nothing to process.")
    for field in fields:
        require_hf_field(first, field, row_label="first row")
    yield first
    yield from iterator


# Sample-text / sample-id helpers (multihop-aware)
def make_chunk_entry(text: str, document_name: str) -> Dict[str, str]:
    """Structured chunk value for JSONL sample_text dicts."""
    return {"text": text, "document_name": document_name}


def _chunk_text_and_name(value: Any) -> tuple[str, Optional[str]]:
    if isinstance(value, dict) and "text" in value:
        text = str(value.get("text", ""))
        raw_name = value.get("document_name")
        document_name = str(raw_name) if raw_name is not None else None
        return text, document_name
    return str(value), None


def _format_chunk_header(chunk_id: Any, document_name: Optional[str] = None) -> str:
    if document_name:
        return f"[Chunk {chunk_id} | {document_name}]"
    return f"[Chunk {chunk_id}]"


def format_sample_text(sample_text: Union[str, Dict, List[Dict]]) -> str:
    """
    Render *sample_text* into a flat string suitable for prompt injection.

    Accepts three shapes:
      - str:                 returned unchanged (legacy / single-chunk).
      - dict {id: text}:      rendered as labelled chunks (multihop).
      - dict {id: {text, document_name}}: chunk label includes document name.
      - list[{chunk_id/hf_row_id, text, document_name?}]: labelled chunks.
    """
    if isinstance(sample_text, str):
        return sample_text

    parts: List[str] = []
    if isinstance(sample_text, dict):
        for chunk_id, value in sample_text.items():
            text, document_name = _chunk_text_and_name(value)
            parts.append(f"{_format_chunk_header(chunk_id, document_name)}\n{text}")
    elif isinstance(sample_text, list):
        for entry in sample_text:
            if not isinstance(entry, dict):
                parts.append(str(entry))
                continue
            chunk_id = entry.get("chunk_id", entry.get("hf_row_id", "?"))
            text = entry.get("text", "")
            document_name = entry.get("document_name")
            doc_name = str(document_name) if document_name is not None else None
            parts.append(f"{_format_chunk_header(chunk_id, doc_name)}\n{text}")
    else:
        return str(sample_text)

    return "\n\n".join(parts)


def extract_doc_names(sample_text: Union[str, Dict, List[Dict]]) -> List[str]:
    """
    Return the unique document names referenced by *sample_text*, in first-seen
    order.  Empty list for legacy plain-string sample_text (no name available).
    """
    names: List[str] = []
    seen: set = set()

    def _add(name: Optional[str]) -> None:
        if name and name not in seen:
            seen.add(name)
            names.append(name)

    if isinstance(sample_text, dict):
        for value in sample_text.values():
            _, document_name = _chunk_text_and_name(value)
            _add(document_name)
    elif isinstance(sample_text, list):
        for entry in sample_text:
            if isinstance(entry, dict):
                name = entry.get("document_name")
                _add(str(name) if name is not None else None)
    return names


def format_sample_text_for_prompt(sample_text: Union[str, Dict, List[Dict]]) -> str:
    """
    Render *sample_text* for conversation/judge prompts as <documento> blocks.

    Unlike :func:`format_sample_text`, this never emits the leaked ``[Chunk N]``
    numbering — only the document name is kept (as a tag attribute), so the model
    can reference the document by name without surfacing chunk metadata.

    Shapes accepted mirror :func:`format_sample_text`; multihop renders one
    ``<documento>`` block per chunk, preserving each chunk's ``document_name``.
    """
    if isinstance(sample_text, str):
        return f"<documento>\n{sample_text}\n</documento>"

    def _block(text: str, document_name: Optional[str]) -> str:
        if document_name:
            return f'<documento nome="{document_name}">\n{text}\n</documento>'
        return f"<documento>\n{text}\n</documento>"

    parts: List[str] = []
    if isinstance(sample_text, dict):
        for value in sample_text.values():
            text, document_name = _chunk_text_and_name(value)
            parts.append(_block(text, document_name))
    elif isinstance(sample_text, list):
        for entry in sample_text:
            if not isinstance(entry, dict):
                parts.append(_block(str(entry), None))
                continue
            text = entry.get("text", "")
            document_name = entry.get("document_name")
            doc_name = str(document_name) if document_name is not None else None
            parts.append(_block(text, doc_name))
    else:
        return _block(str(sample_text), None)

    return "\n\n".join(parts)


def sample_text_from_chunks(chunks: List[Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
    """Build a {hf_row_id: {text, document_name}} mapping from Chroma chunk dicts."""
    out: Dict[str, Dict[str, str]] = {}
    for chunk in chunks:
        hf_row_id = str(chunk["hf_row_id"])
        document_name = chunk.get("document_name")
        if document_name is None:
            raise ValueError(
                f"Chunk {hf_row_id!r} is missing document_name. "
                f"Rebuild the Chroma collection with CHROMA_FORCE_REBUILD after setting "
                f"DATASET_DOC_NAME_FIELD."
            )
        out[hf_row_id] = make_chunk_entry(chunk["text"], str(document_name))
    return out


def is_summary_enabled() -> bool:
    """True when DATASET_SUMMARY_ENABLED is set in the environment."""
    from SeedDataGen.config import DATASET_SUMMARY_ENABLED

    return bool(os.environ.get("DATASET_SUMMARY_ENABLED", str(DATASET_SUMMARY_ENABLED)).lower() in (
        "true",
        "1",
        "yes",
    ))


def format_doc_summary(summary: Optional[str]) -> str:
    """Return a prompt block for *summary*, or '' when absent."""
    if not summary or not str(summary).strip():
        return ""
    return f"Resumo do documento:\n{str(summary).strip()}\n\n"


def format_doc_summaries_for_docs(summary_map: Dict[str, str], doc_ids: List[Any]) -> str:
    """Join formatted summary blocks for unique *doc_ids* (preserves first-seen order)."""
    parts: List[str] = []
    seen: set = set()
    for doc_id in doc_ids:
        key = str(doc_id)
        if key in seen:
            continue
        seen.add(key)
        block = format_doc_summary(summary_map.get(key))
        if block:
            parts.append(block.rstrip("\n"))
    if not parts:
        return ""
    return "\n\n".join(parts) + "\n\n"


def load_doc_summaries(
    *,
    dataset_id: Optional[str] = None,
    dataset_subset: Optional[str] = None,
    dataset_split: Optional[str] = None,
    chunk_type_field: Optional[str] = None,
    summary_value: Optional[str] = None,
    doc_id_field: Optional[str] = None,
    text_field: Optional[str] = None,
) -> Dict[str, str]:
    """
    Stream the HF dataset once and return {str(document_id): summary_text} for
    rows whose *chunk_type_field* equals *summary_value*.
    """
    from datasets import load_dataset

    from SeedDataGen.config import (
        DATASET_CHUNK_TYPE_FIELD,
        DATASET_DOC_ID_FIELD,
        DATASET_ID,
        DATASET_SPLIT,
        DATASET_SUBSET,
        DATASET_SUMMARY_TYPE_VALUE,
        DATASET_TEXT_FIELD,
    )

    dataset_id = dataset_id or os.environ.get("DATASET_ID", DATASET_ID)
    dataset_subset = dataset_subset or os.environ.get("DATASET_SUBSET", DATASET_SUBSET)
    dataset_split = dataset_split or os.environ.get("DATASET_SPLIT", DATASET_SPLIT)
    chunk_type_field = chunk_type_field or os.environ.get(
        "DATASET_CHUNK_TYPE_FIELD", DATASET_CHUNK_TYPE_FIELD
    )
    summary_value = summary_value or os.environ.get(
        "DATASET_SUMMARY_TYPE_VALUE", DATASET_SUMMARY_TYPE_VALUE
    )
    doc_id_field = doc_id_field or os.environ.get("DATASET_DOC_ID_FIELD", DATASET_DOC_ID_FIELD)
    text_field = text_field or os.environ.get("DATASET_TEXT_FIELD", DATASET_TEXT_FIELD)

    ds = load_dataset(dataset_id, dataset_subset, split=dataset_split, streaming=True)
    out: Dict[str, str] = {}
    for rec in ds:
        if rec.get(chunk_type_field) != summary_value:
            continue
        doc_id = rec.get(doc_id_field)
        if doc_id is None:
            continue
        text = rec.get(text_field, "")
        if not isinstance(text, str) or not text.strip():
            continue
        out[str(doc_id)] = text.strip()
    return out


def get_processed_sample_ids(filepath: str) -> set:
    """
    Return the set of all sample ids already present in *filepath*.

    Flattens list-valued sample_id (multihop rows) and includes scalar
    sample_id (single-chunk rows), so generators can skip already-processed
    source records regardless of row shape.
    """
    seen: set = set()
    if not os.path.exists(filepath):
        return seen
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = obj.get("sample_id")
            if isinstance(sid, list):
                seen.update(str(s) for s in sid)
            elif sid is not None:
                seen.add(str(sid))
    return seen


def get_sample_group_key(sample_id: Union[int, List[int]]) -> str:
    """
    Stable grouping key derived from *sample_id*.

    Phases that group rows (qa_filter, conv_expand_var diversity, embed_filter)
    use this instead of the raw sample_id, since multihop rows carry a list of
    HF row ids that cannot be used as a dict key directly.
    """
    if isinstance(sample_id, list):
        return "|".join(str(s) for s in sorted(sample_id))
    return str(sample_id)


# Pydantic schema helpers
_T = Type[BaseModel]


def validate_row(schema: _T, data: Dict) -> BaseModel:
    """Validate a raw dict against *schema*, returning a model instance."""
    return schema.model_validate(data)


def dump_row(obj: BaseModel) -> Dict:
    """Serialise a Pydantic model to a plain dict for JSONL output."""
    return obj.model_dump()


# Levenshtein distance
def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        return levenshtein(b, a)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[-1]


# QA parsing (Phase 1 output → list of dicts)
_QA_RE = re.compile(
    r"[Pp]ergunta:\s*(.+?)\s*[Rr]esposta:\s*(.+?)(?=\n[Pp]ergunta:|\Z)",
    re.DOTALL,
)


def parse_qa_pairs(text: str) -> List[Dict[str, str]]:
    """
    Extract QA pairs from model output.
    Expected format per line: "Pergunta: ... Resposta: ..."
    """
    pairs: List[Dict[str, str]] = []
    for m in _QA_RE.finditer(text):
        q = m.group(1).strip()
        a = m.group(2).strip()
        if q and a:
            pairs.append({"question": q, "answer": a})
    return pairs


# Conversation formatting (for prompt injection)
def format_conversation_history(messages: List[Dict[str, str]]) -> str:
    """Full user+assistant history."""
    parts: List[str] = []
    for m in messages:
        role = "Usuário" if m["role"] == "user" else "Assistente"
        parts.append(f"{role}: {m['content']}")
    return "\n".join(parts)


def format_user_history(messages: List[Dict[str, str]]) -> str:
    """User-only question history."""
    return "\n".join(
        f"- {m['content']}" for m in messages if m["role"] == "user"
    )


def format_conversation_for_judge(messages: List[Dict[str, str]]) -> str:
    return format_conversation_history(messages)


def _normalize_refusal(text: str) -> str:
    """Lowercase, strip whitespace, drop a trailing period for refusal matching."""
    return text.strip().casefold().rstrip(".").strip()


def is_refusal(text: str) -> bool:
    """
    True when *text* is the deterministic refusal string (config.REFUSAL_STRING),
    tolerant of case and a trailing period.
    """
    from SeedDataGen.config import REFUSAL_STRING

    return _normalize_refusal(text) == _normalize_refusal(REFUSAL_STRING)


# Score parsing (Phase 5 — LLM judge output)
_SCORE_LABELS = ["fidelidade", "correção", "clareza", "coerência", "diversidade"]
_SCORE_RE = re.compile(
    r"(?:fidelidade|correção|corre[cç][aã]o|clareza|coer[eê]ncia|diversidade)\s*:\s*(\d(?:[.,]\d)?)",
    re.IGNORECASE,
)


def parse_judge_scores(text: str) -> Optional[List[float]]:
    """
    Return [fidelidade, correção, clareza, coerência, diversidade] or None.
    Accepts comma or dot as decimal separator.
    """
    matches = _SCORE_RE.findall(text)
    if len(matches) < 5:
        return None
    scores: List[float] = []
    for raw in matches[:5]:
        scores.append(float(raw.replace(",", ".")))
    return scores
