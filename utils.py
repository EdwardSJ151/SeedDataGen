"""
Shared utilities for the SeedDataGen pipeline.

JSONL I/O with resume support, Levenshtein distance, QA parsing,
conversation formatting, score parsing, and Pydantic schema helpers.
"""

import json
import os
import re
from typing import Dict, Iterator, List, Optional, Type

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
    """Human-readable dump used inside prompts."""
    parts: List[str] = []
    for m in messages:
        role = "Usuário" if m["role"] == "user" else "Assistente"
        parts.append(f"{role}: {m['content']}")
    return "\n".join(parts)


def format_conversation_for_judge(messages: List[Dict[str, str]]) -> str:
    return format_conversation_history(messages)


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
