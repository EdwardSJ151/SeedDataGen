"""
Chunk-retrieval helpers used by the multihop generators.

Two access patterns:

  local_window()          — ordered, adjacent chunks within a single document
                            (qa_local_multihop).

  similarity_groups_iter()— lazily yields groups of chunks selected by vector
                            similarity according to one or more configured jobs
                            (qa_similarity_multihop), with pairwise exclusion so
                            no chunk pair is reused across emitted groups.

All chunk dicts returned by these helpers have the shape:
    {"id": str, "hf_row_id": Any, "doc_id": Any, "chunk_index": int,
     "document_name": str, "text": str, "similarity": float}
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np


# Lexical-overlap helpers
def _word_set(text: str) -> set:
    """Lowercase, strip, whitespace-split word set (used for lexical overlap)."""
    return {w.strip() for w in text.lower().split() if w.strip()}


def _word_overlap(a: str, b: str) -> int:
    return len(_word_set(a) & _word_set(b))


# Local (adjacent) window retrieval
def doc_chunk_map(collection) -> Dict[Any, List[int]]:
    """Return {doc_id: sorted list of chunk_index} for every document."""
    data = collection.get(include=["metadatas"])
    out: Dict[Any, List[int]] = defaultdict(list)
    for meta in data["metadatas"]:
        out[meta.get("doc_id")].append(int(meta.get("chunk_index", 0)))
    for doc_id in out:
        out[doc_id].sort()
    return dict(out)


def get_doc_chunks(collection, doc_id: Any) -> List[Dict[str, Any]]:
    """Return all chunks of *doc_id*, ordered by chunk_index."""
    res = collection.get(
        where={"doc_id": {"$eq": doc_id}},
        include=["documents", "metadatas"],
    )
    chunks: List[Dict[str, Any]] = []
    for cid, doc, meta in zip(res["ids"], res["documents"], res["metadatas"]):
        chunks.append(
            {
                "id": cid,
                "hf_row_id": meta.get("hf_row_id", cid),
                "doc_id": meta.get("doc_id"),
                "chunk_index": int(meta.get("chunk_index", 0)),
                "document_name": str(meta.get("doc_name", "")),
                "text": doc,
                "similarity": 1.0,
            }
        )
    chunks.sort(key=lambda c: c["chunk_index"])
    return chunks


def local_window(
    collection,
    doc_id: Any,
    start_chunk_index: int,
    num_chunks: int,
) -> List[Dict[str, Any]]:
    """
    Return up to *num_chunks* consecutive chunks from *doc_id* starting at
    *start_chunk_index*, ordered by chunk_index.
    """
    res = collection.get(
        where={
            "$and": [
                {"doc_id": {"$eq": doc_id}},
                {"chunk_index": {"$gte": int(start_chunk_index)}},
                {"chunk_index": {"$lt": int(start_chunk_index) + int(num_chunks)}},
            ]
        },
        include=["documents", "metadatas"],
    )
    chunks: List[Dict[str, Any]] = []
    for cid, doc, meta in zip(res["ids"], res["documents"], res["metadatas"]):
        chunks.append(
            {
                "id": cid,
                "hf_row_id": meta.get("hf_row_id", cid),
                "doc_id": meta.get("doc_id"),
                "chunk_index": int(meta.get("chunk_index", 0)),
                "document_name": str(meta.get("doc_name", "")),
                "text": doc,
                "similarity": 1.0,
            }
        )
    chunks.sort(key=lambda c: c["chunk_index"])
    return chunks


# Similarity-based grouping
def _sim_ok(
    sim: float,
    mode: str,
    threshold: Optional[float],
    smin: Optional[float],
    smax: Optional[float],
) -> bool:
    if mode == "above":
        return threshold is None or sim >= threshold
    if mode == "below":
        return threshold is None or sim <= threshold
    if mode == "range":
        lo = smin if smin is not None else -1.0
        hi = smax if smax is not None else 1.0
        return lo <= sim <= hi
    return False


def _pairs(ids: List[str]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = sorted((ids[i], ids[j]))
            out.append((a, b))
    return out


def similarity_groups_iter(
    collection,
    num_chunks: int,
    jobs: List[Dict[str, Any]],
    doc_constraint: Optional[str],
    min_docs: int,
    used_pairs: set,
    *,
    max_candidates: int = 200,
) -> Iterator[Tuple[List[Dict[str, Any]], Dict[str, Any]]]:
    """
    Yield (chunk_group, job) tuples.  Each *chunk_group* is a list of
    *num_chunks* chunk dicts satisfying the job's similarity criterion plus the
    *doc_constraint* ("same" | "different" | None) and *min_docs* constraint.

    Pairwise exclusion: every pair of chunk ids in an emitted group is added to
    *used_pairs* (mutated in-place); any later candidate group that reuses an
    already-emitted pair is skipped.  This lets a resuming run rebuild
    *used_pairs* from prior output and continue safely.

    Similarities are computed in-memory from the stored embeddings (cosine).
    """
    data = collection.get(include=["embeddings", "documents", "metadatas"])
    ids: List[str] = list(data["ids"])
    if not ids:
        return

    embs = np.asarray(data["embeddings"], dtype=np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs_n = embs / norms

    meta_by_idx = data["metadatas"]
    docs_by_idx = data["documents"]

    def _chunk(idx: int, sim: float) -> Dict[str, Any]:
        meta = meta_by_idx[idx]
        return {
            "id": ids[idx],
            "hf_row_id": meta.get("hf_row_id", ids[idx]),
            "doc_id": meta.get("doc_id"),
            "chunk_index": int(meta.get("chunk_index", 0)),
            "document_name": str(meta.get("doc_name", "")),
            "text": docs_by_idx[idx],
            "similarity": float(sim),
        }

    for seed_idx in range(len(ids)):
        sims = embs_n @ embs_n[seed_idx]
        # Candidate order: most similar first (also serves "below" because the
        # qualifying low-similarity neighbours simply appear later in the list).
        order = np.argsort(-sims)[: max_candidates + 1]

        for job in jobs:
            mode = job.get("QA_SIMILARITY_MULTIHOP_MODE", "above")
            threshold = job.get("QA_SIMILARITY_MULTIHOP_THRESHOLD")
            smin = job.get("QA_SIMILARITY_MULTIHOP_MIN")
            smax = job.get("QA_SIMILARITY_MULTIHOP_MAX")
            min_words = job.get("QA_SIMILARITY_MULTIHOP_MIN_MATCHING_WORDS")

            seed_chunk = _chunk(seed_idx, 1.0)
            group = [seed_chunk]
            docs_seen = {seed_chunk["doc_id"]}

            for cand_idx in order:
                if len(group) >= num_chunks:
                    break
                if cand_idx == seed_idx:
                    continue
                sim = float(sims[cand_idx])
                if not _sim_ok(sim, mode, threshold, smin, smax):
                    continue
                cand = _chunk(int(cand_idx), sim)
                if doc_constraint == "different" and cand["doc_id"] in docs_seen:
                    continue
                if doc_constraint == "same" and cand["doc_id"] != seed_chunk["doc_id"]:
                    continue
                if min_words:
                    if any(_word_overlap(cand["text"], g["text"]) < min_words for g in group):
                        continue
                group.append(cand)
                docs_seen.add(cand["doc_id"])

            if len(group) < num_chunks:
                continue
            if len(docs_seen) < min_docs:
                continue

            group_pairs = _pairs([c["id"] for c in group])
            if any(p in used_pairs for p in group_pairs):
                continue
            for p in group_pairs:
                used_pairs.add(p)

            yield group, job
