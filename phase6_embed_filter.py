"""
Phase 6: Embedding Similarity Filter (per sample_id)

For each sample_id group, embed every conversation as text, then greedily
keep only conversations whose cosine similarity with every already-kept
conversation is ≤ EMBED_SIMILARITY_THRESHOLD.

Rows carry origin_id unchanged from Phase 1.
Resume: tracks last processed input_id so a restart skips already-done groups.

Input:  Phase 5 JSONL  (id, input_id, origin_id, sample_id, sample_text, messages, scores, avg_score)
Output: Phase 6 JSONL  (id re-numbered, input_id saved, origin_id preserved; final output)
"""

import argparse
from collections import defaultdict
from typing import Dict, List

import numpy as np
from tqdm import tqdm

from SeedDataGen.config import (
    EMBED_MODEL_NAME,
    EMBED_DEVICE,
    EMBED_SIMILARITY_THRESHOLD,
    BATCH_SIZE,
    PHASE5_OUTPUT,
    PHASE6_OUTPUT,
)
from SeedDataGen.utils import (
    iter_jsonl_batches,
    write_jsonl_batch,
    get_last_processed_id,
    get_max_int_field,
    count_jsonl_lines,
    format_conversation_history,
)


def _load_embed_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBED_MODEL_NAME, device=EMBED_DEVICE)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def convo_to_text(messages: List[Dict[str, str]]) -> str:
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


def embedding_filter(
    conversations: List[Dict],
    model,
    threshold: float = EMBED_SIMILARITY_THRESHOLD,
) -> List[Dict]:
    if not conversations:
        return []

    texts = [convo_to_text(c["messages"]) for c in conversations]
    embeddings = model.encode(texts, show_progress_bar=False)

    kept_indices: List[int] = []
    kept_embeddings: List[np.ndarray] = []

    for i, emb_i in enumerate(embeddings):
        keep = True
        for emb_j in kept_embeddings:
            if _cosine_similarity(emb_i, emb_j) > threshold:
                keep = False
                break
        if keep:
            kept_indices.append(i)
            kept_embeddings.append(emb_i)

    return [conversations[i] for i in kept_indices]


def main(
    input_file: str = PHASE5_OUTPUT,
    output_file: str = PHASE6_OUTPUT,
    batch_size: int = BATCH_SIZE,
):
    total = count_jsonl_lines(input_file)
    print(f"Phase 6 — reading {total} conversations from {input_file}")
    print(f"Embedding model: {EMBED_MODEL_NAME}  device: {EMBED_DEVICE}")

    model = _load_embed_model()

    last_out_id = get_last_processed_id(output_file)
    next_id = last_out_id + 1 if last_out_id >= 0 else 0

    last_input_id = get_max_int_field(output_file, "input_id")
    resume_from_input = last_input_id + 1 if last_input_id >= 0 else 0

    # Load everything from where we left off, group by sample_id
    groups: Dict[int, List[Dict]] = defaultdict(list)
    for batch in iter_jsonl_batches(
        input_file,
        batch_size=batch_size,
        start_from_id=resume_from_input,
    ):
        for item in batch:
            groups[item["sample_id"]].append(item)

    kept_total = 0
    dropped_total = 0
    out_buf: List[Dict] = []
    pbar = tqdm(total=len(groups), desc="Phase 6: embedding filter")

    for sample_id in sorted(groups.keys()):
        convos = groups[sample_id]
        surviving = embedding_filter(convos, model)

        for item in surviving:
            item["input_id"] = item["id"]
            item["id"] = next_id
            next_id += 1
            out_buf.append(item)

        kept_total += len(surviving)
        dropped_total += len(convos) - len(surviving)

        if len(out_buf) >= batch_size:
            write_jsonl_batch(output_file, out_buf)
            out_buf = []
        pbar.update(1)

    if out_buf:
        write_jsonl_batch(output_file, out_buf)

    pbar.close()
    print(f"Phase 6 done — kept {kept_total}, dropped {dropped_total} → {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 6: embedding similarity filter")
    parser.add_argument("--input", default=PHASE5_OUTPUT)
    parser.add_argument("--output", default=PHASE6_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    main(args.input, args.output, args.batch_size)
