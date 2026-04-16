"""
Phase: embed_filter

Embedding-based deduplication.  For each sample_id group, embeds all
conversations as flat text, then greedily keeps only conversations whose
cosine similarity to every already-kept conversation is ≤ similarity_threshold.

Role:   DEDUP
Input:  JudgedConversationRow
Output: JudgedConversationRow  (re-numbered ids; input_id preserved)
"""

from collections import defaultdict
from typing import Dict, List

import numpy as np
from pydantic_settings import BaseSettings, SettingsConfigDict
from tqdm import tqdm

from SeedDataGen.base_phase import Phase, PhaseRole
from SeedDataGen.registry import register
from SeedDataGen.schemas import JudgedConversationRow
from SeedDataGen.utils import (
    count_jsonl_lines,
    get_last_processed_id,
    get_max_int_field,
    iter_jsonl_batches,
    write_jsonl_batch,
)


class EmbedFilterConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EMBED_FILTER_", env_file=".env", extra="ignore")

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"
    similarity_threshold: float = 0.95
    batch_size: int = 32


def _load_model(cfg: EmbedFilterConfig):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(cfg.model_name, device=cfg.device)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _convo_to_text(messages: List[Dict[str, str]]) -> str:
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


def _embedding_filter(conversations: List[Dict], model, threshold: float) -> List[Dict]:
    if not conversations:
        return []

    texts = [_convo_to_text(c["messages"]) for c in conversations]
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


@register
class EmbedFilterPhase(Phase):
    name = "embed_filter"
    role = PhaseRole.DEDUP
    input_schema = JudgedConversationRow
    output_schema = JudgedConversationRow

    async def run(self, input_file: str, output_file: str, **kwargs) -> None:
        cfg = EmbedFilterConfig()
        batch_size: int = kwargs.get("batch_size", cfg.batch_size)

        total_lines = count_jsonl_lines(input_file)
        print(f"[embed_filter] {total_lines} lines in {input_file}")
        print(f"[embed_filter] model: {cfg.model_name}  device: {cfg.device}")

        model = _load_model(cfg)

        last_out_id = get_last_processed_id(output_file)
        next_id = last_out_id + 1 if last_out_id >= 0 else 0

        last_input_id = get_max_int_field(output_file, "input_id")
        resume_from = last_input_id + 1 if last_input_id >= 0 else 0

        groups: Dict[int, List[Dict]] = defaultdict(list)
        for batch in iter_jsonl_batches(input_file, batch_size=batch_size, start_from_id=resume_from):
            for item in batch:
                groups[item["sample_id"]].append(item)

        n_loaded = sum(len(g) for g in groups.values())
        print(f"[embed_filter] {n_loaded} conversations in {len(groups)} sample_id groups")

        kept_total = 0
        dropped_total = 0
        out_buf: List[Dict] = []
        pbar = tqdm(total=len(groups), desc="[embed_filter]")

        for sample_id in sorted(groups.keys()):
            convos = groups[sample_id]
            surviving = _embedding_filter(convos, model, cfg.similarity_threshold)

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
        n_in = kept_total + dropped_total
        print(
            f"[embed_filter] done — kept {kept_total}, dropped {dropped_total} "
            f"(from {n_in} conversations, {len(groups)} groups) → {output_file}"
        )
