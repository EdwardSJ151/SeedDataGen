"""
Phase: chroma_preprocess

One-time preparation step.  Streams the configured HuggingFace dataset and
upserts every chunk into a persistent Chroma collection so the multihop
generators (qa_local_multihop, qa_similarity_multihop) can retrieve adjacent or
similar chunks efficiently.

Role:   PREPROCESS  (runs once, outside the generator → tail chain)
Input:  HuggingFace dataset (streaming)
Output: a persistent Chroma collection (the *output_file* argument is unused)

If the collection already exists and is non-empty, the phase is a no-op unless
CHROMA_FORCE_REBUILD is set.
"""

import os
from typing import Any, Dict, List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict
from tqdm import tqdm

from SeedDataGen.base_phase import Phase, PhaseRole
from SeedDataGen.config import (
    DATASET_ID,
    DATASET_SPLIT,
    DATASET_SUBSET,
    DATASET_TEXT_FIELD,
)
from SeedDataGen.preprocess.chunk_index import (
    collection_exists,
    delete_collection,
    get_collection,
)
from SeedDataGen.registry import register
from SeedDataGen.schemas import BaseRow
from SeedDataGen.utils import (
    assert_hf_dataset_has_fields,
    require_hf_field,
    require_hf_int_field,
)


class ChromaPreprocessConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CHROMA_", env_file=".env", extra="ignore")

    vectorstore_name: str = "seeddatagen-chunks"
    persist_dir: str = ".cache/chroma"
    metadata_hf_row_id: str = "id"
    metadata_doc_id: str = "document_id"
    metadata_chunk_index: str = "chunk_index"
    # HF column for chunk text.  When unset, run() uses DATASET_TEXT_FIELD instead.
    metadata_text: Optional[str] = None
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    force_rebuild: bool = False
    upsert_batch_size: int = 256


def _stream_dataset():
    from datasets import load_dataset

    dataset_id = os.environ.get("DATASET_ID", DATASET_ID)
    dataset_subset = os.environ.get("DATASET_SUBSET", DATASET_SUBSET)
    dataset_split = os.environ.get("DATASET_SPLIT", DATASET_SPLIT)
    ds = load_dataset(dataset_id, dataset_subset, split=dataset_split, streaming=True)
    return iter(ds)


@register
class ChromaPreprocessPhase(Phase):
    name = "chroma_preprocess"
    role = PhaseRole.PREPROCESS
    input_schema = None
    output_schema = BaseRow  # not used for chaining; preprocess runs standalone

    async def run(self, input_file: str, output_file: str, **kwargs) -> None:
        cfg = ChromaPreprocessConfig()

        if cfg.force_rebuild:
            print(f"[chroma_preprocess] FORCE_REBUILD — dropping '{cfg.vectorstore_name}'")
            delete_collection(cfg.vectorstore_name, cfg.persist_dir)
        elif collection_exists(cfg.vectorstore_name, cfg.persist_dir):
            print(
                f"[chroma_preprocess] collection '{cfg.vectorstore_name}' already "
                f"populated in {cfg.persist_dir} — skipping (set CHROMA_FORCE_REBUILD to rebuild)."
            )
            return

        text_col = cfg.metadata_text or os.environ.get("DATASET_TEXT_FIELD", DATASET_TEXT_FIELD)

        collection = get_collection(cfg.vectorstore_name, cfg.persist_dir, cfg.embed_model)
        print(
            f"[chroma_preprocess] building '{cfg.vectorstore_name}' "
            f"(embed_model={cfg.embed_model}, text_column={text_col!r}) → {cfg.persist_dir}"
        )

        ds_iter = _stream_dataset()
        dataset_id = os.environ.get("DATASET_ID", DATASET_ID)
        ds_iter = assert_hf_dataset_has_fields(
            ds_iter,
            [cfg.metadata_hf_row_id, cfg.metadata_chunk_index, text_col],
            dataset_id=dataset_id,
        )
        pbar = tqdm(desc="[chroma_preprocess] upserting")

        buf_ids: List[str] = []
        buf_docs: List[str] = []
        buf_meta: List[Dict[str, Any]] = []
        total = 0

        def _flush() -> None:
            nonlocal buf_ids, buf_docs, buf_meta, total
            if not buf_ids:
                return
            collection.upsert(ids=buf_ids, documents=buf_docs, metadatas=buf_meta)
            total += len(buf_ids)
            pbar.update(len(buf_ids))
            buf_ids, buf_docs, buf_meta = [], [], []

        for stream_idx, rec in enumerate(ds_iter):
            text = rec.get(text_col)
            if not isinstance(text, str) or not text.strip():
                continue

            row_label = f"row {stream_idx}"
            hf_row_id = require_hf_field(rec, cfg.metadata_hf_row_id, row_label=row_label)
            chunk_index = require_hf_int_field(rec, cfg.metadata_chunk_index, row_label=row_label)
            doc_id = rec.get(cfg.metadata_doc_id, hf_row_id)

            buf_ids.append(str(hf_row_id))
            buf_docs.append(text)
            buf_meta.append(
                {
                    "hf_row_id": str(hf_row_id),
                    "doc_id": str(doc_id),
                    "chunk_index": chunk_index,
                }
            )

            if len(buf_ids) >= cfg.upsert_batch_size:
                _flush()

        _flush()
        pbar.close()
        print(f"[chroma_preprocess] done — {total} chunks indexed in '{cfg.vectorstore_name}'")
