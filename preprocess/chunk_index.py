"""
ChromaDB access helpers shared by the preprocess phase and the multihop
generators.

A single persistent collection holds one entry per document chunk:
  - id:        str(hf_row_id)
  - document:  chunk text
  - embedding: computed by the configured sentence-transformers model
  - metadata:  {hf_row_id, doc_id, chunk_index}

The collection name and persist directory come from the CHROMA_* config block.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import chromadb


def _client(persist_dir: str):
    import chromadb

    os.makedirs(persist_dir, exist_ok=True)
    return chromadb.PersistentClient(path=persist_dir)


def _embedding_function(embed_model: str):
    """Return a Chroma embedding function backed by sentence-transformers."""
    from chromadb.utils import embedding_functions

    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=embed_model
    )


def collection_exists(vectorstore_name: str, persist_dir: str) -> bool:
    """True if a non-empty Chroma collection *vectorstore_name* already exists."""
    try:
        client = _client(persist_dir)
    except Exception:
        return False
    try:
        existing = {c.name for c in client.list_collections()}
    except Exception:
        return False
    if vectorstore_name not in existing:
        return False
    try:
        coll = client.get_collection(vectorstore_name)
        return coll.count() > 0
    except Exception:
        return False


def get_collection(
    vectorstore_name: str,
    persist_dir: str,
    embed_model: str,
):
    """
    Return (creating if necessary) the persistent Chroma collection bound to the
    configured embedding model.  Uses cosine space so similarity scores are
    directly comparable to the thresholds configured for similarity jobs.
    """
    client = _client(persist_dir)
    return client.get_or_create_collection(
        name=vectorstore_name,
        embedding_function=_embedding_function(embed_model),
        metadata={"hnsw:space": "cosine"},
    )


def get_collection_from_env():
    """
    Open the Chroma collection using the CHROMA_* environment configuration
    (the same keys the chroma_preprocess phase reads).  Used by the multihop
    generators so they bind to the collection populated during preprocess.
    """
    vectorstore_name = os.environ.get("CHROMA_VECTORSTORE_NAME", "seeddatagen-chunks")
    persist_dir = os.environ.get("CHROMA_PERSIST_DIR", ".cache/chroma")
    embed_model = os.environ.get(
        "CHROMA_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    return get_collection(vectorstore_name, persist_dir, embed_model)


def delete_collection(vectorstore_name: str, persist_dir: str) -> None:
    """Drop the collection if it exists (used by FORCE_REBUILD)."""
    try:
        client = _client(persist_dir)
        client.delete_collection(vectorstore_name)
    except Exception:
        pass
