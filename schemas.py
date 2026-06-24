"""
Pydantic row schemas for the SeedDataGen pipeline.

These models act as data contracts between phases.  Extra fields are silently
ignored on validation so that downstream phases never break when an upstream
phase adds new fields.

Inheritance hierarchy:
    BaseRow
    ├── QARow              (generator / filter / editor phases)
    └── ConversationRow    (expansion / filter phases)
        └── JudgedConversationRow  (judge + dedup phases)
"""

from typing import Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict


class BaseRow(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int
    origin_id: int
    # sample_id is a single HF row id for legacy/single-chunk rows, or a list of
    # HF row ids for multihop rows (one entry per chunk used).
    sample_id: Union[int, List[int]]
    # sample_text is the flat document string for legacy/single-chunk rows, or a
    # {hf_row_id: chunk_text} or {hf_row_id: {text, document_name}} mapping for multihop rows.
    sample_text: Union[str, Dict[str, Union[str, Dict[str, str]]]]


class QARow(BaseRow):
    question: str
    answer: str
    input_id: Optional[int] = None

    # Provenance fields (set by multihop generators; None on legacy rows).
    GEN_TYPE: Optional[str] = None
    num_chunks: Optional[int] = None
    doc_constraint: Optional[str] = None

    # Similarity-job fields (set only by qa_similarity_multihop).
    similarity_job_index: Optional[int] = None
    similarity_mode: Optional[str] = None
    similarity_threshold: Optional[float] = None
    similarity_min: Optional[float] = None
    similarity_max: Optional[float] = None
    min_matching_words: Optional[int] = None
    chunk_group_similarity: Optional[float] = None


class StyledQARow(QARow):
    """QARow produced by qa_gen_var — carries the style that generated the question."""
    question_style: str


class ConversationRow(BaseRow):
    messages: List[Dict[str, str]]
    input_id: Optional[int] = None


class JudgedConversationRow(ConversationRow):
    scores: Dict[str, float]
    avg_score: float
