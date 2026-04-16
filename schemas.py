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

from typing import Dict, List, Optional
from pydantic import BaseModel, ConfigDict


class BaseRow(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int
    origin_id: int
    sample_id: int
    sample_text: str


class QARow(BaseRow):
    question: str
    answer: str
    input_id: Optional[int] = None


class StyledQARow(QARow):
    """QARow produced by qa_gen_var — carries the style that generated the question."""
    question_style: str


class ConversationRow(BaseRow):
    messages: List[Dict[str, str]]
    input_id: Optional[int] = None


class JudgedConversationRow(ConversationRow):
    scores: Dict[str, float]
    avg_score: float
