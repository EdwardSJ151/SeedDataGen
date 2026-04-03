"""
Shared configuration for the SeedDataGen pipeline.

All env-overridable settings, output paths, and generation parameters.
"""

import os

# -----------------------------------------------------------------------
# vLLM SERVER (same model for generation + LLM judge)
# -----------------------------------------------------------------------
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8020/v1")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "no-key-needed")

# -----------------------------------------------------------------------
# EMBEDDING MODEL (phase 6 only — different model)
# -----------------------------------------------------------------------
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_DEVICE = os.environ.get("EMBED_DEVICE", "cpu")  # "cpu" or "cuda"
EMBED_SIMILARITY_THRESHOLD = float(os.environ.get("EMBED_SIMILARITY_THRESHOLD", "0.95"))

# -----------------------------------------------------------------------
# SOURCE DATASET
# -----------------------------------------------------------------------
DATASET_ID = os.environ.get("DATASET_ID", "cemig-ceia/sites_educacionais")
DATASET_SUBSET = os.environ.get("DATASET_SUBSET", "default")
DATASET_SPLIT = os.environ.get("DATASET_SPLIT", "brasil_escola")
DATASET_TEXT_FIELD = os.environ.get("DATASET_TEXT_FIELD", "text")
DATASET_MAX_CHARS = int(os.environ.get("DATASET_MAX_CHARS", "120000"))
DATASET_MIN_CHARS = int(os.environ.get("DATASET_MIN_CHARS", "800"))

# -----------------------------------------------------------------------
# GENERATION PARAMETERS
# -----------------------------------------------------------------------
# Phase 1: QA generation
QA_TEMPERATURE = float(os.environ.get("QA_TEMPERATURE", "0.7"))
QA_TOP_P = float(os.environ.get("QA_TOP_P", "0.9"))
QA_MAX_TOKENS = int(os.environ.get("QA_MAX_TOKENS", "2048"))

# Phase 3: Conversation expansion (user turn generation)
USER_TURN_TEMPERATURE = float(os.environ.get("USER_TURN_TEMPERATURE", "0.9"))
USER_TURN_TOP_P = float(os.environ.get("USER_TURN_TOP_P", "0.95"))
USER_TURN_MAX_TOKENS = int(os.environ.get("USER_TURN_MAX_TOKENS", "512"))

# Phase 3: Conversation expansion (assistant turn generation)
ASSISTANT_TURN_TEMPERATURE = float(os.environ.get("ASSISTANT_TURN_TEMPERATURE", "0.7"))
ASSISTANT_TURN_TOP_P = float(os.environ.get("ASSISTANT_TURN_TOP_P", "0.9"))
ASSISTANT_TURN_MAX_TOKENS = int(os.environ.get("ASSISTANT_TURN_MAX_TOKENS", "2048"))

# Phase 3: number of extra user turns to add (drawn uniformly from this range)
N_USER_TURNS_MIN = int(os.environ.get("N_USER_TURNS_MIN", "3"))
N_USER_TURNS_MAX = int(os.environ.get("N_USER_TURNS_MAX", "3"))

# Phase 5: LLM judge
JUDGE_TEMPERATURE = float(os.environ.get("JUDGE_TEMPERATURE", "0.0"))
JUDGE_TOP_P = float(os.environ.get("JUDGE_TOP_P", "0.9"))
JUDGE_MAX_TOKENS = int(os.environ.get("JUDGE_MAX_TOKENS", "2048"))
JUDGE_MIN_AVG_SCORE = float(os.environ.get("JUDGE_MIN_AVG_SCORE", "4.0"))

# -----------------------------------------------------------------------
# HEURISTIC THRESHOLDS
# -----------------------------------------------------------------------
QA_MIN_ANSWER_LEN = int(os.environ.get("QA_MIN_ANSWER_LEN", "10"))
QA_LEVENSHTEIN_THRESHOLD = int(os.environ.get("QA_LEVENSHTEIN_THRESHOLD", "20"))
CONV_MIN_MESSAGES = int(os.environ.get("CONV_MIN_MESSAGES", "4"))
CONV_ASSISTANT_MIN_LEN = int(os.environ.get("CONV_ASSISTANT_MIN_LEN", "10"))
CONV_USER_LEVENSHTEIN_THRESHOLD = int(os.environ.get("CONV_USER_LEVENSHTEIN_THRESHOLD", "20"))
CONV_ADJACENT_LEVENSHTEIN_THRESHOLD = int(os.environ.get("CONV_ADJACENT_LEVENSHTEIN_THRESHOLD", "20"))

# -----------------------------------------------------------------------
# EXECUTION CONTROL
# -----------------------------------------------------------------------
NUM_ROWS = int(os.environ.get("NUM_ROWS", "100"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))
MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT", "64"))

# -----------------------------------------------------------------------
# OUTPUT FILES
# -----------------------------------------------------------------------
PHASE1_OUTPUT = os.environ.get("PHASE1_OUTPUT", "seed_phase1_qa.jsonl")
PHASE2_OUTPUT = os.environ.get("PHASE2_OUTPUT", "seed_phase2_qa_filtered.jsonl")
PHASE3_OUTPUT = os.environ.get("PHASE3_OUTPUT", "seed_phase3_conversations.jsonl")
PHASE4_OUTPUT = os.environ.get("PHASE4_OUTPUT", "seed_phase4_conv_filtered.jsonl")
PHASE5_OUTPUT = os.environ.get("PHASE5_OUTPUT", "seed_phase5_judged.jsonl")
PHASE6_OUTPUT = os.environ.get("PHASE6_OUTPUT", "seed_phase6_final.jsonl")

# -----------------------------------------------------------------------
# vLLM EXTRAS
# -----------------------------------------------------------------------
STOP_STRINGS = ["<|im_end|>", "<|end_of_text|>"]
