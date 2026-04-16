"""
Global infrastructure configuration for SeedDataGen.

Only settings that are shared across multiple phases stay here. The rest are in each phase.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

_cfg_dir = Path(__file__).resolve().parent
_repo_root = _cfg_dir.parent
load_dotenv(_repo_root / ".env")
load_dotenv(_cfg_dir / ".env", override=True)

# vLLM server
VLLM_BASE_URL: str = os.environ.get("VLLM_BASE_URL", "http://localhost:8020/v1")
VLLM_API_KEY: str = os.environ.get("VLLM_API_KEY", "no-key-needed")
STOP_STRINGS: list[str] = ["<|im_end|>", "<|end_of_text|>"]

# Source dataset
DATASET_ID: str = os.environ.get("DATASET_ID", "cemig-ceia/sites_educacionais")
DATASET_SUBSET: str = os.environ.get("DATASET_SUBSET", "default")
DATASET_SPLIT: str = os.environ.get("DATASET_SPLIT", "brasil_escola")
DATASET_TEXT_FIELD: str = os.environ.get("DATASET_TEXT_FIELD", "text")
DATASET_MAX_CHARS: int = int(os.environ.get("DATASET_MAX_CHARS", "120000"))
DATASET_MIN_CHARS: int = int(os.environ.get("DATASET_MIN_CHARS", "800"))

# Execution control (global defaults; phases may override via their own config)
NUM_ROWS: int = int(os.environ.get("NUM_ROWS", "100"))
BATCH_SIZE: int = int(os.environ.get("BATCH_SIZE", "32"))
MAX_CONCURRENT: int = int(os.environ.get("MAX_CONCURRENT", "64"))
