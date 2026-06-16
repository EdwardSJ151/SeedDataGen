# SeedDataGen

Turns long text from a Hugging Face dataset into filtered multi-turn chats in JSONL. Most steps call a vLLM server; the last step uses sentence-transformers on CPU (unless you point it at CUDA).

## Running

From the repo root (`synthgen/`, the folder that contains `SeedDataGen/`):

Example:
```bash
python -m SeedDataGen.run_pipeline --num-rows 10000 --batch-size 32