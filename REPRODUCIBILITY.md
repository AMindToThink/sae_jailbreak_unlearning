# Reproducibility — sae_jailbreak_unlearning

Investigating how well intervening on Sparse Autoencoder internals prevents adversaries
from accessing dangerous knowledge (WMDP).

## Environment
- Conda environment from **`environment.yml`**:
  ```bash
  conda env create -f environment.yml
  ```
  (`setup.py` and `req.txt` are also present.)

## Models (HuggingFace)
- `google/gemma-2-2b`, `google/gemma-2-2b-it`, `gpt2` via `from_pretrained(...)`.
  Gemma is **gated** — run `hf auth login` after accepting the license.

## Data — must be downloaded (NOT stored in the repo)
WMDP corpora / MCQs from the CAIS S3 bucket (referenced in the code, e.g.
`src/scripts/wmdp/data/get_cyber_corpus.py`):
- `https://cais-wmdp.s3.us-west-1.amazonaws.com/wmdp-corpora.zip`
- `https://cais-wmdp.s3.us-west-1.amazonaws.com/wmdp-mcqs.zip`
- `https://cais-wmdp.s3.us-west-1.amazonaws.com/mmlu-auxiliary-corpora.zip`
- ARENA 3.0 utilities: `https://github.com/callummcdougall/ARENA_3.0/archive/refs/heads/main.zip`
- The **bio-forget corpus is gated** (request access via the WMDP/CAIS form); it is not
  redistributable and was ~713 MB on the source machine.

Unzip into `data/` following the layout the `src/scripts/` code expects.

## Layout
- `src/scripts/` — WMDP data, unlearning-with-SAEs, llm-auditing, evaluation
  (an `lm-evaluation-harness` submodule).
- `results/`, `models/`, `experiments.csv`.

---
> Reproducibility note: added 2026-07-02 during a machine migration. Uncommitted
> migration WIP for this repo was also pushed to `backup/migration-*` branches on origin.
