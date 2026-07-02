# Reproducing this repo

This repo has no access to the original `cs29824` machine assumed. Every command below is
written to work from a fresh clone. Where something could not be verified from the code,
it is marked `TODO (Matthew)` instead of guessed.

## 1. Environment

Two environment specs exist and disagree on Python version — pick one, don't mix:

- **`req.txt`** (tracked on `main`, this is the committed source of truth) — a
  `conda list --export`-style file pinned to **Python 3.11.11**:
  ```bash
  conda create --name sae_jailbreak_unlearning --file req.txt
  conda activate sae_jailbreak_unlearning
  ```
- **`environment.yml`** — present in this working copy (and on the `backup/migration-*`
  branches on origin) but **not committed to `main`**, pinned to **Python 3.9.12**:
  ```bash
  conda env create -f environment.yml
  ```
  Its `prefix:` line (`/home/cs29824/miniconda3`) is machine-specific — delete it or let
  `conda` ignore it. `TODO (Matthew)`: confirm which of `req.txt` / `environment.yml` was
  actually used to produce the results in `results/` — they list different package sets
  (e.g. `req.txt` includes `automated-interpretability`, `blobfile`, `boostedblob`, which
  `environment.yml` lacks).

Then install this repo's own package (declares no dependencies of its own — see `setup.py`):
```bash
pip install -e .
```

Submodules (see `.gitmodules`, section 4) must be checked out before anything imports them:
```bash
git submodule update --init --recursive
```

**`src/scripts/unlearning_with_saes/` has its own, separate dependency set** (Poetry,
`pyproject.toml` + `poetry.lock`, requires `python>=3.10,<3.12` — incompatible with the
3.9.12 `environment.yml` above, compatible with `req.txt`'s 3.11.11). It pulls
`sae-lens` straight from `git+https://github.com/jbloomAus/SAELens.git` and
`lm-eval` from the local path `../evaluation/lm-evaluation-harness`. To work on SAE
training / unlearning specifically:
```bash
cd src/scripts/unlearning_with_saes
poetry install
```
`TODO (Matthew)`: confirm whether this Poetry env is meant to be used standalone or
whether its deps should be merged into the root conda env.

## 2. Data

- **WMDP multiple-choice eval sets**: `datasets.load_dataset("cais/wmdp", "wmdp-bio")`
  (also `"wmdp-cyber"`, `"wmdp-chem"`) — public on the Hub, no auth needed.
- **MMLU**: `datasets.load_dataset("cais/mmlu", <subject>)`, e.g. `"college_biology"`.
- **Retain corpus for RMU**: `datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="test")`.
- **Cyber forget corpus**: public — `datasets.load_dataset("cais/wmdp-corpora", "cyber-forget-corpus")`
  (see `src/scripts/wmdp/data/get_cyber_corpus.py`), then write to
  `src/scripts/wmdp/data/cyber-forget-corpus.jsonl`.
- **Bio forget corpus — GATED, cannot be redistributed.** Request access via the CAIS
  Google Form linked from `src/scripts/wmdp/README.md`:
  `https://docs.google.com/forms/d/e/1FAIpQLSdnQc8Qn0ozSDu3VE8HLoHPvhpukX1t1dIwE5K5rJw9lnOjKw/viewform`.
  Once granted, place the file at `src/scripts/wmdp/data/bio-forget-corpus.jsonl` (the
  RMU code in `src/scripts/wmdp/rmu/utils.py::get_data()` reads `data/<corpus-name>.jsonl`
  relative to the process's working directory, so run unlearning from inside
  `src/scripts/wmdp/`).
- **WMDP corpora / MCQ mirrors** (alternative to the Hub, from `src/scripts/wmdp/README.md`,
  vendored from `centerforaisafety/wmdp`):
  - `https://cais-wmdp.s3.us-west-1.amazonaws.com/wmdp-corpora.zip` (password `wmdpcorpora`)
  - `https://cais-wmdp.s3.us-west-1.amazonaws.com/wmdp-mcqs.zip` (password `wmdpmcqs`)
  - `https://cais-wmdp.s3.us-west-1.amazonaws.com/mmlu-auxiliary-corpora.zip` (password `wmdpauxiliarycorpora`)
- **Pre-built combined eval set**: `data/mmlu_wmdp_bio_combined/` (HF `datasets` arrow
  format, `validation`/`test` splits) already exists in this repo checkout. The closest
  matching build script is `src/scripts/evaluation/make_mmluVal_wmdpTest.py`, which
  combines `cais/mmlu` + `cais/wmdp` subjects and `push_to_hub`s the result — it does not
  exactly reproduce this on-disk directory's name/layout. `TODO (Matthew)`: the exact
  script/notebook that produced `data/mmlu_wmdp_bio_combined/` was not found by search.
- SAE checkpoints for `unlearning_with_saes` are pulled at runtime from the HF Hub repo
  **`eoinf/unlearning_saes`** via `huggingface_hub.hf_hub_download` (see
  `src/scripts/unlearning_with_saes/unlearning/var.py::SAE_MAPPING`, `REPO_ID`) — no
  manual download needed, just HF auth (section 4).

## 3. Running

All commands assume the conda/poetry env from section 1 is active and submodules are
initialized. Paths are relative to the repo root unless noted.

**RMU unlearning sweep on Gemma-2-2B** (trains many models over a layer × steering-coeff
× retain-alpha grid, then evaluates each with `lm-eval`):
```bash
cd src/scripts/wmdp
python gemma_rmu_sweep.py \
  --model_name google/gemma-2-2b \
  --output_folder <output_dir> \
  --forget_corpora bio-forget-corpus \
  --device 0
```
(`--forget_corpora` must be `bio-forget-corpus` or `cyber-forget-corpus`; `--hf_user`/
`--hf_key` are optional, to push trained models to the HF Hub.)

**Single RMU unlearning run** (what the sweep calls internally):
```bash
cd src/scripts/wmdp
python -m rmu.unlearn \
  --model_name_or_path google/gemma-2-2b-it \
  --forget_corpora bio-forget-corpus,cyber-forget-corpus \
  --retain_corpora wikitext,wikitext \
  --output_dir <output_dir>
```

**Evaluate an existing (unlearned) model on WMDP/MMLU** via the `lm-evaluation-harness`
fork (needs `pip install -e src/scripts/evaluation/lm-evaluation-harness` first):
```bash
lm-eval --model hf \
  --model_args pretrained=<model_path_or_hf_id> \
  --tasks wmdp_bio,mmlu_college_biology,mmlu_high_school_us_history,mmlu_high_school_geography,mmlu_human_aging \
  --batch_size auto:3 \
  --output_path <output_dir>
```
Or the batch driver `src/scripts/evaluation/eval_all_rmu_gemma_2_2b.py --model_name
google/gemma-2-2b --output_folder <dir> --device 0` — **note**: this script references
`args.hf_user` on line ~53 but the `--hf_user` CLI argument is commented out (dead code /
bug as committed); it will raise `AttributeError` unless patched.

**Adversarial jailbreak attacks (GCG / DSS) on an (unlearned) model** (`llm-auditing/`):
```bash
cd src/scripts/llm-auditing
python run_attack.py \
  --model_path <path/to/model> \
  --config_path configs/attack_config.json \
  --hf_token <token if model is gated> \
  -v
```
Attack type is chosen with `-a {greedy,causal,greedyc}`; see `configs/*.json` for the
question sets these attacks are run against (WMDP-derived).

**SAE training / unlearning-with-SAEs experiments** live under
`src/scripts/unlearning_with_saes/` (own Poetry env, section 1) — notebooks in
`notebooks1/`, `notebooks2/`, core logic in `unlearning/` and `sae/`.
`TODO (Matthew)`: no single top-level "run everything" script was found for this
subproject; entry points are the individual scripts/notebooks in that tree.

**Interactive steering exploration**: `tests.ipynb` (repo root) and
`src/notebooks/gcg_against_unlearning.ipynb`.

## 4. External dependencies

- **Gated HF models**: `google/gemma-2-2b`, `google/gemma-2-2b-it` (and `gemma-2-9b-it`
  is referenced in places). Accept the license at
  `https://huggingface.co/google/gemma-2-2b` on the account you'll authenticate with,
  then authenticate (`huggingface-cli login` or `HF_TOKEN` env var) before running
  anything that calls `AutoModelForCausalLM.from_pretrained("google/gemma-2-2b...")`.
- **Submodule forks** (from `.gitmodules`):
  - `src/scripts/evaluation/lm-evaluation-harness` → `https://github.com/AMindToThink/lm-evaluation-harness.git`, declared branch **`sae_steered`**. **Caveat**: the commit actually checked out in this working copy (`7d6cca28`) is on a different branch, `about_to_ditch_HFLM`, not `sae_steered` — the two may have diverged. For an exact match to results in `results/`, check out commit `7d6cca282769ab3a75230bee144463c6c3d4756a` explicitly.
  - `src/scripts/wmdp` → `.gitmodules` declares `https://github.com/centerforaisafety/wmdp.git`, **but** the submodule's `git remote` in this checkout is actually `https://github.com/AMindToThink/wmdp.git` (Matthew's fork, which adds `gemma_rmu_sweep.py` etc. on top of upstream CAIS code) at commit `b5a740e03fedd75d25ab8ec159188ba5afa5e379` (branch `main`). Use the fork, not upstream CAIS, to get the scripts referenced in section 3.
  - `src/scripts/evaluation/MMLU/test` → `https://github.com/hendrycks/test.git` (not yet initialized in this checkout).
  - `.gitmodules` also has a stray/malformed 5th entry named after an absolute path (`/home/cs29824/matthew/sae_jailbreak_unlearning/src/scripts/evaluation/lm-evaluation-harness`, `branch = main`, no `url`/`path`) and a duplicate `src/scripts/evaluation/matthew/lm-evaluation-harness` entry whose directory doesn't exist on disk — `git submodule update --init --recursive` may warn or no-op on these; `TODO (Matthew)`: clean up `.gitmodules`.
- **Weights & Biases**: used for SAE training logging (`src/scripts/unlearning_with_saes/sae/train.py`, `wandb.init(project=self.cfg.wandb_project, ...)`). Run `wandb login` or set `WANDB_API_KEY` before training SAEs; the specific `wandb_project` name is a config field, not hardcoded — `TODO (Matthew)`: state which W&B project/entity these results were logged to.
- **Environment variables referenced in code**:
  - `HF_TOKEN` — HF Hub auth (also passed explicitly as `--hf_token`/`--hf_key` CLI args in some scripts).
  - `CUDA_VISIBLE_DEVICES` — set by scripts themselves from a `--device` CLI flag.
  - `OPENAI_API_KEY` — used by `src/scripts/llm-auditing/gpt4-data.py`, which currently **hardcodes a placeholder** (`os.environ["OPENAI_API_KEY"] = 'key'`) — replace with a real key or export the env var and remove that line's effect before running.
  - Optional, only if you touch the underlying `lm-evaluation-harness`'s non-HF model backends or optional tasks: `ANTHROPIC_API_KEY`, `ZENO_API_KEY`, `PERSPECTIVE_API_KEY`, `WATSONX_API_KEY`/`WATSONX_URL`/`WATSONX_PROJECT_ID`, `TEXTSYNTH_API_SECRET_KEY`.

## 5. Hardware

- An NVIDIA GPU with CUDA support is required — `torch==2.1.2` plus the pinned
  `nvidia-*-cu12` wheels in `req.txt`/`environment.yml` imply a CUDA 12.1-compatible
  driver.
- Scripts select a single GPU via `--device` → `CUDA_VISIBLE_DEVICES` (e.g.
  `gemma_rmu_sweep.py`, `eval_all_rmu_gemma_2_2b.py`); model loading uses
  `device_map="auto"` (`src/scripts/wmdp/rmu/utils.py`), so it can shard across multiple
  visible GPUs if more than one is exposed. RMU training additionally caps usage at 90%
  of the GPU via `torch.cuda.set_per_process_memory_fraction(0.9)`.
- `TODO (Matthew)`: no specific GPU model or VRAM figure (e.g. A100/40GB) is stated
  anywhere in the code, configs, or existing docs — please fill in what was actually used
  to produce `results/`.

---
*This document was generated by inspecting the code, `.gitmodules`, `environment.yml`,
`req.txt`, `setup.py`, and submodule state as of this repo's `main` branch; anything not
directly verifiable in the repo is marked `TODO (Matthew)` rather than guessed. A prior,
shorter `REPRODUCIBILITY.md` (committed separately, also present in this repo) covers
similar ground with less detail and was not merged into this file.*
