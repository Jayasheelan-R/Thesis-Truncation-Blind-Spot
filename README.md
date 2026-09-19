# Truncation Blind Spot

Single repo for the thesis's model-collapse experiments (forked from
[Drayson et al.'s `model_collapse`](https://github.com/GeorgeDrayson/model_collapse),
EMNLP 2025). Everything needed to run the full sweep lives here -- no other
repo is touched going forward.

## What a run does

For each decoding arm in `config/arms.yaml`:

1. Fine-tune GPT-2 on human WikiText-103 -> generation 0.
2. For 8 iterations: generate text from the current model with that arm's
   decoding strategy, mix it into the training set at a fixed **30%
   synthetic contamination** level, fine-tune again -> next generation.
3. Every generation's metrics (diversity, self-BLEU, readability, rare-token
   / contamination metrics, eval loss/perplexity) are logged to W&B and
   written to disk under the run's output directory.

12 arms total: 3 `min_p` values, 5 `top_k` values, 4 nucleus (`top_p`)
values -- see `config/arms.yaml` for the exact values and to add/change
arms. That file is the **only** place you should need to edit locally
before pushing; everything downstream (the Colab notebook, the runner)
reads it.

### Contamination level

30% synthetic per generation, reached via `human_data_alpha=1.0`,
`ai_beta=0.4286` (`ai_beta / (ai_beta + human_data_alpha) = 0.30` -- see
the corrected mixing-mechanism derivation in `config/arms.yaml`'s
comments). 30% sits between this fork's previously-validated 25% and 50%
ablation points: aggressive enough to show collapse within 8 generations,
not so aggressive that early generations are uninformative.

## Repo layout

- `main.py` -- runs **one** arm: gen-0 training, then the
  generate -> mix -> train loop for `num_iterations`. Resumable: restarting
  against the same `hydra.run.dir` skips whatever's already done.
- `run_all_arms.py` -- runs arms from `config/arms.yaml` back-to-back
  (all of them, or a `--arms` subset), skipping any that are already fully
  complete. This is what the Colab notebook calls.
- `config/arms.yaml` -- the experiment manifest (decoding arms, iteration
  count, contamination level).
- `config/` -- Hydra configs (model, dataset, decoding, detector, training).
  `ours.yaml` is the thesis entry point (`--config-name=ours`).
- `src/` -- training (`train.py`), generation (`generate.py`), data
  loading (`load_data_ours.py`), contamination/diversity metrics
  (`utils/contamination_metrics.py`), decay-curve analysis
  (`analyze_metrics.py`), environment snapshot (`print_env.py`).
  `generate.py` checkpoints every generated batch to a `*.partial.jsonl`
  file next to its output as it goes, so a Colab disconnect mid-generation
  (the slowest step, and the likeliest place to lose a session) only loses
  the in-flight batch, not the whole iteration -- re-running resumes from
  the last flushed batch. `main.py` separately skips re-running `generate.py`
  entirely for an iteration whose `data.json`/`data_metrics.json` already
  exist, going straight to training.
- `colab_run.ipynb` -- clones this repo on Colab, does the one-time data
  load + environment snapshot, and runs either one arm (`ARM_NAME`, the
  default -- for a first full validation pass) or the whole sweep
  (`ARM_NAME = None`) unattended.
- `local_gpu_run.ipynb` -- same as `colab_run.ipynb`, for a machine with
  its own GPU instead of Colab. No `google.colab` APIs; `DRIVE_ROOT` points
  at a locally-synced copy of the same Drive folder (Google Drive for
  Desktop or similar), so runs on this machine and on Colab share the same
  experiment state and resume logic.

## Storage split

- **Google Drive** (`/content/drive/MyDrive/FinalProject`):
  WikiText-103 data (20,000-row cap, already loaded there), model
  checkpoints, generated data, and per-run metrics JSON -- everything
  that needs to survive a Colab session reset.
- **W&B**: every generation's metrics, for cross-arm comparison/plotting
  without re-reading Drive.

## Running on Colab

Open `colab_run.ipynb` in Colab, set `REPO_URL` in cell 2 to this repo's
GitHub URL (the one manual edit point), and run all cells top to bottom.
Re-running the notebook after a disconnect resumes automatically -- no
other changes needed.

## Running on a GPU laptop

Open `local_gpu_run.ipynb`, set `DRIVE_ROOT` in cell 2 to wherever Google
Drive for Desktop (or equivalent) syncs `MyDrive/FinalProject` to on this
machine, and run all cells top to bottom. Make sure `torch` was installed
matching this machine's actual GPU/CUDA driver *before* running the
notebook (`requirements.txt`'s `torch >= 1.3` pin alone doesn't guarantee
a CUDA build -- install it yourself first per pytorch.org if needed).

## Running locally (Mac, smoke test only)

```bash
python -m venv ../.venv-truncation-blind-spot   # outside the repo -- see note below
source ../.venv-truncation-blind-spot/bin/activate
pip install -r requirements.txt
python main.py --config-name=ours smoke_test=true num_iterations=1 decoding=min_p
```

Create the venv **outside** this directory, not nested inside it -- a
recent `nltk` security check blocks imports resolved from inside the
current working directory, which trips over a nested venv's own
`site-packages`.

On Mac (no CUDA), `torch_dtype` auto-falls-back to `float32` (bfloat16
segfaults on MPS); `--device` auto-detects `cuda -> mps -> cpu`.
