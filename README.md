# Stability-Constrained Evolution for Continual Generative Recommendation

This repository provides the main experimental code for **Stability-Constrained Evolution for Continual Generative Recommendation**

---

## Overview

Generative recommendation systems typically use a two-stage pipeline: a tokenizer maps each item to a discrete semantic identifier, and an autoregressive recommender predicts the identifier of the next item. In continual scenarios, new interactions and new items require the tokenizer to adapt, but changing too many existing identifiers can disrupt the recommender's learned token space.

The released code implements a teacher-ranked constrained migration pipeline. Given the previous identifiers and a fully updated identifier set, the selector estimates the ranking utility of candidate item migrations with a trained generative recommender, then applies support-aware constraints to keep high-support and head items stable. The selected identifiers are used by a TIGER-style generative recommender with trie-constrained decoding.

---

## Method

### Stage 1: Tokenizer Update

The `rqvae/` module contains the tokenizer backbone used to produce semantic identifiers. It supports warm-starting from a previous tokenizer checkpoint and exporting item-level code files in `.npy` format.

### Stage 2: Rank-Aligned Migration Selection

`scope/rank_aligned_migration.py` compares the previous code file with a fully updated code file. It uses a teacher recommender checkpoint to evaluate the rank contribution of changed items on a validation split, combines this signal with item support and optional drift magnitude, and writes constrained code files for one or more migration budgets.

### Stage 3: Downstream Generative Recommendation

`TIGER-backbone/main_trie.py` trains and evaluates a T5-based generative recommender. During inference, a trie built from valid item identifiers restricts beam search to legal item codes.

---

## Repository Layout

```text
scope/
  rank_aligned_migration.py          # teacher-ranked migration selector
  compute_migration_stability.py     # item-, token-, and sequence-level churn
TIGER-backbone/
  main_trie.py                       # TIGER training and constrained decoding
  dataset.py, dataloader.py
rqvae/
  main.py, trainer.py, models/       # tokenizer training and code extraction
scripts/
  run_main_pipeline_tools.sh         # end-to-end main experiment
  generate_scope_codes_tools.sh      # migration selection
  train_tiger_tools.sh               # downstream recommender training
  evaluate_stability_tools.sh        # migration stability diagnostics
configs/
  tools_main.env.example             # editable experiment configuration
data/
```

---

## Getting Started

### Requirements

```bash
pip install -r requirements.txt
```

The code is path-configurable and does not assume a specific server, conda environment, or absolute project location.

### Prepare Configuration

Copy the example configuration and edit paths if your files use different names or live outside the repository.

```bash
cp configs/tools_main.env.example configs/tools_main.env
```

The main experiment expects processed sequential recommendation splits, previous-period item codes, fully updated item codes, and two recommender checkpoints:

```text
data/Tools/train_0.7.parquet
data/Tools/valid_0.7.parquet
data/Tools/test_0.7.parquet
data/Tools/Tools_0.6_cf.npy
data/Tools/Tools_0.7_full.npy
TIGER-backbone/ckpt/tiger_Tools_0.6_cf.pth
TIGER-backbone/ckpt/tiger_Tools_0.7_full_teacher.pth
```

---

## Main Experiment

Run the complete pipeline from any working directory:

```bash
bash /path/to/repo/scripts/run_main_pipeline_tools.sh /path/to/repo/configs/tools_main.env
```

The script performs three steps:

1. Select constrained semantic-ID migrations.
2. Train the TIGER recommender with the selected identifiers.
3. Compute migration stability diagnostics.

---

## Individual Commands

**Generate constrained codes**

```bash
bash scripts/generate_scope_codes_tools.sh configs/tools_main.env
```

**Train downstream recommender**

```bash
bash scripts/train_tiger_tools.sh configs/tools_main.env
```

**Evaluate migration stability**

```bash
bash scripts/evaluate_stability_tools.sh configs/tools_main.env
```

---

## Outputs

By default, selected code files and stability summaries are written to `results/`, while downstream recommender checkpoints and logs are written under `TIGER-backbone/ckpt/` and `TIGER-backbone/logs/`. These generated files are ignored by `.gitignore`.
