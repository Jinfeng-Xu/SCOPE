#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-configs/tools_main.env}"
source "${SCRIPT_DIR}/_load_env.sh" "${CONFIG}"
cd "${REPO_ROOT}"

METHOD="scope_${PRIMARY_SELECTION}_${PRIMARY_BUDGET_TAG}"
CODE_PATH="$(repo_path "${OUT_DIR}/${DATASET}_${PERIOD_ID}_scope_${PRIMARY_SELECTION}_${PRIMARY_BUDGET_TAG}.npy")"
PREV_TIGER_CKPT_PATH="$(repo_path "${PREV_TIGER_CKPT}")"
DATASET_PATH_RESOLVED="$(repo_path "${DATASET_PATH}")"
LOG_PATH="$(repo_path "TIGER-backbone/logs/${METHOD}_${DATASET}_${PERIOD_ID}.log")"
SAVE_PATH="$(repo_path "TIGER-backbone/ckpt/tiger_${DATASET}_${PERIOD_ID}_${METHOD}.pth")"

mkdir -p TIGER-backbone/logs TIGER-backbone/ckpt

cd TIGER-backbone
python main_trie.py \
  --ckpt_path "${PREV_TIGER_CKPT_PATH}" \
  --code_path "${CODE_PATH}" \
  --log_path "${LOG_PATH}" \
  --save_path "${SAVE_PATH}" \
  --dataset "${DATASET}" \
  --dataset_path "${DATASET_PATH_RESOLVED}" \
  --data "${PERIOD}" \
  --seed "${SEED}" \
  --num_epochs "${NUM_EPOCHS}" \
  --early_stop "${EARLY_STOP}" \
  --batch_size "${TRAIN_BATCH_SIZE}" \
  --infer_size "${INFER_BATCH_SIZE}" \
  --beam_size "${BEAM_SIZE}" \
  --desc "${METHOD}"
