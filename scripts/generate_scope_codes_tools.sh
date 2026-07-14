#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-configs/tools_main.env}"
source "${SCRIPT_DIR}/_load_env.sh" "${CONFIG}"
cd "${REPO_ROOT}"

mkdir -p "${OUT_DIR}"

OLD_CODE_PATH="$(repo_path "${OLD_CODE}")"
FULL_MIGRATION_CODE_PATH="$(repo_path "${FULL_MIGRATION_CODE}")"
TRAIN_PARQUET_PATH="$(repo_path "${TRAIN_PARQUET}")"
VALID_PARQUET_PATH="$(repo_path "${VALID_PARQUET}")"
TEACHER_TIGER_CKPT_PATH="$(repo_path "${TEACHER_TIGER_CKPT}")"
OUT_DIR_PATH="$(repo_path "${OUT_DIR}")"

IFS=',' read -ra SELS <<< "${SELECTIONS}"
for selection in "${SELS[@]}"; do
  python scope/rank_aligned_migration.py \
    --old-code "${OLD_CODE_PATH}" \
    --new-code "${FULL_MIGRATION_CODE_PATH}" \
    --train-parquet "${TRAIN_PARQUET_PATH}" \
    --valid-parquet "${VALID_PARQUET_PATH}" \
    --teacher-ckpt "${TEACHER_TIGER_CKPT_PATH}" \
    --out-dir "${OUT_DIR_PATH}" \
    --prefix "${DATASET}_${PERIOD_ID}_scope" \
    --budgets "${BUDGETS}" \
    --selection "${selection}" \
    --batch-size "${BATCH_SIZE}" \
    --beam-size "${BEAM_SIZE}"
done
