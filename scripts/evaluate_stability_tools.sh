#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-configs/tools_main.env}"
source "${SCRIPT_DIR}/_load_env.sh" "${CONFIG}"
cd "${REPO_ROOT}"

PRIMARY_CODE="$(repo_path "${OUT_DIR}/${DATASET}_${PERIOD_ID}_scope_${PRIMARY_SELECTION}_${PRIMARY_BUDGET_TAG}.npy")"
OLD_CODE_PATH="$(repo_path "${OLD_CODE}")"
FULL_MIGRATION_CODE_PATH="$(repo_path "${FULL_MIGRATION_CODE}")"
TRAIN_PARQUET_PATH="$(repo_path "${TRAIN_PARQUET}")"
VALID_PARQUET_PATH="$(repo_path "${VALID_PARQUET}")"
TEST_PARQUET_PATH="$(repo_path "${TEST_PARQUET}")"
STABILITY_CSV_PATH="$(repo_path "${STABILITY_CSV}")"
STABILITY_JSON_PATH="$(repo_path "${STABILITY_JSON}")"
mkdir -p "$(dirname "${STABILITY_CSV_PATH}")"

python scope/compute_migration_stability.py \
  --old-code "${OLD_CODE_PATH}" \
  --train-parquet "${TRAIN_PARQUET_PATH}" \
  --valid-parquet "${VALID_PARQUET_PATH}" \
  --test-parquet "${TEST_PARQUET_PATH}" \
  --candidate "full_migration" "${FULL_MIGRATION_CODE_PATH}" \
  --candidate "scope_primary" "${PRIMARY_CODE}" \
  --out-csv "${STABILITY_CSV_PATH}" \
  --out-json "${STABILITY_JSON_PATH}"
