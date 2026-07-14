#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-configs/tools_main.env}"

bash "${SCRIPT_DIR}/generate_scope_codes_tools.sh" "${CONFIG}"
bash "${SCRIPT_DIR}/train_tiger_tools.sh" "${CONFIG}"
bash "${SCRIPT_DIR}/evaluate_stability_tools.sh" "${CONFIG}"
