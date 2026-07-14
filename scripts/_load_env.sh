#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ $# -ne 1 ]]; then
  echo "usage: <script> <config.env>" >&2
  return 1 2>/dev/null || exit 1
fi

CONFIG="$1"
if [[ "${CONFIG}" != /* ]]; then
  CONFIG="${REPO_ROOT}/${CONFIG}"
fi

if [[ ! -f "${CONFIG}" ]]; then
  echo "config not found: ${CONFIG}" >&2
  return 1 2>/dev/null || exit 1
fi

set -a
source "${CONFIG}"
set +a

export REPO_ROOT

repo_path() {
  if [[ "$1" == /* ]]; then
    printf '%s\n' "$1"
  else
    printf '%s/%s\n' "${REPO_ROOT}" "$1"
  fi
}
