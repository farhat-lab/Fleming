#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODULE_NAME="${MODULE_NAME:-conda/miniforge3/24.11.3-0}"
ENV_PREFIX="${ENV_PREFIX:-$ROOT_DIR/.conda-env/fleming-codex-sdk}"
REQ_FILE="${REQ_FILE:-$ROOT_DIR/requirements-codex-sdk.txt}"

module load "$MODULE_NAME"

echo "Using module: $MODULE_NAME"
echo "Creating env at: $ENV_PREFIX"
echo "Installing requirements from: $REQ_FILE"

mkdir -p "$(dirname "$ENV_PREFIX")"

if [[ ! -d "$ENV_PREFIX" ]]; then
  conda create -y --prefix "$ENV_PREFIX" python=3.11 pip
fi

eval "$(conda shell.bash hook)"
conda activate "$ENV_PREFIX"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "$REQ_FILE"

echo
echo "Activate with:"
echo "  module load $MODULE_NAME"
echo "  conda activate \"$ENV_PREFIX\""
echo
echo "Validate with:"
echo "  python \"$ROOT_DIR/scripts/check_fleming_env.py\" --scope codex-sdk"
