#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv-fleming-codex-sdk}"
REQ_FILE="${REQ_FILE:-$ROOT_DIR/requirements-codex-sdk.txt}"

echo "Using python: $PYTHON_BIN"
echo "Creating venv at: $VENV_DIR"
echo "Installing requirements from: $REQ_FILE"

"$PYTHON_BIN" -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "$REQ_FILE"

echo
echo "Activate with:"
echo "  source \"$VENV_DIR/bin/activate\""
echo
echo "Validate with:"
echo "  python \"$ROOT_DIR/scripts/check_fleming_env.py\" --scope codex-sdk"
