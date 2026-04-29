#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODULE_NAME="${MODULE_NAME:-conda/miniforge3/24.11.3-0}"
ENV_PREFIX="${ENV_PREFIX:-$ROOT_DIR/.conda-env/fleming-codex-sdk}"
ENV_FILE="${ENV_FILE:-$ROOT_DIR/environment.conda.yml}"
PKGS_DIR="${PKGS_DIR:-$ROOT_DIR/.conda-pkgs}"

module load "$MODULE_NAME"

echo "Using module: $MODULE_NAME"
echo "Using env file: $ENV_FILE"
echo "Creating env at: $ENV_PREFIX"
echo "Using package cache: $PKGS_DIR"

mkdir -p "$(dirname "$ENV_PREFIX")"
mkdir -p "$PKGS_DIR"
export CONDA_PKGS_DIRS="$PKGS_DIR"

conda env create --prefix "$ENV_PREFIX" --file "$ENV_FILE" || \
conda env update --prefix "$ENV_PREFIX" --file "$ENV_FILE" --prune

echo
echo "Activate with:"
echo "  module load $MODULE_NAME"
echo "  conda activate \"$ENV_PREFIX\""
echo
echo "Validate with:"
echo "  python \"$ROOT_DIR/scripts/check_fleming_env.py\" --scope codex-sdk"
