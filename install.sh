#!/usr/bin/env bash
set -euo pipefail

if ! command -v uv &>/dev/null; then
    echo "Error: uv is not installed."
    echo "Install it from: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

uv python install 3.14t

deactivate 2>/dev/null || true
uv venv --python 3.14t --clear
grep -q 'PYTHON_GIL=0' .venv/bin/activate || echo 'export PYTHON_GIL=0' >> .venv/bin/activate
source .venv/bin/activate

uv pip install -e ".[dev]" --upgrade

echo ""
echo "Virtual environment created: .venv"
echo "To activate it run: source .venv/bin/activate"
