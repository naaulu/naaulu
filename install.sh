#!/usr/bin/env bash
set -euo pipefail

if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.local/bin/env"
fi

uv python install 3.14t
rm -rf .venv
uv venv --python 3.14t
echo 'export PYTHON_GIL=0' >> .venv/bin/activate
source .venv/bin/activate

uv pip install -e ".[dev]" --upgrade

echo ""
echo "Installing reference data (networks.json, radars.json)..."
mkdir -p "$HOME/.local/share/naaulu"
cp -f naaulu/data/*.json "$HOME/.local/share/naaulu/" || true

echo ""
echo "To activate the environment in future sessions run:"
echo "  source .venv/bin/activate"
echo "To exit it run: deactivate"
