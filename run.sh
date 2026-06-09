#!/usr/bin/env bash
set -euo pipefail

# Move to project root (where this script is located)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade packaging tools
pip install --upgrade pip setuptools wheel

# Install project in editable mode (always safe in dev)
pip install -e .

# Run the application
python main.py "$@"