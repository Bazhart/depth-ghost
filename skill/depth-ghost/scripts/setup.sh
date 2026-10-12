#!/usr/bin/env bash
# One-time setup for the depth-ghost skill. Creates a private Python env at
# ~/.depth-ghost/.venv with torch + the depth model's dependencies (~1-2 GB).
# Safe to re-run: exits immediately if everything is already installed.
set -euo pipefail

VENV="$HOME/.depth-ghost/.venv"
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$VENV/bin/python"

if [ -x "$PY" ] && "$PY" -c "import torch, transformers, cv2, pillow_heif" 2>/dev/null; then
  echo "depth-ghost: ready ($PY)"
  exit 0
fi

mkdir -p "$HOME/.depth-ghost"

# torch needs Python 3.10-3.12 (newer Pythons often have no torch wheels yet,
# and Intel Macs top out at torch 2.2 / Python 3.12).
if command -v uv >/dev/null 2>&1; then
  uv venv "$VENV" --python 3.12 -q
  uv pip install -q --python "$PY" -r "$HERE/requirements.txt"
else
  BASE=""
  for v in python3.12 python3.11 python3.10; do
    if command -v "$v" >/dev/null 2>&1; then BASE="$v"; break; fi
  done
  if [ -z "$BASE" ]; then
    echo "depth-ghost: need Python 3.10-3.12 or uv. Install uv (https://docs.astral.sh/uv/) or 'brew install python@3.12', then re-run." >&2
    exit 1
  fi
  "$BASE" -m venv "$VENV"
  "$PY" -m pip install -q --upgrade pip
  "$PY" -m pip install -q -r "$HERE/requirements.txt"
fi

"$PY" -c "import torch, transformers, cv2, pillow_heif; print('depth-ghost: ready, torch', torch.__version__)"
echo "The depth model (~1.3 GB) downloads automatically on the first run."
