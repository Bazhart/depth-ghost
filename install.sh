#!/bin/bash
# Installs the depth-ghost skill for Claude Code, then its Python environment.
# Run from the repo folder:  bash install.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

# Homebrew + uv aren't on PATH in a bare script; add the usual places.
eval "$(/opt/homebrew/bin/brew shellenv 2>/dev/null || /usr/local/bin/brew shellenv 2>/dev/null || true)"
export PATH="$HOME/.local/bin:$PATH"

echo "== 1/3  Skill -> ~/.claude/skills/depth-ghost"
mkdir -p ~/.claude/skills
if [ -d ~/.claude/skills/depth-ghost ]; then
  mv ~/.claude/skills/depth-ghost ~/.claude/skills/depth-ghost.bak-$(date +%Y%m%d-%H%M%S)
  echo "   (your old copy is backed up next to it)"
fi
cp -R "$HERE/skill/depth-ghost" ~/.claude/skills/

echo "== 2/3  Tools (ffmpeg, uv)"
for t in ffmpeg uv; do
  if command -v "$t" >/dev/null 2>&1; then echo "   $t: ok"
  elif command -v brew >/dev/null 2>&1; then echo "   installing $t ..."; brew install -q "$t"
  else echo "   $t is missing and Homebrew isn't installed: get Homebrew from https://brew.sh, then re-run." >&2; exit 1
  fi
done

echo "== 3/3  Python environment (first time: a few minutes, ~1-2 GB)"
bash ~/.claude/skills/depth-ghost/scripts/setup.sh

echo
echo "Done. Open a NEW Claude Code session and type /depth-ghost with a photo or video attached."
