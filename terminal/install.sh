#!/usr/bin/env bash
# Install the qq "!!" tweak into your zsh, idempotently.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
line="source \"$here/qq.zsh\""
rc="${ZDOTDIR:-$HOME}/.zshrc"

if grep -Fqs "$line" "$rc" 2>/dev/null; then
  echo "Already installed in $rc"
else
  printf '\n# qq_terminal: the "!!" -> local LLM tweak\n%s\n' "$line" >> "$rc"
  echo "Added to $rc"
fi
echo "Open a new zsh, or run: source \"$rc\""
