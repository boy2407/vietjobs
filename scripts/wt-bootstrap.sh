#!/bin/bash
# Run once inside a git worktree: link the untracked inputs and venvs from
# the main checkout. Safe to re-run; does nothing in the main checkout.
main=$(git worktree list | head -1 | awk '{print $1}')
[ "$main" = "$(git rev-parse --show-toplevel)" ] && { echo "main checkout, nothing to do"; exit 0; }
for p in data/raw data/processed .venv .venv-dl; do
  [ -e "$p" ] || ln -s "$main/$p" "$p" 2>/dev/null
done
echo "bootstrapped from $main"
