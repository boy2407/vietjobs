#!/bin/bash
# PostToolUse(Edit|Write): append-only results log (Rule 2) and pytest after code edits.
f=$(python3 -c 'import json,sys;print(json.load(sys.stdin).get("tool_input",{}).get("file_path",""))')
case "$f" in
  */docs/04-results.md)
    if git diff -U0 -- docs/04-results.md | grep -q '^-[^-]'; then
      echo "Rule 2: docs/04-results.md is append-only. A written row was changed; revert it." >&2
      exit 2
    fi ;;
  */src/*|*/tests/*)
    if [ -f data/processed/manifest.json ]; then
      if ! out=$(PYTHONPATH=src pytest -q -x 2>&1); then
        echo "pytest failed after editing $f:" >&2
        echo "$out" | tail -15 >&2
        exit 2
      fi
    else
      echo "pytest skipped: data/processed/manifest.json missing (do T0.1 first)"
    fi ;;
esac
exit 0
