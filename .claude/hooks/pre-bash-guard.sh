#!/bin/bash
# PreToolUse(Bash): AGENTS.md Rule 5 and destructive commands.
cmd=$(python3 -c 'import json,sys;print(json.load(sys.stdin).get("tool_input",{}).get("command",""))')
if echo "$cmd" | grep -q -- '--confirm-test' && [ "$VIETJOBS_ALLOW_TEST" != "1" ]; then
  echo "Rule 5: test is read once, by a human. Run it yourself with VIETJOBS_ALLOW_TEST=1." >&2
  exit 2
fi
if echo "$cmd" | grep -Eq 'push +(-f|--force)|reset +--hard|rm +-rf +(data|artifacts)'; then
  echo "Blocked: destructive command. Ask the human." >&2
  exit 2
fi
exit 0
