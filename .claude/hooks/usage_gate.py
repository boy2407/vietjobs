"""PreToolUse gate: block once ~4% of the Max 5-hour window has been used since the
human's last grant (the human's limit is 5%; 4% is the warning point). Data comes
from statusline.py; if there is none, warn only."""
import json, os, sys, time
STEP, HARD = 4.0, 90
u = os.path.expanduser("~/.claude/vietjobs-usage.json")
g = os.path.expanduser("~/.claude/vietjobs-usage-grant.json")
if not os.path.exists(u):
    print("usage gate: no rate-limit data yet (statusline not running?) - allowing", file=sys.stderr); sys.exit(0)
d = json.load(open(u)); fh = d["five_hour"]; age = time.time() - d["ts"]
base = json.load(open(g))["baseline"] if os.path.exists(g) else fh
if age > 1800:
    print(f"usage gate: data is {age/60:.0f} min old - allowing, check /usage", file=sys.stderr); sys.exit(0)
if fh >= HARD or fh - base >= STEP:
    print(f"USAGE GATE - 5h window at {fh:.0f}%, +{fh-base:.1f}% since last grant (limit 5%).\n"
          "FOR HUMAN: stop here. To allow the next 5%, run in your own terminal:\n"
          "  bash .claude/hooks/grant-usage.sh\n"
          "then tell the agent to continue.", file=sys.stderr)
    sys.exit(2)
