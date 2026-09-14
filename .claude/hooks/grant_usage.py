"""Human runs this (via grant-usage.sh) to grant the next ~5% of the 5-hour window."""
import json, os, time
u = os.path.expanduser("~/.claude/vietjobs-usage.json")
g = os.path.expanduser("~/.claude/vietjobs-usage-grant.json")
fh = json.load(open(u))["five_hour"] if os.path.exists(u) else 0
json.dump({"baseline": fh, "ts": int(time.time())}, open(g, "w"))
print(f"granted: next 5% allowed from {fh:.0f}% of the 5h window (warning at +4%)")
