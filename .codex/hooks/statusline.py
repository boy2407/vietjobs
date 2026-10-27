"""Statusline: show the Max 5h/7d windows and write them to a file the usage gate reads."""
import json, os, sys, time
d = json.load(sys.stdin)
rl = d.get("rate_limits") or {}
fh = (rl.get("five_hour") or {}).get("used_percentage")
sd = (rl.get("seven_day") or {}).get("used_percentage")
ctx = (d.get("context_window") or {}).get("used_percentage")
if fh is None:
    print("5h ?% (no rate_limits in statusline input yet)"); sys.exit(0)
u = os.path.expanduser("~/.claude/vietjobs-usage.json")
g = os.path.expanduser("~/.claude/vietjobs-usage-grant.json")
json.dump({"five_hour": fh, "seven_day": sd, "ts": int(time.time())}, open(u, "w"))
if not os.path.exists(g):
    json.dump({"baseline": fh, "ts": int(time.time())}, open(g, "w"))
base = json.load(open(g))["baseline"]
print(f"5h {fh:.0f}% (+{max(fh-base,0):.1f}% since grant, gate at +4%) | 7d {sd if sd is not None else '?'}% | ctx {ctx if ctx is not None else '?'}%")
