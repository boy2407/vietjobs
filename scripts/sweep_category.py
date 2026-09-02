"""Run the fair-comparison sweep for the job-classification task.

    python scripts/sweep_category.py --dry-run
    python scripts/sweep_category.py --deadline 07:00 | tee sweep.log

Six models, six configurations each, all on the same train split, the same
236.596-dimension feature space, the same seed, scored on val. The point is not
to find a better model — it is to remove the asymmetry that made the previous
comparison unreadable: svm had been swept over nine values of C and its best
kept, while each tree model had been run exactly once at a configuration
somebody guessed.

Three properties this script exists to guarantee
------------------------------------------------
1. **Strictly sequential.** One parent process, ``subprocess.run`` per run, which
   returns only when the child has exited. An earlier attempt used two
   background jobs waiting on each other through a hand-rolled ``pgrep``
   condition; two models ended up training at once, which corrupted the timing
   column and cost an hour of wall clock. There is no condition to get wrong
   here.
2. **Wave ordering.** Wave j runs configuration #j of every model before any
   model's configuration #j+1. If the night is cut short, what survives is k
   configurations for each of the six models — still a fair comparison, merely
   a smaller one. Ordering by model instead would leave "knn finished, xgb never
   started", which compares nothing.
3. **Anchors.** Configuration #1 of five models reproduces a row that already
   exists in 04-results.md. Wave 1 therefore doubles as a reproducibility check
   that costs no extra time: if an anchor misses its known macro-F1, something
   changed and the whole sweep is suspect.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"

# Fixed for every run in the sweep. Changing any of these makes the 36 rows
# incomparable with each other and with the 33 rows already in the log.
COMMON = ["--task", "category", "--scope", "full", "--province", "--eval", "val"]

# (config dict, estimated seconds, anchor macro-F1 or None)
# Estimates come from measured runs on this corpus; --dry-run scales them by
# --calib, the factor measured on the machine that will actually run the sweep.
GRID: dict[str, list[tuple[dict, int, float | None]]] = {
    # One knob. metric=cosine and algorithm=brute are not tuning choices —
    # KD-tree and Ball-tree cannot index a high-dimensional sparse matrix at all
    # (see models.py). Sweeping k from 3 to 50 settles whether knn collapsed
    # because k was wrong or because of the curse of dimensionality.
    "knn": [
        ({"n_neighbors": 30}, 43, 0.4059),
        ({"n_neighbors": 15}, 43, None),
        ({"n_neighbors": 5}, 43, None),
        ({"n_neighbors": 50}, 43, None),
        ({"n_neighbors": 10}, 43, None),
        ({"n_neighbors": 3}, 43, None),
    ],
    # Six values bracketing the known optimum at 0.02. class_weight is left out
    # on purpose: mixing a second knob into six cells would cost the resolution
    # around the peak.
    "svm": [
        ({"C": 0.02}, 28, 0.6050),
        ({"C": 0.05}, 34, None),
        ({"C": 0.01}, 34, None),
        ({"C": 0.1}, 38, None),
        ({"C": 0.005}, 45, None),
        ({"C": 0.2}, 91, None),
    ],
    # Only two points measured before this sweep — C=4 -> 0.5941 and C=1 ->
    # 0.6038 — both at or above 1, and the score was still climbing as C fell.
    # Nobody had gone lower, while svm's optimum on this same matrix sits at
    # 0.02. So the grid runs downhill all the way to 0.02.
    #
    # C=2.0 was in this list and was dropped: it falls between two measured
    # points that bracket it, so its result is interpolable at ~0.599 and it
    # would have spent one of six cells confirming something already known.
    # This is the model most likely to overturn the conclusion — runner-up by
    # 0.0012, which is inside the noise band, and never swept properly.
    "logreg": [
        ({"C": 1.0}, 685, 0.6038),
        ({"C": 0.5}, 500, None),
        ({"C": 0.25}, 400, None),
        ({"C": 0.1}, 300, None),
        ({"C": 0.02}, 220, None),
        ({"C": 0.05}, 250, None),
    ],
    # max_features="sqrt" draws 486 of 236.596 columns per split on a matrix
    # that is 99.79% zero, so most draws cannot split anything. #5 and #6
    # bracket that argument with two numbers instead of leaving it as prose.
    "rf": [
        ({"n_estimators": 100}, 69, 0.5630),
        ({"n_estimators": 300}, 208, None),
        ({"n_estimators": 600}, 416, None),
        ({"n_estimators": 300, "min_samples_leaf": 1}, 270, None),
        ({"n_estimators": 300, "max_features": 1000}, 428, None),
        ({"n_estimators": 300, "max_features": "log2"}, 30, None),
    ],
    # learning_rate * n_estimators is held near 30 wherever the round count
    # moves, so a low score reads as "boosting is weak here" rather than "we
    # stopped early".
    "lgbm": [
        ({"n_estimators": 100, "learning_rate": 0.3}, 744, 0.5555),
        ({"n_estimators": 100, "learning_rate": 0.3, "num_leaves": 31}, 400, None),
        ({"n_estimators": 100, "learning_rate": 0.3, "colsample_bytree": 0.1}, 300, None),
        ({"n_estimators": 200, "learning_rate": 0.15, "num_leaves": 31}, 800, None),
        ({"n_estimators": 100, "learning_rate": 0.3, "num_leaves": 95}, 1050, None),
        ({"n_estimators": 100, "learning_rate": 0.3, "colsample_bytree": 0.5}, 1200, None),
    ],
    # Cost scales with node count 2**depth - 1, so depth is the lever, not the
    # round count. depth=8 is what killed the 400-round run after 4h07m.
    "xgb": [
        ({"n_estimators": 100, "learning_rate": 0.3, "max_depth": 6}, 3209, 0.5861),
        ({"n_estimators": 100, "learning_rate": 0.3, "max_depth": 4}, 765, None),
        ({"n_estimators": 100, "learning_rate": 0.3, "max_depth": 4,
          "colsample_bytree": 0.1}, 255, None),
        ({"n_estimators": 100, "learning_rate": 0.3, "max_depth": 6,
          "colsample_bytree": 0.1}, 1070, None),
        ({"n_estimators": 200, "learning_rate": 0.15, "max_depth": 4,
          "colsample_bytree": 0.2}, 1030, None),
        ({"n_estimators": 100, "learning_rate": 0.3, "max_depth": 5,
          "colsample_bytree": 0.2}, 530, None),
    ],
}

# Boosting models default to 400 rounds in the registry — the configuration that
# ran for 4h07m without finishing. A cell that forgets to state its budget would
# silently inherit that.
MUST_DECLARE = {"lgbm": {"n_estimators", "learning_rate"},
                "xgb": {"n_estimators", "learning_rate"}}

DEFAULT_TIMEOUT = 2700
LONG_TIMEOUT = 5400          # xgb #1 is estimated at 3209s
LONG_RUNS = {("xgb", 1)}


def run_id(model: str, n: int) -> str:
    """Deterministic — a timestamp here would break resume."""
    return f"cat-SW-{model}-{n}"


def already_done(rid: str) -> bool:
    d = ARTIFACTS / rid
    if not ((d / "metrics.json").exists() and (d / "predictions.npz").exists()):
        return False
    try:
        return json.loads((d / "metrics.json").read_text())["eval_split"] == "val"
    except Exception:
        return False


def build_plan() -> list[dict]:
    """Wave-major order; cheapest first inside each wave."""
    plan = []
    for wave in range(6):
        cells = []
        for model, configs in GRID.items():
            cfg, est, anchor = configs[wave]
            missing = MUST_DECLARE.get(model, set()) - cfg.keys()
            if missing:
                raise SystemExit(
                    f"{model} #{wave + 1} must state {sorted(missing)} explicitly — "
                    f"otherwise it inherits the registry default that ran 4h07m "
                    f"without finishing."
                )
            cells.append({"model": model, "n": wave + 1, "cfg": cfg,
                          "est": est, "anchor": anchor,
                          "rid": run_id(model, wave + 1)})
        plan.extend(sorted(cells, key=lambda c: c["est"]))
    return plan


def command(cell: dict) -> list[str]:
    cmd = [sys.executable, "-m", "vietjobs.train", *COMMON,
           "--model", cell["model"], "--run-id", cell["rid"]]
    for k, v in cell["cfg"].items():
        cmd += ["--set", f"{k}={v}"]
    return cmd


def machine_state() -> dict:
    def sh(*a):
        try:
            return subprocess.check_output(a, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return ""
    return {"uptime": sh("uptime"), "vm_stat_head": sh("vm_stat").splitlines()[:4]}


def fmt(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--deadline", default=None, metavar="HH:MM",
                    help="stop starting new runs after this local time")
    ap.add_argument("--calib", type=float, default=1.0,
                    help="machine factor from the calibration run; scales every estimate")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="run even with uncommitted changes (rows will not be reproducible)")
    args = ap.parse_args()

    plan = build_plan()
    pending = [c for c in plan if not already_done(c["rid"])]
    total_est = sum(c["est"] for c in pending) * args.calib + 40 * len(pending)

    print(f"{len(plan)} runs · {len(plan) - len(pending)} already done · "
          f"{len(pending)} to go")
    print(f"estimate (calib {args.calib:g}): {fmt(total_est)}\n")
    print(f"{'wave':>4} {'run_id':22} {'est':>7}  config")
    for c in plan:
        mark = "skip" if already_done(c["rid"]) else "    "
        anchor = f"  anchor={c['anchor']}" if c["anchor"] else ""
        cfg = ",".join(f"{k}={v}" for k, v in c["cfg"].items())
        print(f"{c['n']:>4} {c['rid']:22} {fmt(c['est'] * args.calib):>7}  "
              f"{mark} {cfg}{anchor}")
    if args.dry_run:
        return

    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()
    if dirty and not args.allow_dirty:
        raise SystemExit(
            "\nRefusing to run with uncommitted changes.\n"
            "Every row would be logged as 'uncommit+dirty', which docs/03-protocol.md §3\n"
            "defines as not reproducible — the one thing this sweep is meant to establish.\n"
            "Commit first, or pass --allow-dirty deliberately."
        )

    stop_at = None
    if args.deadline:
        hh, mm = (int(x) for x in args.deadline.split(":"))
        stop_at = datetime.now().replace(hour=hh, minute=mm, second=0, microsecond=0)
        if stop_at <= datetime.now():
            stop_at += timedelta(days=1)
        print(f"\ndeadline: {stop_at:%Y-%m-%d %H:%M}")

    status_dir = ARTIFACTS / f"sweep-{datetime.now():%Y%m%d-%H%M}"
    status_dir.mkdir(parents=True, exist_ok=True)
    status = {"started": datetime.now().isoformat(timespec="seconds"),
              "machine_start": machine_state(), "runs": []}

    t_start = time.perf_counter()
    for i, cell in enumerate(pending, 1):
        if stop_at and datetime.now() >= stop_at:
            print(f"\n[deadline] {len(pending) - i + 1} runs left unstarted")
            break
        timeout = LONG_TIMEOUT if (cell["model"], cell["n"]) in LONG_RUNS else DEFAULT_TIMEOUT
        print(f"\n===== [{i}/{len(pending)}] {cell['rid']} "
              f"start {datetime.now():%H:%M:%S} (timeout {fmt(timeout)}) =====", flush=True)
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(command(cell), cwd=ROOT, timeout=timeout,
                                  capture_output=True, text=True)
            elapsed, outcome = time.perf_counter() - t0, ("ok" if proc.returncode == 0
                                                          else f"exit {proc.returncode}")
            for line in proc.stdout.splitlines():
                if line.startswith("  fit ") or line.startswith("  macroF1"):
                    print(line, flush=True)
            if proc.returncode != 0:
                print(proc.stderr[-2000:], flush=True)
        except subprocess.TimeoutExpired:
            elapsed, outcome = time.perf_counter() - t0, "DNF (timeout)"
            print(f"  {outcome} after {fmt(elapsed)}", flush=True)

        done = time.perf_counter() - t_start
        remaining = sum(c["est"] for c in pending[i:]) * args.calib
        print(f"===== {cell['rid']} {outcome} in {fmt(elapsed)} · "
              f"elapsed {fmt(done)} · eta {fmt(remaining)} =====", flush=True)

        status["runs"].append({"run_id": cell["rid"], "outcome": outcome,
                               "seconds": round(elapsed, 1),
                               "finished": datetime.now().isoformat(timespec="seconds")})
        status["machine_end"] = machine_state()
        (status_dir / "status.json").write_text(json.dumps(status, indent=2))

    print(f"\nsweep finished in {fmt(time.perf_counter() - t_start)} · "
          f"status: {status_dir / 'status.json'}")


if __name__ == "__main__":
    main()
