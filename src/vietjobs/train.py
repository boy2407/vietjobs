"""Train and evaluate one run.

    python -m vietjobs.train --task category --model svm --scope full --segment
    python -m vietjobs.train --task salary   --model lgbm --scope full
    python -m vietjobs.train --task disclosed --model logreg
    python -m vietjobs.train --task category --model xgb --set n_estimators=100 \\
                             --set learning_rate=0.3 --set max_depth=4

Each run writes ``artifacts/<run_id>/`` (fitted pipeline + metrics.json +
predictions.npz + environment fingerprint) and appends one row to
``docs/04-results.md``.

``predictions.npz`` holds the raw per-row val predictions. It exists so that two
runs can be compared by *paired* bootstrap — resampling the same val rows for
both models — which is far more sensitive than comparing their independent error
bars. See ``evaluate.paired_delta``.

Model and hyper-parameter selection use ``--eval val``. Scoring the test set
requires ``--eval test --confirm-test``, which exists so that touching it is a
deliberate act rather than a default.
"""
from __future__ import annotations

import argparse
import ast
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from . import config as C
from . import evaluate as E
from . import features as F
from .dataset import load_split
from .models import build_estimator


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=C.ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return ""


def _environment() -> dict:
    import sklearn

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "git_sha": _git("rev-parse", "HEAD") or "uncommitted",
        "git_dirty": bool(_git("status", "--porcelain")),
    }


def prepare(task: str, df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Select the rows and the target column for a task."""
    if task == C.TASK_CATEGORY:
        return df, df["category"].to_numpy()
    if task == C.TASK_DISCLOSED:
        return df, df["salary_disclosed"].to_numpy()
    if task == C.TASK_SALARY:
        # Regression trains only on ads that published a number.
        sub = df[df["salary_disclosed"] == 1].reset_index(drop=True)
        return sub, sub["salary_mid_log"].to_numpy(dtype=float)
    raise ValueError(task)


def evaluate_run(task, y_true, y_pred, y_proba, y_score, frame, labels) -> dict:
    if task == C.TASK_CATEGORY:
        out = E.classification_metrics(y_true, y_pred, y_proba=y_proba, labels=labels)
        out["labels"] = list(labels)
        out["per_class"] = E.per_class_report(y_true, y_pred, labels=labels)
        out["confusion_matrix"] = E.confusion(y_true, y_pred, labels)
        out["top_confusions"] = E.top_confusions(y_true, y_pred, labels)
        return out
    if task == C.TASK_DISCLOSED:
        return E.binary_metrics(y_true, y_pred, y_score)
    out = E.regression_metrics(y_true, y_pred)
    out["by_category"] = E.slice_report(y_true, y_pred, frame["category"])
    out["by_experience"] = E.slice_report(y_true, y_pred, frame["experience_raw"])
    out["by_province"] = E.slice_report(y_true, y_pred, frame["province"])
    return out


def parse_overrides(pairs) -> dict:
    """``["num_leaves=63", "max_features=log2"]`` -> ``{"num_leaves": 63, ...}``.

    Values go through ``ast.literal_eval`` so ``0.3`` becomes a float and ``100``
    an int; anything that is not a Python literal (``log2``, ``sqrt``,
    ``balanced``) falls back to the string. Never ``eval`` — the sweep script
    builds these strings, but a typo should raise, not execute.
    """
    out = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--set expects KEY=VALUE, got {pair!r}")
        key, _, raw = pair.partition("=")
        try:
            out[key.strip()] = ast.literal_eval(raw.strip())
        except (ValueError, SyntaxError):
            out[key.strip()] = raw.strip()
    return out


def model_label(model: str, overrides: dict) -> str:
    """Name for the Model column: ``svm`` plus the knobs that were overridden.

    A lone ``C`` renders as ``svm(C=0.02)`` — byte-identical to the format the
    first 33 rows used — so old and new rows in 04-results.md stay comparable.
    """
    if not overrides:
        return model
    inner = ",".join(f"{k}={v}" for k, v in overrides.items())
    return f"{model}({inner})"


# Metrics inside one cell are separated by ' · ', never '|': the log is a
# markdown table, and a '|' here silently pushes Time and Commit out of the row.
HEADLINE = {
    C.TASK_CATEGORY: lambda m: (
        f"macroF1={m['f1_macro']:.4f}"
        + (f"±{m['f1_macro_boot_std']:.4f}" if "f1_macro_boot_std" in m else "")
        + f" · acc={m['accuracy']:.4f} · balAcc={m['balanced_accuracy']:.4f}"
        + (f" · F1@15={m['f1_macro_no_junk']:.4f}" if "f1_macro_no_junk" in m else "")
        + (f" · top3={m['top3_accuracy']:.4f}" if "top3_accuracy" in m else "")
    ),
    C.TASK_SALARY: lambda m: (
        f"MAE={m['mae_trieu']:.2f}tr · MedAE={m['median_ae_trieu']:.2f}tr · "
        f"R2log={m['r2_log']:.3f} · ±20%={m['within_20pct'] * 100:.1f}%"
    ),
    C.TASK_DISCLOSED: lambda m: (
        f"macroF1={m['f1_macro']:.4f} · acc={m['accuracy']:.4f}"
        + (f" · AUC={m['roc_auc']:.4f}" if "roc_auc" in m else "")
    ),
}


def append_results_row(run_id, args, prep, metrics, seconds, env, label=None) -> None:
    """Append-only experiment log. Never edit a row that is already written."""
    path = C.RESULTS_LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(
            "# Experiment log (append-only)\n\n"
            "Selection uses `val`. `test` is scored once, at the end.\n\n"
            "| RunID | UTC | Task | Model | Scope | Prep | Eval | n | Headline | Time | Commit |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|\n",
            encoding="utf-8",
        )
    row = (
        f"| {run_id} | {datetime.now(timezone.utc).strftime('%m-%d %H:%M')} | {args.task} "
        f"| {label or args.model} "
        f"| {args.scope}{f'+svd{args.svd}' if args.svd else ''} | {prep.tag()} "
        f"| {args.eval} | {metrics['n']} | {HEADLINE[args.task](metrics)} | {seconds:.1f}s "
        f"| {env['git_sha'][:8]}{'+dirty' if env['git_dirty'] else ''} |\n"
    )
    with path.open("a", encoding="utf-8") as fh:
        fh.write(row)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--task", choices=C.TASKS, required=True)
    ap.add_argument("--model", default=None, help="see vietjobs/models.py")
    ap.add_argument("--scope", choices=["structured", "title", "full"], default="full")
    ap.add_argument("--svd", type=int, default=None, help="project to N dense dims")

    # Vietnamese preprocessing ablation switches (vitext.py sections)
    ap.add_argument("--segment", action="store_true", help="2.4 word segmentation")
    ap.add_argument("--charfold", action="store_true", help="2.6 accent-folded char channel")
    ap.add_argument("--province", action="store_true", help="2.7 district -> province")
    ap.add_argument("--stopwords", action="store_true", help="2.5 drop stopwords")

    ap.add_argument("--C", type=float, default=None, help="regularisation for svm / logreg")
    ap.add_argument("--k", type=int, default=None, help="n_neighbors for knn")
    ap.add_argument(
        "--set", action="append", default=[], dest="overrides", metavar="KEY=VALUE",
        help="any estimator hyper-parameter, repeatable: --set num_leaves=63",
    )
    ap.add_argument(
        "--boot", type=int, default=1000, metavar="N",
        help="bootstrap resamples for the macro-F1 error bar (0 disables)",
    )
    ap.add_argument("--eval", choices=["val", "test"], default="val")
    ap.add_argument("--confirm-test", action="store_true")
    ap.add_argument("--seed", type=int, default=C.RANDOM_SEED)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    if args.model is None:
        args.model = "ridge" if args.task == C.TASK_SALARY else "svm"
    if args.eval == "test" and not args.confirm_test:
        raise SystemExit(
            "Refusing to score the test set without --confirm-test.\n"
            "Model and hyper-parameter selection use --eval val."
        )

    prep = F.PrepConfig(
        segment=args.segment,
        charfold=args.charfold,
        province=args.province,
        stopwords=args.stopwords,
    )
    run_id = args.run_id or (
        f"{args.task[:3]}-{args.model}-{args.scope}-{prep.tag()}"
        f"-{datetime.now().strftime('%m%d%H%M')}"
    )
    np.random.seed(args.seed)

    train_df = load_split("train")
    eval_df = load_split(args.eval)
    Xtr, ytr = prepare(args.task, train_df)
    Xev, yev = prepare(args.task, eval_df)

    estimator, reads_dataframe = build_estimator(args.task, args.model, args.seed)
    # Hyper-parameter overrides. Set on the built estimator rather than adding a
    # registry entry per value, so the sweep stays one line per run in the log.
    #
    # An unknown key is fatal, not ignored. The previous `hasattr` guard let
    # `--C 0.5 --model knn` pass silently, which is exactly how a "fair" sweep
    # turns into six identical runs that nobody notices.
    overrides = {}
    if args.C is not None:
        overrides["C"] = args.C
    if args.k is not None:
        overrides["n_neighbors"] = args.k
    overrides.update(parse_overrides(args.overrides))
    if overrides:
        valid = estimator.get_params()
        unknown = sorted(k for k in overrides if k not in valid)
        if unknown:
            raise SystemExit(
                f"unknown hyper-parameter(s) {unknown} for model {args.model!r}.\n"
                f"  valid: {', '.join(sorted(valid))}"
            )
        estimator.set_params(**overrides)
    label = model_label(args.model, overrides)

    t0 = time.perf_counter()
    if reads_dataframe:
        pipeline = estimator.fit(Xtr, ytr)
    else:
        pipeline = Pipeline(
            [
                ("features", F.build_features(args.task, args.scope, prep, args.svd)),
                ("estimator", estimator),
            ]
        )
        pipeline.fit(Xtr, ytr)
    fit_seconds = time.perf_counter() - t0

    t1 = time.perf_counter()
    y_pred = pipeline.predict(Xev)
    predict_seconds = time.perf_counter() - t1

    y_proba = y_score = None
    if hasattr(pipeline, "predict_proba"):
        try:
            y_proba = pipeline.predict_proba(Xev)
        except Exception:
            pass
    if args.task == C.TASK_DISCLOSED:
        if y_proba is not None:
            y_score = y_proba[:, 1]
        elif hasattr(pipeline, "decision_function"):
            y_score = pipeline.decision_function(Xev)

    labels = sorted(pd.unique(train_df["category"])) if args.task == C.TASK_CATEGORY else None
    results = evaluate_run(args.task, yev, y_pred, y_proba, y_score, Xev, labels)

    # Error bar on the headline metric, so every new row carries its own
    # uncertainty instead of being compared against a sigma nobody can recompute.
    if args.task == C.TASK_CATEGORY and args.boot > 0:
        idx = E.bootstrap_indices(len(yev), n_boot=args.boot, seed=args.seed)
        boot = E.bootstrap_summary(E.bootstrap_scores(yev, y_pred, idx))
        results["f1_macro_boot_std"] = boot["std"]
        results["f1_macro_boot_ci95"] = [boot["ci95_lo"], boot["ci95_hi"]]
        results["f1_macro_boot_n"] = args.boot

    env = _environment()
    payload = {
        "run_id": run_id,
        "task": args.task,
        "model": args.model,
        "model_label": label,
        "overrides": overrides,
        # Full estimator state, so a run describes itself without anyone having
        # to reconstruct which registry entry it came from.
        "estimator_params": json.loads(json.dumps(estimator.get_params(), default=repr)),
        "scope": args.scope,
        "svd": args.svd,
        "prep": prep.tag(),
        "prep_detail": prep.__dict__,
        "eval_split": args.eval,
        "seed": args.seed,
        "fit_seconds": round(fit_seconds, 2),
        "predict_seconds": round(predict_seconds, 2),
        "n_train": int(len(Xtr)),
        "n_eval": int(len(Xev)),
        "environment": env,
        "dataset_manifest": (
            json.loads(C.MANIFEST.read_text(encoding="utf-8")) if C.MANIFEST.exists() else None
        ),
        "metrics": results,
    }

    if not args.no_save:
        out_dir = C.ARTIFACT_DIR / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metrics.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        joblib.dump(pipeline, out_dir / "model.joblib", compress=3)
        # Raw predictions, for paired bootstrap against other runs. row_index is
        # not decoration: the report script asserts two runs scored the same rows
        # before it is allowed to subtract their scores.
        arrays = {
            "row_index": np.asarray(Xev.index),
            "y_true": np.asarray(yev),
            "y_pred": np.asarray(y_pred),
            "labels": np.asarray(labels if labels is not None else []),
        }
        if y_proba is not None:
            arrays["y_proba"] = np.asarray(y_proba, dtype=np.float32)
        np.savez_compressed(out_dir / "predictions.npz", **arrays)
        append_results_row(run_id, args, prep, results, fit_seconds, env, label=label)

    bulky = {"per_class", "confusion_matrix", "labels", "top_confusions",
             "by_category", "by_experience", "by_province"}
    print(
        f"\n[{run_id}]  task={args.task} model={args.model} scope={args.scope} "
        f"prep={prep.tag()} eval={args.eval}"
    )
    print(f"  fit {fit_seconds:.1f}s | predict {predict_seconds:.1f}s | n={results['n']}")
    print("  " + HEADLINE[args.task](results))
    print(json.dumps({k: v for k, v in results.items() if k not in bulky}, indent=2))
    if args.task == C.TASK_CATEGORY and results.get("top_confusions"):
        print("\n  top confusions (true -> predicted):")
        for c in results["top_confusions"][:6]:
            print(f"    {c['true'][:36]:36s} -> {c['predicted'][:36]:36s} {c['n']}")
    if not args.no_save:
        print(f"\n  saved -> artifacts/{run_id}/")


if __name__ == "__main__":
    main()
