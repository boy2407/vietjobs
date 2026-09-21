"""Metrics for the three tasks.

Classification leads with macro-F1: the largest class is 27x the smallest, so
accuracy alone would happily hide a model that ignores the tail entirely.

A second macro-F1 excludes ``nhóm_nghề_khác``. That class is a catch-all
containing "Nhân Viên Seo Web" and "Nhân Viên Quản Trị Website" — postings that
belong to marketing and IT. It is label noise, not model error, and reporting
both numbers separates "the model is weak" from "the label is wrong".

Regression is reported in the original unit (triệu VND/month), never in log
space, because a log-space MAE is not a number anyone can act on.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from . import config as C


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def classification_metrics(y_true, y_pred, labels=None) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    out = {
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "n": int(len(y_true)),
    }

    # Same metric with the junk class removed — see module docstring.
    keep = y_true != C.JUNK_CATEGORY
    if keep.sum() and keep.sum() < len(y_true):
        real_labels = [l for l in (labels or sorted(set(y_true))) if l != C.JUNK_CATEGORY]
        out["f1_macro_no_junk"] = float(
            f1_score(
                y_true[keep], y_pred[keep], labels=real_labels,
                average="macro", zero_division=0,
            )
        )

    return out


def per_class_report(y_true, y_pred, labels=None) -> dict:
    return classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )


def confusion(y_true, y_pred, labels) -> list[list[int]]:
    return confusion_matrix(y_true, y_pred, labels=labels).tolist()


def top_confusions(y_true, y_pred, labels, k: int = 10) -> list[dict]:
    """The k most frequent off-diagonal cells.

    More useful in a report than the full 16x16 grid: it names which pairs of
    categories actually overlap.
    """
    cm = np.array(confusion_matrix(y_true, y_pred, labels=labels))
    np.fill_diagonal(cm, 0)
    flat = np.argsort(cm, axis=None)[::-1][:k]
    out = []
    for idx in flat:
        i, j = np.unravel_index(idx, cm.shape)
        if cm[i, j] == 0:
            break
        out.append({"true": labels[i], "predicted": labels[j], "n": int(cm[i, j])})
    return out


# ---------------------------------------------------------------------------
# Bootstrap — turning "is this difference real?" into a number
# ---------------------------------------------------------------------------
#
# docs/ quotes sigma ~= 0.009 in five places to rule differences in or out, but
# until now nothing in this repo computed it. These four functions do, and they
# are deliberately split so that ONE index matrix can be shared across every
# model in a sweep.
#
# That sharing is the whole point. Comparing two models through their separate
# error bars asks "could these two numbers have come from the same model?",
# which is far too conservative: both models see the same val rows and their
# mistakes are strongly correlated. Resampling the SAME rows for both and
# looking at the paired difference removes that shared variance, so std(delta)
# comes out much smaller than either model's own sigma. A gap the independent
# rule calls noise can be decisively real under the paired one.


def bootstrap_indices(n: int, n_boot: int = 1000, seed: int = C.RANDOM_SEED):
    """``(n_boot, n)`` matrix of row indices, sampled with replacement.

    Build this once and pass the same matrix to every model being compared —
    that is what makes the comparisons paired.

    On this project's val split (7.159 rows) the rarest class holds 48 rows, so
    the chance it is absent from a resample is ``(1 - 48/7159) ** 7159`` ~= 1e-21.
    No guard needed. Those small classes are nonetheless where most of macro-F1's
    variance comes from, which is worth saying out loud in any report.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_boot, n), dtype=np.int32)


def bootstrap_scores(y_true, y_pred, idx, metric: str = "f1_macro"):
    """Score the model once per resample. Returns ``(n_boot,)`` floats.

    ``f1_macro`` goes through a confusion-matrix path rather than calling
    sklearn once per resample. A full sweep needs 36 models x 1000 resamples,
    and sklearn's per-call overhead turns that into minutes of pure dispatch.
    One ``bincount`` per resample gives the same numbers in about a second.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"length mismatch: {y_true.shape} vs {y_pred.shape}")

    if metric == "f1_macro":
        classes, codes = np.unique(np.concatenate([y_true, y_pred]), return_inverse=True)
        k = len(classes)
        t, p = codes[: len(y_true)], codes[len(y_true) :]
        combo = (t * k + p).astype(np.int64)
        out = np.empty(len(idx), dtype=float)
        for i, rows in enumerate(idx):
            cm = np.bincount(combo[rows], minlength=k * k).reshape(k, k)
            tp = np.diag(cm).astype(float)
            fp = cm.sum(axis=0) - tp
            fn = cm.sum(axis=1) - tp
            denom = 2 * tp + fp + fn
            # sklearn averages over the classes present in y_true or y_pred of
            # this resample; a class absent from both is not counted as a zero.
            present = denom > 0
            f1 = np.zeros(k)
            np.divide(2 * tp, denom, out=f1, where=present)
            out[i] = f1[present].mean() if present.any() else 0.0
        return out

    if metric != "accuracy":
        raise ValueError(f"metric không hỗ trợ: {metric}")
    return np.array([float(accuracy_score(y_true[i], y_pred[i])) for i in idx])


def bootstrap_summary(scores) -> dict:
    scores = np.asarray(scores, dtype=float)
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return {
        "mean": float(scores.mean()),
        "std": float(scores.std(ddof=1)) if scores.size > 1 else 0.0,
        "ci95_lo": float(lo),
        "ci95_hi": float(hi),
    }


def paired_delta(scores_a, scores_b) -> dict:
    """Distribution of ``a - b`` over the shared resamples.

    ``p_gt_0`` is the fraction of resamples where A beat B. Two identical models
    give 0.5 by convention (no resample has a strictly positive difference, and
    none has a negative one either), so the tie reads as "cannot distinguish"
    rather than as a win for either side.
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired comparison needs equal shapes: {a.shape} vs {b.shape}")
    d = a - b
    lo, hi = np.percentile(d, [2.5, 97.5])
    wins = float((d > 0).mean())
    ties = float((d == 0).mean())
    return {
        "mean": float(d.mean()),
        "std": float(d.std(ddof=1)) if d.size > 1 else 0.0,
        "ci95_lo": float(lo),
        "ci95_hi": float(hi),
        "p_gt_0": wins + ties / 2.0,
    }


# ---------------------------------------------------------------------------
# Binary (the "did this ad publish a salary?" stage)
# ---------------------------------------------------------------------------


def binary_metrics(y_true, y_pred, y_score=None) -> dict:
    out = {
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_positive": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "n": int(len(y_true)),
    }
    if y_score is not None:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        out["average_precision"] = float(average_precision_score(y_true, y_score))
    return out


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------


def regression_metrics(y_true_log, y_pred_log) -> dict:
    """Inputs are log1p(salary); everything reported is back-transformed."""
    y_true = np.expm1(np.asarray(y_true_log, dtype=float))
    y_pred = np.expm1(np.asarray(y_pred_log, dtype=float))
    y_pred = np.clip(y_pred, 0.5, None)  # a zero or negative salary is not a prediction

    abs_err = np.abs(y_true - y_pred)
    ape = abs_err / np.clip(np.abs(y_true), 1e-6, None)

    return {
        "mae_trieu": float(mean_absolute_error(y_true, y_pred)),
        "rmse_trieu": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": float(np.mean(ape)),
        "r2_log": float(r2_score(y_true_log, y_pred_log)),
        "r2_raw": float(r2_score(y_true, y_pred)),
        "within_10pct": float(np.mean(ape <= 0.10)),
        "within_20pct": float(np.mean(ape <= 0.20)),
        "within_3_trieu": float(np.mean(abs_err <= 3.0)),
        "n": int(len(y_true)),
    }


def slice_report(y_true_log, y_pred_log, groups, min_n: int = 30) -> dict:
    """Per-slice MAE — one average hides which segments the model fails on."""
    y_true_log = np.asarray(y_true_log, dtype=float)
    y_pred_log = np.asarray(y_pred_log, dtype=float)
    groups = np.asarray(groups)
    out = {}
    for g in sorted(set(groups.tolist())):
        m = groups == g
        if m.sum() >= min_n:
            out[str(g)] = regression_metrics(y_true_log[m], y_pred_log[m])
    return out
