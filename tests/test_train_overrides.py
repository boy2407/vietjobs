"""The --set mechanism, and the log row it produces.

These exist because a sweep that silently ignores a hyper-parameter produces
six identical runs wearing six different names, and nothing about the output
looks wrong. Both failures below are silent by nature, so they need tests:

  * an override that never reaches the estimator
  * a metric string containing '|', which pushes Time and Commit out of the
    markdown row without raising anything
"""
from __future__ import annotations

import pytest

from vietjobs import config as C
from vietjobs.models import CLASSIFIERS
from vietjobs.train import HEADLINE, model_label, parse_overrides


# --- parse_overrides: types must survive the trip -------------------------


@pytest.mark.parametrize(
    "pair, key, value",
    [
        ("num_leaves=63", "num_leaves", 63),
        ("learning_rate=0.3", "learning_rate", 0.3),
        ("max_features=log2", "max_features", "log2"),  # not a Python literal
        ("class_weight=None", "class_weight", None),
        ("bootstrap=True", "bootstrap", True),
    ],
)
def test_parse_overrides_coerces_types(pair, key, value):
    got = parse_overrides([pair])
    assert got == {key: value}
    assert type(got[key]) is type(value)


def test_parse_overrides_rejects_missing_equals():
    with pytest.raises(SystemExit):
        parse_overrides(["num_leaves"])


# --- model_label: new rows must read like the old ones ---------------------


def test_model_label_matches_the_legacy_c_format():
    # 33 existing rows in 04-results.md are written as e.g. "svm(C=0.02)".
    assert model_label("svm", {"C": 0.02}) == "svm(C=0.02)"


def test_model_label_bare_when_nothing_overridden():
    assert model_label("rf", {}) == "rf"


def test_model_label_lists_several_knobs():
    assert model_label("xgb", {"max_depth": 4, "colsample_bytree": 0.1}) == (
        "xgb(max_depth=4,colsample_bytree=0.1)"
    )


# --- overrides must actually reach the estimator ---------------------------


@pytest.mark.parametrize(
    "name, knob, value",
    [
        ("rf", "min_samples_leaf", 1),
        ("rf", "max_features", "log2"),
        ("lgbm", "num_leaves", 31),
        # xgb hid colsample_bytree inside fit() until it was hoisted onto
        # __init__. Without this case, six "xgb configurations" would silently
        # collapse into six runs that differ only in depth.
        ("xgb", "colsample_bytree", 0.1),
        ("xgb", "max_bin", 128),
    ],
)
def test_set_params_reaches_the_estimator(name, knob, value):
    est = CLASSIFIERS[name](C.RANDOM_SEED)
    assert knob in est.get_params(), f"{knob} not tunable on {name}"
    est.set_params(**{knob: value})
    assert est.get_params()[knob] == value


def test_unknown_knob_is_visible_in_get_params():
    # The guard in train.main() rejects anything absent here, so this is the
    # contract that guard relies on.
    assert "nonsense_knob" not in CLASSIFIERS["svm"](C.RANDOM_SEED).get_params()


# --- the log row must stay inside its columns ------------------------------


def test_headline_contains_no_pipe():
    metrics = {
        "f1_macro": 0.6050,
        "f1_macro_boot_std": 0.0091,
        "accuracy": 0.6445,
        "balanced_accuracy": 0.6713,
        "f1_macro_no_junk": 0.6376,
        "top3_accuracy": 0.9318,
    }
    headline = HEADLINE[C.TASK_CATEGORY](metrics)
    assert "|" not in headline, "a '|' here silently truncates the results table"
    assert "0.6050±0.0091" in headline
