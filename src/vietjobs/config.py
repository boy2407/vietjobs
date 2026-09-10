"""Paths, column groups, split policy and task names shared by every module."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

RAW_CSV = ROOT / "data" / "raw" / "VietJobs.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
SPLIT_DIR = PROCESSED_DIR / "splits"
MANIFEST = PROCESSED_DIR / "manifest.json"

RESOURCE_DIR = ROOT / "resources"
ABBREVIATIONS = RESOURCE_DIR / "abbreviations_vi.json"
STOPWORDS = RESOURCE_DIR / "stopwords_vi.txt"
PROVINCES = RESOURCE_DIR / "provinces_vi.json"

ARTIFACT_DIR = ROOT / "artifacts"
DOCS_DIR = ROOT / "docs"
RESULTS_LOG = DOCS_DIR / "04-results.md"

# --- Split policy -----------------------------------------------------------
# Reposted ads are common in this dump, so rows are grouped by a hash of the
# folded posting text and the *group* — not the row — is assigned to a split.
# Freeze these values once a model has been trained; see docs/03-protocol.md.
#
# Two stages, in this order (scheme v2, 2026-09-09):
#   1. train_pool : test = 8 : 2
#   2. train_pool -> train : dev = 9 : 1
# which lands at 72 / 8 / 20 of the rows. The stages are kept separate rather
# than collapsed into one three-way draw because the second stage must be able
# to run again on the same pool without moving a single row of test.
SPLIT_SEED = 20260826
SPLIT_TEST_FRACTION = 0.20   # stage 1
SPLIT_DEV_FRACTION = 0.10    # stage 2, taken out of what stage 1 left
SPLIT_NAMES = ("train", "dev", "test")
# The splits are plain CSV so they can be opened in anything. CSV stores no
# types, so `manifest.json` carries a `column_dtypes` map and `dataset.load_split`
# is the only supported way to read them back — see the comment above it.
SPLIT_SUFFIX = ".csv"
# Derived, recorded in the manifest — never the thing that drives the split.
SPLIT_FRACTIONS = {
    "train": round((1 - SPLIT_TEST_FRACTION) * (1 - SPLIT_DEV_FRACTION), 4),
    "dev": round((1 - SPLIT_TEST_FRACTION) * SPLIT_DEV_FRACTION, 4),
    "test": SPLIT_TEST_FRACTION,
}
RANDOM_SEED = 42

# --- Tasks ------------------------------------------------------------------
TASK_CATEGORY = "category"    # 16-class, imbalanced 27:1
TASK_SALARY = "salary"        # regression on log1p(salary_mid), disclosed subset
TASK_DISCLOSED = "disclosed"  # binary: did the ad publish a number?
TASKS = (TASK_CATEGORY, TASK_SALARY, TASK_DISCLOSED)

# Tasks whose feature matrix must read the salary-masked text mirrors.
MASKED_TASKS = frozenset({TASK_SALARY, TASK_DISCLOSED})

# Salary figures in this dataset are in millions of Vietnamese dong per month.
SALARY_UNIT = "triệu VND/month"

# Flagged, never dropped: the upper tail is genuine management pay.
SALARY_EXTREME_THRESHOLD = 100.0
# Outside these bounds a monthly figure is almost certainly the wrong unit —
# hourly/per-shift pay below, annual pay above. Measured in docs/05, §6.
SALARY_IMPLAUSIBLE_LOW = 2.0
SALARY_IMPLAUSIBLE_HIGH = 200.0

# The catch-all label. It mixes marketing/IT/manufacturing ads that belong
# elsewhere, so macro-F1 is also reported with this class excluded.
JUNK_CATEGORY = "nhóm_nghề_khác"

MAJOR_CITIES = frozenset(
    {"hà nội", "hồ chí minh", "đà nẵng", "hải phòng", "cần thơ"}
)

# --- Raw schema -------------------------------------------------------------
# Columns stored as the repr of a Python list, e.g. "['Cao đẳng', 'Đại học']".
LIST_COLUMNS = ("qualifications", "technical_skills", "soft_skills", "benefits")

# Columns that encode the salary target in one form or another. Dropped from
# every feature matrix, for both tasks, so the two models see the same inputs.
TARGET_LEAK_COLUMNS = ("salary", "salary_min", "salary_max", "salary_avg")

# Free-text columns that exist in three mirrors: raw, salary-masked, segmented.
TEXT_COLUMNS = ("job_title", "description", "requirements_text")
