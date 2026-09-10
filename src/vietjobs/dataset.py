"""Clean the raw dump, derive columns and freeze the train/val/test splits.

    python -m vietjobs.dataset build

Writes ``data/processed/splits/{train,dev,test}.csv`` plus a manifest with
row counts, the source-file hash and the split seed. Re-running against the same
raw file reproduces the splits exactly.

Word segmentation (vitext 2.4) is expensive, so it runs HERE, once, and the
result is cached in the CSV files as ``*_seg`` columns. Nothing in the
training loop ever calls the segmenter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict

import numpy as np
import pandas as pd

from . import config as C
from . import vitext as V

# Every text column exists in up to four mirrors. Which one a model may read is
# decided by features.py; the salary tasks are restricted to the masked ones.
#
#   <name>            raw, normalised          (classification only)
#   <name>_seg        raw + segmented          (classification only)
#   <name>_masked     salary figures removed   (salary / disclosed)
#   <name>_masked_seg masked + segmented       (salary / disclosed)


def _file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw(path=C.RAW_CSV) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — see data/README.md")
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])


def derive_salary(df: pd.DataFrame) -> pd.DataFrame:
    """The five salary columns, from the two raw bounds. No text, no segmenter.

    Split out of ``clean`` so that ``scripts/analyze_data.py`` can describe the
    raw corpus without paying for preprocessing — and, more to the point, so the
    arithmetic behind every salary number in docs/05 is the same arithmetic the
    model trains on. Two copies of this would drift apart silently (rule 4).
    """
    smin = pd.to_numeric(df["salary_min"], errors="coerce").fillna(0.0)
    smax = pd.to_numeric(df["salary_max"], errors="coerce").fillna(0.0)
    lo, hi = np.minimum(smin, smax), np.maximum(smin, smax)  # a few rows have min > max
    one_sided = (lo == 0) & (hi > 0)                          # only one bound published
    lo = lo.where(~one_sided, hi)

    mid = (lo + hi) / 2.0
    disclosed = (mid > 0).astype(int)
    return pd.DataFrame({
        "salary_min": lo,
        "salary_max": hi,
        "salary_mid": mid,
        "salary_disclosed": disclosed,
        "salary_is_range": ((hi > lo) & (lo > 0)).astype(int),
        "salary_mid_log": np.where(disclosed == 1, np.log1p(mid), np.nan),
        # Flagged, not dropped — the tail is real (p99 = 50.0, max = 500 triệu).
        "salary_extreme": (mid >= C.SALARY_EXTREME_THRESHOLD).astype(int),
    }, index=df.index)


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------


def clean(df: pd.DataFrame, *, segment: bool = True, verbose: bool = True) -> pd.DataFrame:
    """Raw dump -> the modelling frame. See docs/01-data-audit.md for the why."""
    n0 = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    dropped_exact = n0 - len(df)

    out = pd.DataFrame(index=df.index)

    # --- free text, two variants each ---------------------------------------
    # Classification may read the posting verbatim. The salary tasks read the
    # masked mirror, because ~5% of postings restate the offered pay.
    def _base(col: str) -> pd.Series:
        return df[col].map(lambda s: V.preprocess(s, tone=True, abbrev=True, mask=False))

    def _masked(col: str) -> pd.Series:
        return df[col].map(lambda s: V.preprocess(s, tone=True, abbrev=True, mask=True))

    out["job_title"] = _base("job_title")
    out["description"] = _base("description")
    out["requirements_text"] = _base("requirements_text")

    out["job_title_masked"] = _masked("job_title")
    out["description_masked"] = _masked("description")
    out["requirements_masked"] = _masked("requirements_text")

    # --- list-ish fields flattened ------------------------------------------
    for src in C.LIST_COLUMNS:
        out[f"{src}_text"] = df[src].map(V.join_list_field)
        out[f"n_{src}"] = df[src].map(lambda v: len(V.parse_list_field(v)))
    # benefits repeat pay more than any other field (10.6% of rows) — mask it.
    out["benefits_masked"] = out["benefits_text"].map(V.mask_salary)

    # --- structured fields ---------------------------------------------------
    locs = df["location"].map(V.split_locations)
    out["location_raw"] = locs.map(lambda xs: xs[0] if xs else "unknown")
    out["province"] = df["location"].map(V.normalize_province)
    out["n_locations"] = locs.map(len)
    out["is_major_city"] = out["province"].isin(C.MAJOR_CITIES).astype(int)
    out["country"] = df["country"].map(V.normalize_unicode)

    langs = df["languages_required"].map(V.split_languages)
    out["languages_text"] = langs.map(" ".join)
    out["n_languages"] = langs.map(len)
    out["requires_english"] = langs.map(lambda xs: int(any("anh" in x for x in xs)))

    out["experience_months"] = df["experience_required"].map(V.experience_to_months)
    out["experience_raw"] = df["experience_required"].map(V.normalize_unicode)
    out["contract_type"] = df["contract_type"].map(V.normalize_unicode).replace("", "unknown")
    out["working_hours"] = df["working_hours"].map(V.normalize_unicode)
    out["has_working_hours"] = (out["working_hours"] != "").astype(int)

    out["desc_len"] = out["description"].str.len()
    out["req_len"] = out["requirements_text"].str.len()
    out["title_len"] = out["job_title"].str.len()
    # ALL-CAPS tokens in a title are strong industry markers: SEO, IT, PHP, QA.
    out["n_acronyms"] = df["job_title"].map(
        lambda s: sum(1 for t in V.normalize_unicode(s).split() if len(t) > 1 and t.isupper())
    )

    # --- targets -------------------------------------------------------------
    out["category"] = df["category"].map(V.normalize_unicode)

    for col, values in derive_salary(df).items():
        out[col] = values

    # --- grouping key: reposts must not straddle splits ---------------------
    out["group_id"] = [
        V.group_key(t, d, r)
        for t, d, r in zip(out["job_title"], out["description"], out["requirements_text"])
    ]

    # --- word segmentation (cached here, never in the training loop) --------
    if segment:
        before = V.segment_failures
        t0 = time.perf_counter()
        for src, dst in [
            ("job_title", "job_title_seg"),
            ("description", "description_seg"),
            ("requirements_text", "requirements_seg"),
            ("job_title_masked", "job_title_masked_seg"),
            ("description_masked", "description_masked_seg"),
            ("requirements_masked", "requirements_masked_seg"),
            ("benefits_masked", "benefits_masked_seg"),
            ("technical_skills_text", "technical_skills_seg"),
            ("qualifications_text", "qualifications_seg"),
        ]:
            out[dst] = V.segment_many(out[src])
            if verbose:
                print(f"  segmented {dst:28s} ({time.perf_counter() - t0:6.1f}s)", flush=True)
        out.attrs["segment_failures"] = V.segment_failures - before
        out.attrs["segment_seconds"] = round(time.perf_counter() - t0, 1)
    else:
        out.attrs["segment_failures"] = None

    out.attrs["dropped_exact_duplicates"] = dropped_exact
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def group_stratified_split(
    df: pd.DataFrame, seed: int = C.SPLIT_SEED, fractions: dict | None = None
) -> pd.Series:
    """Assign every row to one of ``fractions``' buckets — one draw, one level.

    Group-disjoint: all rows sharing a ``group_id`` land in one bucket. Reposted
    ads are common here, and a row-level split would put the same posting in
    both train and test, inflating every score.

    Stratified by ``category`` so the smallest classes survive into every bucket.
    Callers wanting the project's two-stage scheme use :func:`two_stage_split`.
    """
    fractions = fractions or C.SPLIT_FRACTIONS
    rng = np.random.default_rng(seed)

    groups = (
        df.groupby("group_id")
        .agg(n=("category", "size"), stratum=("category", lambda s: s.mode().iat[0]))
        .reset_index()
    )

    assignment: dict[str, str] = {}
    for _stratum, block in groups.groupby("stratum", sort=True):
        block = block.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1)))
        total = block["n"].sum()
        targets = {k: v * total for k, v in fractions.items()}
        filled: defaultdict[str, float] = defaultdict(float)
        # Largest groups first — they are hardest to place without overshoot.
        for gid, n in block.sort_values("n", ascending=False)[["group_id", "n"]].itertuples(
            index=False
        ):
            pick = max(fractions, key=lambda k: targets[k] - filled[k])
            assignment[gid] = pick
            filled[pick] += n

    return df["group_id"].map(assignment)


def two_stage_split(
    df: pd.DataFrame,
    seed: int = C.SPLIT_SEED,
    test_fraction: float = C.SPLIT_TEST_FRACTION,
    dev_fraction: float = C.SPLIT_DEV_FRACTION,
) -> pd.Series:
    """train:test = 8:2, then that train pool split again into train:dev = 9:1.

    Two draws, not one three-way draw. The point is that stage 2 can be redrawn
    — a different dev slice, a k-fold over the pool — without a single row of
    ``test`` moving, which is what makes ``test`` usable exactly once at the end.

    Both stages are group-disjoint and stratified by ``category``; stage 2 gets
    ``seed + 1`` so the two draws do not share a permutation.
    """
    stage1 = group_stratified_split(
        df, seed=seed, fractions={"train": 1.0 - test_fraction, "test": test_fraction}
    )
    pool = df[stage1 == "train"]
    stage2 = group_stratified_split(
        pool, seed=seed + 1, fractions={"train": 1.0 - dev_fraction, "dev": dev_fraction}
    )
    out = stage1.copy()
    out.loc[stage2.index] = stage2
    return out


def split_path(name: str, out_dir=C.SPLIT_DIR):
    return out_dir / f"{name}{C.SPLIT_SUFFIX}"


def build(raw_path=C.RAW_CSV, out_dir=C.SPLIT_DIR, seed=C.SPLIT_SEED, segment=True) -> dict:
    print(f"reading {raw_path} ...", flush=True)
    raw = load_raw(raw_path)
    print(f"  {len(raw):,} rows. cleaning{' + segmenting' if segment else ''} ...", flush=True)
    df = clean(raw, segment=segment)
    df["split"] = two_stage_split(df, seed=seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in C.SPLIT_NAMES:
        part = df[df["split"] == split].drop(columns=["split"]).reset_index(drop=True)
        part.to_csv(split_path(split, out_dir), index=False)
        counts[split] = {
            "rows": int(len(part)),
            "groups": int(part["group_id"].nunique()),
            "salary_disclosed": int(part["salary_disclosed"].sum()),
            "categories": int(part["category"].nunique()),
        }

    straddling = int((df.groupby("group_id")["split"].nunique() > 1).sum())
    assert straddling == 0, f"{straddling} groups straddle splits — split is leaking"

    manifest = {
        "source_file": str(raw_path),
        "source_sha256": _file_sha256(raw_path),
        "source_rows": int(len(raw)),
        "rows_after_exact_dedup": int(len(df)),
        "dropped_exact_duplicates": int(df.attrs.get("dropped_exact_duplicates", 0)),
        "unique_groups": int(df["group_id"].nunique()),
        "split_seed": seed,
        "split_scheme": "two-stage: train:test = 8:2, then train:dev = 9:1",
        "split_fractions": C.SPLIT_FRACTIONS,
        "split_test_fraction": C.SPLIT_TEST_FRACTION,
        "split_dev_fraction": C.SPLIT_DEV_FRACTION,
        "groups_straddling_splits": straddling,
        "segmented": bool(segment),
        "segmenter_available": V.segmenter_available(),
        "segmenter": V.segmenter_name(),
        "segment_failures": df.attrs.get("segment_failures"),
        "segment_seconds": df.attrs.get("segment_seconds"),
        "splits": counts,
        "salary_unit": C.SALARY_UNIT,
        # CSV carries no types. Without this map, load_split would have to guess,
        # and guessing is exactly what turns an empty cell into a missing one.
        "column_dtypes": {c: str(t) for c, t in df.drop(columns=["split"]).dtypes.items()},
    }
    C.MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    C.MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


# ---------------------------------------------------------------------------
# Reading a split back
# ---------------------------------------------------------------------------
#
# The splits are CSV, which stores no types and — worse — cannot tell an empty
# cell from a missing one. Both are written as two adjacent commas, and pandas
# reads both back as NaN by default. On this corpus that silently rewrites
# 4,524 cells of the dev split alone: "this posting lists no language
# requirement" and "this field was never filled in" become the same value.
#
# So the reader never guesses. ``na_filter=False`` keeps every field a string
# exactly as written, and the numeric columns are cast back from the dtype map
# the build recorded in the manifest.


def _column_dtypes(out_dir=C.SPLIT_DIR) -> dict:
    manifest = C.MANIFEST if out_dir == C.SPLIT_DIR else out_dir.parent / C.MANIFEST.name
    if not manifest.exists():
        raise FileNotFoundError(
            f"{manifest} missing — it holds the column types the CSV cannot. "
            "Run `python -m vietjobs.dataset build`."
        )
    types = json.loads(manifest.read_text(encoding="utf-8")).get("column_dtypes")
    if not types:
        raise KeyError(
            f"{manifest} has no 'column_dtypes' — it predates the CSV splits. "
            "Run `python -m vietjobs.dataset build`."
        )
    return types


def load_split(name: str, out_dir=C.SPLIT_DIR) -> pd.DataFrame:
    path = split_path(name, out_dir)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run `python -m vietjobs.dataset build`")

    types = _column_dtypes(out_dir)
    df = pd.read_csv(path, na_filter=False, dtype=str)

    unknown = [c for c in df.columns if c not in types]
    if unknown:
        raise KeyError(f"{path} has columns absent from the manifest: {unknown}")

    for col, dtype in types.items():
        if col not in df.columns or dtype in ("str", "object", "string"):
            continue
        # errors="coerce" is what restores NaN for the genuinely missing values —
        # salary_mid_log is NaN on every posting that published no pay.
        num = pd.to_numeric(df[col], errors="coerce")
        df[col] = num.astype(dtype) if num.notna().all() else num
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Build and freeze the VietJobs splits.")
    ap.add_argument("command", choices=["build"])
    ap.add_argument("--raw", default=str(C.RAW_CSV))
    ap.add_argument("--seed", type=int, default=C.SPLIT_SEED)
    ap.add_argument("--no-segment", action="store_true", help="skip word segmentation (fast, for debugging)")
    args = ap.parse_args()

    from pathlib import Path

    manifest = build(Path(args.raw), seed=args.seed, segment=not args.no_segment)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
