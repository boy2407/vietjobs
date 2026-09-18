"""VietJobs-37K + a salary column read off the text → one CSV.

The release has no salary field. ``external.extract_salary`` reads the pay a
posting states in its own text (silver, regex); ``dataset.derive_salary`` turns
the two bounds into the same five columns VietJobs.csv carries. The text is
kept **unmasked** — anyone training ``salary`` on this file must mask it first
(``vitext.mask_salary``), or the model reads the answer off its input.

    PYTHONPATH=src .venv/bin/python scripts/build_ext37k.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vietjobs import dataset as D  # noqa: E402
from vietjobs import external as X  # noqa: E402

OUT = X.EXTERNAL_DIR / "ext37k.csv"
# Only the postings with a salary, each text once (gold repeats part of test).
OUT_SAL = X.EXTERNAL_DIR / "ext37k-sal.csv"
# Hand-checked sample behind the label quality claim (T9.5).
OUT_REVIEW = X.EXTERNAL_DIR / "review-salary.csv"


def build() -> pd.DataFrame:
    frames = []
    for name in ("train", "dev", "test", "gold"):
        frames.append(X.load_37k(X.SUBSETS[name]))
    df = pd.concat(frames, ignore_index=True)

    text = df["job_title"] + "\n" + df["description"] + "\n" + df["requirements_text"]
    found = text.map(X.extract_salary)
    df["salary_min"] = found.map(lambda r: r[0] if r else 0.0)
    df["salary_max"] = found.map(lambda r: r[1] if r else 0.0)
    salary = D.derive_salary(df)
    for col in salary.columns:
        df[col] = salary[col]
    df["salary_text"] = found.map(lambda r: r[2] if r else "")
    # T9.3 — why the empty ones are empty, so "no salary" is never silent.
    df["salary_note"] = ""
    empty = ~found.notna()
    df.loc[empty, "salary_note"] = text[empty].map(X.salary_note)
    df["template_id"] = X.template_ids(df["description"], df["requirements_text"])
    # Which rule read the pay — the review (T9.5) scores each rule separately.
    df["salary_rule"] = ""
    df.loc[~empty, "salary_rule"] = text[~empty].map(X.salary_rule)
    return df


def audit(df: pd.DataFrame) -> pd.DataFrame:
    """T9.1 — every row in exactly one bucket, overall and per source."""
    text = df["job_title"] + "\n" + df["description"] + "\n" + df["requirements_text"]
    bucket = text.map(X.audit_salary)
    total = len(df)
    counts = bucket.value_counts().reindex(X.BUCKETS).fillna(0).astype(int)
    print(f"rows={total}")
    for name, n in counts.items():
        print(f"  {name:22s} {n:6d}  {n / total:6.2%}")
    print(f"  {'TOTAL':22s} {counts.sum():6d}")
    print()
    print(pd.crosstab(bucket, df["source"]).reindex(X.BUCKETS).fillna(0).astype(int))
    return bucket


def review_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """T9.5 — n labelled rows to read by hand, every rule represented.

    Stratified by the rule that read the pay, not by source alone: the review
    has to say whether *each* widening rule holds, and the rules are wildly
    unequal in size (``payword`` is 5,287 rows, ``dong_range`` 47). Within a
    rule the draw is proportional to source, and one row per template group, so
    a single reposted ad cannot fill the sample.
    """
    sal = df[df["salary_disclosed"] == 1].drop_duplicates("template_id")
    per = max(n // len(X.RULES), 1)
    parts = []
    for rule in X.RULES:
        rows = sal[sal["salary_rule"] == rule]
        parts.append(rows.sample(min(per, len(rows)), random_state=seed))
    out = pd.concat(parts).sort_values(["salary_rule", "source"])
    out = out[["id", "source", "salary_rule", "template_id", "job_title",
               "salary_min", "salary_max", "salary_text",
               "description", "requirements_text"]].copy()
    out.insert(3, "verdict", "")        # correct | wrong | ambiguous
    out.insert(4, "note", "")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--audit", action="store_true",
                    help="coverage only: bucket every row by why it has no salary, write nothing")
    ap.add_argument("--review", type=int, metavar="N",
                    help=f"write {OUT_REVIEW.name}: N labelled rows to check by hand (T9.5)")
    ap.add_argument("--seed", type=int, default=20260826)
    args = ap.parse_args()

    df = build()
    if args.audit:
        audit(df)
        return
    if args.review:
        sample = review_sample(df, args.review, args.seed)
        sample.to_csv(OUT_REVIEW, index=False)
        print(f"wrote {OUT_REVIEW}  rows={len(sample)}")
        print(sample.groupby(["salary_rule", "source"]).size())
        return
    df.to_csv(args.out, index=False)
    sal = (df[df["salary_disclosed"] == 1]
           .drop_duplicates(["job_title", "description", "requirements_text"]))
    sal.to_csv(OUT_SAL, index=False)

    has = df["salary_disclosed"] == 1
    print(f"wrote {args.out}  rows={len(df)}  with salary={has.sum()} ({has.mean():.1%})")
    print(df.groupby("source")["salary_disclosed"].agg(["size", "sum", "mean"]).round(3))
    print(f"wrote {OUT_SAL}  rows={len(sal)}")
    print("salary_mid (triệu):", df.loc[has, "salary_mid"].describe().round(2).to_dict())


if __name__ == "__main__":
    main()
