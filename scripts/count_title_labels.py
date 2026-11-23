"""Count the category labels carried by postings whose title contains a phrase.

Measured on the original file after exact de-duplication (Rule 8), with titles
NFC-normalised and lower-cased so that composed and decomposed spellings match.
Read-only: writes artifacts/eda/title_labels.json.

    .venv/bin/python scripts/count_title_labels.py "kinh doanh bất động sản"
"""
from __future__ import annotations

import json
import sys
import unicodedata
from pathlib import Path

import pandas as pd

RAW = Path("data/raw/VietJobs.csv")
OUT = Path("artifacts/eda/title_labels.json")


def main(phrases: list[str]) -> None:
    df = pd.read_csv(RAW).drop_duplicates()
    titles = df["job_title"].fillna("").map(lambda s: unicodedata.normalize("NFC", s).lower())
    out = {"source": str(RAW), "rows_after_exact_dedup": len(df), "phrases": {}}
    for p in phrases:
        hit = df[titles.str.contains(unicodedata.normalize("NFC", p).lower(), regex=False)]
        out["phrases"][p] = {"n": len(hit), "labels": hit["category"].value_counts().to_dict()}
        print(p, len(hit), hit["category"].value_counts().head(4).to_dict())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
