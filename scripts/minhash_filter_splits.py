"""v7a: lọc thêm bằng MinHash trên một bộ chia đã có, giữ nguyên tập của từng dòng được giữ lại.

Văn bản = job_title . description . requirements_text (đã làm sạch), chữ thường; shingle 5 ký tự,
128 hoán vị, ngưỡng Jaccard 0,9; các cặp giống nhau nối bắc cầu thành cụm. Cụm có >= 2 nhãn bị bỏ hẳn,
mỗi cụm còn lại giữ dòng có chỉ số thấp nhất trong file gốc. Vector đã mã hoá của bộ nguồn được cắt lại
theo các dòng giữ, không mã hoá lại.
    PYTHONPATH=src .venv/bin/python scripts/minhash_filter_splits.py \\
        --src splits-dedup-norm-1label --out splits-dedup-norm-1label-mh
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from datasketch import MinHash, MinHashLSH

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vietjobs import config as C  # noqa: E402
from vietjobs import dataset as D  # noqa: E402
from encode_minhash_splits import base_rows  # noqa: E402

EMB = C.ARTIFACT_DIR / "embeddings"
VEC = "cafebert-{}-raw-len512.npy"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    src_dir, out_dir = C.PROCESSED_DIR / args.src, C.PROCESSED_DIR / args.out

    parts = []
    for split in C.SPLIT_NAMES:
        d = pd.read_csv(D.split_path(split, src_dir), dtype=str, keep_default_na=False)
        d["_split"], d["_pos"] = split, range(len(d))
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    raw_dd = D.load_raw(C.RAW_CSV).drop_duplicates().reset_index(drop=True)
    df["_base"] = base_rows(df, raw_dd)

    text = (df.job_title.str.strip().str.cat(df.description.str.strip(), sep=" . ")
            .str.cat(df.requirements_text.str.strip(), sep=" . ")
            .str.replace(r"\s+", " ", regex=True).str.strip().str.lower()).tolist()
    lsh, sigs = MinHashLSH(threshold=0.9, num_perm=128), []
    for i, t in enumerate(text):
        m = MinHash(num_perm=128)
        for j in range(max(1, len(t) - 4)):
            m.update(t[j:j + 5].encode())
        sigs.append(m)
        lsh.insert(i, m)
    par = list(range(len(text)))

    def find(x: int) -> int:
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    pairs = 0
    for i, m in enumerate(sigs):
        for j in lsh.query(m):
            if j > i:
                pairs += 1
                par[find(i)] = find(j)
    df["_cl"] = [find(i) for i in range(len(text))]
    n_lab = df.groupby("_cl")["category"].nunique()
    n_size = df.groupby("_cl").size()
    single = df["_cl"].map(n_lab) == 1
    first = df[single].sort_values("_base").drop_duplicates("_cl")
    keep = df.loc[first.index]

    out_dir.mkdir(parents=True, exist_ok=True)
    (EMB / args.out).mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in C.SPLIT_NAMES:
        k = keep[keep["_split"] == split].sort_values("_pos")
        k.drop(columns=[c for c in k.columns if c.startswith("_")]).to_csv(D.split_path(split, out_dir), index=False)
        X = np.load(EMB / args.src / VEC.format(split), mmap_mode="r")
        np.save(EMB / args.out / VEC.format(split), np.asarray(X[k["_pos"].to_numpy()]))
        counts[split] = int(len(k))
    manifest = {
        "source": args.src, "rows_in": int(len(df)), "pairs": pairs, "clusters": int(len(n_size)),
        "clusters_ge2_rows": int((n_size >= 2).sum()), "clusters_ge2_labels": int((n_lab >= 2).sum()),
        "rows_in_ge2_label_clusters": int(n_size[n_lab >= 2].sum()), "rows_out": int(len(keep)),
        "rule": "MinHash clusters (5-char shingles, 128 perms, Jaccard >= 0.9); clusters with >= 2 labels dropped; "
                "lowest raw row kept per cluster; split of the kept row unchanged",
        "splits": counts,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
