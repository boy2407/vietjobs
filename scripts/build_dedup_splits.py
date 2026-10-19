"""Bộ chia mới: một dòng cho mỗi chuỗi mà BERT đọc, ghi ra data/processed/splits-dedup/.

Chuỗi "BERT đọc" = job_title . description . requirements_text sau NFC → dấu thanh → viết tắt
(có dấu, giữ chữ hoa), ghép bằng " . ". Dòng nào trùng chuỗi đó thì giữ dòng xuất hiện đầu tiên
trong file gốc (nhãn là nhãn của dòng đó). Mỗi dòng giữ thêm 18 cột gốc nguyên bản với tiền tố raw_ (raw_salary, raw_description… còn nguyên số lương:
không đưa vào bài salary, Rule 3). Rồi chia bằng đúng thuật toán hai tầng 8:2 → 9:1 với
SPLIT_SEED, mỗi dòng là một đơn vị. Không ghi đè data/processed/splits/ (Rule 5).
    PYTHONPATH=src .venv/bin/python scripts/build_dedup_splits.py            # khoá phân biệt hoa/thường
    PYTHONPATH=src .venv/bin/python scripts/build_dedup_splits.py --ci       # bước 3: bỏ qua hoa/thường
    PYTHONPATH=src .venv/bin/python scripts/build_dedup_splits.py --norm     # bước 4: thường, bỏ dấu câu và khoảng trắng
    PYTHONPATH=src .venv/bin/python scripts/build_dedup_splits.py --norm --drop-multi   # bỏ hẳn tin có >= 2 nhãn
    PYTHONPATH=src .venv/bin/python scripts/build_dedup_splits.py --minhash --drop-multi # cụm MinHash thay khoá văn bản
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vietjobs import config as C  # noqa: E402
from vietjobs import dataset as D  # noqa: E402



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ci", action="store_true", help="xét trùng không phân biệt hoa/thường")
    ap.add_argument("--norm", action="store_true",
                    help="khoá = chữ thường, bỏ mọi dấu câu, ký tự đặc biệt và khoảng trắng")
    ap.add_argument("--drop-multi", action="store_true",
                    help="bỏ hẳn mọi tin có từ 2 nhãn trở lên thay vì giữ nhãn của dòng đầu")
    ap.add_argument("--minhash", action="store_true",
                    help="gộp các tin gần-trùng (MinHash, 5-gram ký tự, 128 hoán vị, Jaccard >= 0.9) thành một cụm")
    args = ap.parse_args()
    ci, norm, drop_multi, minhash = args.ci, args.norm, args.drop_multi, args.minhash
    OUT_DIR = C.PROCESSED_DIR / ("splits-minhash-1label" if minhash and drop_multi else "splits-minhash" if minhash
                                 else "splits-dedup-norm-1label" if drop_multi else "splits-dedup-norm" if norm
                                 else "splits-dedup-ci" if ci else "splits-dedup")
    raw = D.load_raw(C.RAW_CSV)
    df = D.clean(raw, segment=True)
    df["raw_row"] = range(len(df))  # chỉ số dòng trong khung đã làm sạch (47.707 dòng), để ánh xạ vector đã mã hoá
    # giữ nguyên 18 cột gốc (tiền tố raw_) bên cạnh các cột đã làm sạch; clean() đã drop_duplicates()
    raw_dd = raw.drop_duplicates().reset_index(drop=True)
    assert len(raw_dd) == len(df)
    for c in raw_dd.columns:
        df["raw_" + c] = raw_dd[c].to_numpy()
    key = (df["job_title"].str.strip().str.cat(df["description"].str.strip(), sep=" . ")
           .str.cat(df["requirements_text"].str.strip(), sep=" . ")
           .str.replace(r"\s+", " ", regex=True).str.strip())
    if minhash:
        from datasketch import MinHash, MinHashLSH
        text = key.str.lower().tolist()
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

        for i, m in enumerate(sigs):
            for j in lsh.query(m):
                if j > i:
                    par[find(i)] = find(j)
        key = pd.Series([find(i) for i in range(len(text))], index=key.index)
    elif norm:
        key = key.str.lower().str.replace(r"[\W_]+", "", regex=True)
    elif ci:
        key = key.str.lower()
    n_cat = df.groupby(key.values)["category"].nunique()
    multi = int((n_cat > 1).sum())
    keep = ~key.duplicated(keep="first")
    if drop_multi:
        keep &= key.map(n_cat) == 1
    df = df[keep].drop(columns=["group_id"]).reset_index(drop=True)
    key = key[keep].reset_index(drop=True)
    print(f"{len(key):,} tin khác nhau · {multi:,} tin có >= 2 nhãn (giữ nhãn của dòng đầu)")

    # thuật toán chia cần một cột nhóm: mỗi tin là một nhóm riêng
    df["group_id"] = range(len(df))
    df["split"] = D.two_stage_split(df)
    df = df.drop(columns=["group_id"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in C.SPLIT_NAMES:
        part = df[df["split"] == split].drop(columns=["split"]).reset_index(drop=True)
        part.to_csv(D.split_path(split, OUT_DIR), index=False)
        counts[split] = int(len(part))
    manifest = {
        "source_sha256": D._file_sha256(C.RAW_CSV),
        "source_rows": int(len(raw)),
        "unique_model_inputs": int(len(df)),
        "texts_with_multiple_labels": multi,
        "label_rule": "texts with >= 2 labels dropped" if drop_multi else "first row in the raw file",
        "key": "MinHash clusters (5-char shingles, 128 perms, Jaccard >= 0.9)" if minhash else "lowercase, no punctuation, no whitespace" if norm else "case-insensitive" if ci else "case-sensitive",
        "split_seed": C.SPLIT_SEED,
        "split_scheme": "two-stage: train:test = 8:2, then train:dev = 9:1; each text is its own unit",
        "splits": counts,
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(manifest)


if __name__ == "__main__":
    main()
