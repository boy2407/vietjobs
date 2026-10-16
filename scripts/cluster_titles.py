"""T12.1 - how many "role" groups hide in job_title? Descriptive only.

Scope: original file after exact de-duplication (47,707 rows, Rule 8).
No label is changed and no split is read. Frozen CafeBERT, mean pooling,
KMeans for k in 6..20, silhouette (cosine on L2-normalised vectors) per k.

    PYTHONPATH=src .venv-dl/bin/python scripts/cluster_titles.py
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from vietjobs import config as C
from vietjobs import dataset as D

MODEL = "uitnlp/CafeBERT"
KS = range(6, 21)
SEED = 20260826
SAMPLE_N = 10
OUT = C.ARTIFACT_DIR / "eda" / "title_clusters.json"
CACHE = C.ARTIFACT_DIR / "eda" / "title_vectors_cafebert.npy"


def encode(titles: list[str], batch: int = 128) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    mod = AutoModel.from_pretrained(MODEL).to(dev).eval()
    uniq = sorted(set(titles), key=len)
    vecs = {}
    with torch.no_grad():
        for i in range(0, len(uniq), batch):
            chunk = uniq[i:i + batch]
            enc = tok(chunk, padding=True, truncation=True, max_length=32, return_tensors="pt").to(dev)
            h = mod(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            v = ((h * m).sum(1) / m.sum(1)).cpu().numpy()
            vecs.update(zip(chunk, v))
    return np.stack([vecs[t] for t in titles]).astype("float32")


def main() -> None:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    raw = D.load_raw().drop_duplicates().reset_index(drop=True)
    df = pd.DataFrame({"title": raw["job_title"].fillna("").str.strip(),
                       "category": raw["category"].str.strip()})
    df.loc[df["title"] == "", "title"] = "(trống)"
    if CACHE.exists():
        X = np.load(CACHE)
    else:
        t0 = time.time()
        X = encode(df["title"].tolist())
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.save(CACHE, X)
        print(f"encoded {len(X)} in {time.time() - t0:.0f}s")
    assert len(X) == len(df)
    X = X / np.linalg.norm(X, axis=1, keepdims=True).clip(1e-9)

    rng = np.random.RandomState(SEED)
    sub = rng.choice(len(X), size=min(10000, len(X)), replace=False)
    sil, labels = {}, {}
    for k in KS:
        lab = KMeans(n_clusters=k, n_init=5, random_state=SEED).fit_predict(X)
        labels[k] = lab
        sil[k] = float(silhouette_score(X[sub], lab[sub], metric="cosine"))
        print(f"k={k:2d} silhouette={sil[k]:.4f}")
    best = max(sil, key=sil.get)
    df["cluster"] = labels[best]

    samples = {}
    for c, g in df.groupby("cluster"):
        samples[str(c)] = {"n": int(len(g)),
                           "titles": g["title"].sample(min(SAMPLE_N, len(g)), random_state=SEED).tolist()}
    ct = pd.crosstab(df["cluster"], df["category"])
    out = {"scope": f"original file after exact de-dup ({len(df)} rows)", "model": MODEL,
           "pooling": "mean", "max_length": 32, "seed": SEED,
           "silhouette": "cosine, 10000-row random subsample, same subsample for every k",
           "silhouette_by_k": {str(k): v for k, v in sil.items()}, "best_k": best,
           "samples": samples,
           "crosstab": {str(c): {k: int(v) for k, v in r.items()} for c, r in ct.iterrows()}}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("best k", best, sil[best])


if __name__ == "__main__":
    main()
