"""Chấm lại ``best.pt`` đã lưu, không train lại (T12.1; AGENTS.md Rule 5 item 6).

    PYTHONPATH=src python scripts/score_t12.py --dataset test --confirm-test --device cpu \\
        --runs dl-cat-cafe-dense-512,dl-cat-cafe-dense-512-s43

``--dataset dev`` dùng để smoke-test đường chấm (so với ``metrics.json`` của run).
``test`` cần ``--confirm-test``: chỉ *đọc* test, không ghi ``test.csv``, không ghi gì dưới
``data/``. Kết quả ghi ở ``artifacts/<run_id>/t12_score/<dataset>.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vietjobs import config as C  # noqa: E402
from vietjobs import dataset as D  # noqa: E402
from vietjobs import evaluate as E  # noqa: E402
from vietjobs.dl import encode as ENC  # noqa: E402
from vietjobs.dl import text as T  # noqa: E402
from encode_t12_cpu import cache_path_t12, ext_sal_frame  # noqa: E402

def run_kind(run_dir: Path) -> dict:
    """Run cũ không có ``encoder`` (mặc định phobert); run MTL không có ``task``."""
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    mtl = (run_dir / "best_sal.pt").exists() or "alpha_sal" in cfg
    return {"cfg": cfg, "task": "mtl" if mtl else cfg["task"], "head": cfg.get("head", "dense"),
            "encoder": cfg.get("encoder", ENC.DEFAULT_ENCODER), "max_len": cfg["max_len"]}


def family_of(task: str) -> str:
    """Giống ``encode.family``: phân lớp đọc raw, lương và MTL đọc masked (Rule 3)."""
    return "raw" if task == C.TASK_CATEGORY else "masked"


def row_masks(df: pd.DataFrame) -> tuple:
    """Dòng dùng để chấm — đúng ``train_dl.load_task`` / ``train_mtl.load_mtl``."""
    cat = df["category"].notna().to_numpy()
    sal = ((df["salary_disclosed"] == 1) & df["salary_mid"].notna()).to_numpy()
    return cat, sal


def score_frame(df: pd.DataFrame, proba, sal_log, labels: list) -> dict:
    out = {}
    cat, sal = row_masks(df)
    if proba is not None:
        pred = np.array(labels)[proba.argmax(1)]
        out["category"] = E.classification_metrics(df.loc[cat, "category"].to_numpy(), pred[cat], labels)
    if sal_log is not None:
        y = np.log1p(df.loc[sal, "salary_mid"].to_numpy(dtype=np.float32))
        out["salary"] = E.regression_metrics(y, sal_log[sal])
    return out


def _make(k: dict, out_dim: int):
    from vietjobs.dl.heads import BiGruLstmCnn, DenseHead, MultiTaskHead, MultiTaskRnn

    c, d_in = k["cfg"], ENC.hidden_of(k["encoder"])
    rnn = dict(spatial_dropout=c.get("spatial_dropout", 0.2), conv_channels=c.get("conv_channels", 64),
               branches=c.get("branches", "both"))
    if k["task"] == "mtl":
        if k["head"] == "rnn":
            return MultiTaskRnn(d_in, c["hidden"], out_dim, c["dropout"], **rnn)
        return MultiTaskHead(d_in, c["hidden"], out_dim, c["dropout"], pre_hidden=c.get("pre_hidden", 0))
    if k["head"] == "rnn":
        return BiGruLstmCnn(d_in, c["hidden"], out_dim, c["dropout"], **rnn)
    return DenseHead(d_in, c["hidden"], out_dim, c["dropout"], pre_hidden=c.get("pre_hidden", 0))


def load_predictor(run_dir: Path, k: dict, n_classes: int, device):
    """Trả về ``(fn, mu, sigma)``; ``fn(x, mask) -> (cat_logits | None, sal_log | None)``."""
    import torch

    out_dim = 1 if k["task"] == C.TASK_SALARY else n_classes

    def load(which: str):
        m = _make(k, out_dim)
        m.load_state_dict(torch.load(run_dir / which, map_location="cpu"))
        return m.eval().to(device)

    sc = np.load(run_dir / "scaler.npz")
    if k["task"] == "mtl":
        # run MTL cũ (dl-mtl-h256-*) chỉ có best.pt: cả hai đầu đọc từ đó
        m_cat = load("best.pt")
        m_sal = load("best_sal.pt" if (run_dir / "best_sal.pt").exists() else "best.pt")
        fn = lambda x, mk: (m_cat(x, mk)[0], m_sal(x, mk)[1].squeeze(-1))  # noqa: E731
    else:
        m = load("best.pt")
        if k["task"] == C.TASK_SALARY:
            fn = lambda x, mk: (None, m(x, mk).squeeze(-1))  # noqa: E731
        else:
            fn = lambda x, mk: (m(x, mk), None)  # noqa: E731
    return fn, sc["mean"], sc["std"]


def predict_dense(fn, X: np.ndarray, mu, sigma, device, chunk: int = 4096):
    import torch

    cats, sals = [], []
    with torch.no_grad():
        for i in range(0, len(X), chunk):
            x = torch.tensor((X[i:i + chunk] - mu) / sigma, dtype=torch.float32, device=device)
            c, s = fn(x, None)
            if c is not None:
                cats.append(torch.softmax(c, dim=1).cpu().numpy())
            if s is not None:
                sals.append(s.cpu().numpy())
    return (np.concatenate(cats) if cats else None), (np.concatenate(sals) if sals else None)


def predict_rnn_group(preds: dict, texts: list, encoder: str, max_len: int, device, batch: int = 16):
    """Nhúng token từng batch một lần, chạy mọi run cùng encoder trên đó (không ghi token cache).

    Dựng đúng như cache lúc train: float16, pad tới ``max_len`` bằng 0, mask uint8
    (``eval_external_mtl.predict_rnn``). ``preds``: run_id → ``(fn, mu, sigma)``.
    """
    import torch

    tok, enc, dev = ENC._load_model(str(device), encoder)
    stats = {r: (torch.tensor(mu, dtype=torch.float32, device=dev), torch.tensor(sg, dtype=torch.float32, device=dev))
             for r, (_, mu, sg) in preds.items()}
    outs = {r: ([], []) for r in preds}
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            e = tok(texts[i:i + batch], padding="max_length", truncation=True,
                    max_length=max_len, return_tensors="pt").to(dev)
            m = e["attention_mask"]
            h = (enc(**e).last_hidden_state * m.unsqueeze(-1)).to(torch.float16).float()
            mk = m.to(torch.uint8)
            for r, (fn, _, _) in preds.items():
                c, sl = fn((h - stats[r][0]) / stats[r][1], mk)
                if c is not None:
                    outs[r][0].append(torch.softmax(c, dim=1).cpu().numpy())
                if sl is not None:
                    outs[r][1].append(sl.cpu().numpy())
            if (i // batch) % 50 == 0:
                print(f"  rnn {encoder} {i + len(m):,}/{len(texts):,}", flush=True)
    return {r: ((np.concatenate(c) if c else None), (np.concatenate(sl) if sl else None))
            for r, (c, sl) in outs.items()}


def split_frame(dataset: str, limit: int) -> pd.DataFrame:
    df = D.load_split(dataset)
    return df.head(limit) if limit else df


def save(run_id: str, dataset: str, device: str, df: pd.DataFrame, res: dict) -> None:
    out = C.ARTIFACT_DIR / run_id / "t12_score"
    out.mkdir(parents=True, exist_ok=True)
    tag = "test: author override 2026-10-15" if dataset == "test" else dataset
    (out / f"{dataset}.json").write_text(json.dumps(
        {"run_id": run_id, "dataset": dataset, "tag": tag, "n": len(df), "device": device, **res},
        indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def score_ext(runs: list, kinds: dict, labels: list, device, args) -> None:
    """``ext37k-sal.csv`` (T9/T10, lương silver — tạm thời): chỉ run có đầu lương (salary, mtl).

    Run chỉ phân lớp cần cache raw của ext37k-sal, tác giả chưa quyết → bỏ qua và nói rõ.
    Cách bỏ trùng + crosswalk + chỉ số giống ``eval_external_mtl.py``.
    """
    from vietjobs import external as X
    from eval_external import score as score_category
    from eval_external_mtl import salary_metrics

    for r in [r for r in kinds if kinds[r]["task"] == C.TASK_CATEGORY]:
        print(f"skip {r}: chỉ phân lớp, cần ext37k-sal raw (chưa quyết)", flush=True)
    cw, index = X.load_crosswalk(), X.OurIndex()
    groups = defaultdict(list)
    for r, k in kinds.items():
        if k["task"] != C.TASK_CATEGORY:
            groups[(k["encoder"], k["max_len"], k["head"])].append(r)
    for (enc, L, head), rs in groups.items():
        full = ext_sal_frame(ENC.ENCODERS[enc]["segmented"])
        overlap, dedup = X.overlap_mask(full, index)
        keep = ~np.asarray(overlap)
        df, cw_report = X.apply_crosswalk(full[keep].reset_index(drop=True), cw)
        preds = {r: load_predictor(C.ARTIFACT_DIR / r, kinds[r], len(labels), device) for r in rs}
        if head == "rnn":
            texts = T.build_text(df, task=C.TASK_SALARY, segmented=ENC.ENCODERS[enc]["segmented"])
            out = predict_rnn_group(preds, texts, enc, L, device)
        else:
            emb = np.load(cache_path_t12("ext37k-sal", "masked", enc, L))[keep]
            out = {r: predict_dense(fn, emb, mu, sg, device) for r, (fn, mu, sg) in preds.items()}
        cat_rows = (df["n_targets"] > 0).to_numpy()
        for r in rs:
            proba, sal = out[r]
            res = {"salary": salary_metrics(df["salary_mid"].to_numpy(dtype=float), sal), "n_file": len(full),
                   "dedup": dedup, "crosswalk": cw_report}
            if proba is not None:
                res["category"] = score_category(df[cat_rows].reset_index(drop=True), proba[cat_rows], labels, 1000)
            out_dir = C.ARTIFACT_DIR / r / "t12_score"
            out_dir.mkdir(parents=True, exist_ok=True)
            tag = "ext37k-sal: provisional, T10.1 overridden"
            (out_dir / "ext37k-sal.json").write_text(json.dumps(
                {"run_id": r, "dataset": "ext37k-sal", "tag": tag, "n": len(df), "device": args.device, **res},
                indent=2, ensure_ascii=False, default=str), encoding="utf-8")
            print(r, {"sal_mae": round(res["salary"]["mae_trieu"], 3), "spearman": round(res["salary"]["spearman"], 3)}, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", required=True, help="run_id, phẩy-cách")
    ap.add_argument("--dataset", required=True, choices=["dev", "test", "ext37k-sal"])
    ap.add_argument("--device", default="cpu", help="Dense: cpu (Rule 13); RNN: cuda")
    ap.add_argument("--limit", type=int, default=0, help="chỉ cho smoke test trên dev")
    ap.add_argument("--confirm-test", action="store_true")
    args = ap.parse_args()
    if args.dataset == "test" and not args.confirm_test:
        raise SystemExit("chấm test cần --confirm-test (docs/03-protocol.md, điều 2; Rule 5 item 6)")
    if args.limit and args.dataset == "test":
        raise SystemExit("--limit chỉ dùng với dev")

    device = ENC.pick_device(args.device)
    runs = [r.strip() for r in args.runs.split(",") if r.strip()]
    kinds = {r: run_kind(C.ARTIFACT_DIR / r) for r in runs}
    labels = sorted(set(D.load_split("train")["category"].dropna()))
    if args.dataset == "ext37k-sal":
        return score_ext(runs, kinds, labels, device, args)
    df = split_frame(args.dataset, args.limit)

    groups = defaultdict(list)
    for r, k in kinds.items():
        groups[(k["encoder"], k["max_len"], family_of(k["task"]), k["head"])].append(r)
    for (enc, L, fam, head), rs in groups.items():
        preds = {r: load_predictor(C.ARTIFACT_DIR / r, kinds[r], len(labels), device) for r in rs}
        if head == "rnn":
            task = C.TASK_SALARY if fam == "masked" else C.TASK_CATEGORY
            texts = T.build_text(df, task=task, segmented=ENC.ENCODERS[enc]["segmented"])
            out = predict_rnn_group(preds, texts, enc, L, device)
        else:
            X = np.load(cache_path_t12(args.dataset, fam, enc, L))[:len(df)]
            if len(X) != len(df):
                raise SystemExit(f"cache {enc} {fam} {L}: {len(X)} dòng ≠ {len(df)}")
            out = {r: predict_dense(fn, X, mu, sigma, device) for r, (fn, mu, sigma) in preds.items()}
        for r in rs:
            proba, sal = out[r]
            res = score_frame(df, proba, sal, labels)
            save(r, args.dataset, args.device, df, res)
            print(r, {t: {k: round(v, 4) for k, v in m.items() if k in ("f1_macro", "accuracy", "mae_trieu", "r2_log")}
                      for t, m in res.items()}, flush=True)


if __name__ == "__main__":
    main()
