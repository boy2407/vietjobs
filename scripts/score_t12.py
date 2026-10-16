"""Chấm lại ``best.pt`` đã lưu, không train lại (T12.1; AGENTS.md Rule 5 item 6).

    PYTHONPATH=src python scripts/score_t12.py --dataset test --confirm-test --device cpu \\
        --runs dl-cat-cafe-dense-512,dl-cat-cafe-dense-512-s43

``--dataset dev`` dùng để smoke-test đường chấm (so với ``metrics.json`` của run).
``test`` cần ``--confirm-test``: chỉ *đọc* test, không ghi ``test.csv``, không ghi gì dưới
``data/``. Kết quả ghi ở ``artifacts/<run_id>/t12_score/<dataset>.json``. Bốn hàng dashboard
đang chờ tác giả (xem ``is_blocked``) bị bỏ qua.
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
from encode_t12_cpu import cache_path_t12  # noqa: E402

# Hàng dashboard chờ tác giả (tasks/T12-dashboard-map.md): không chấm.
BLOCKED_EXACT = {"dl-cat-focal-0920", "dl-cat-focal-s2", "dl-sal-0920", "dl-sal-s2"}
BLOCKED_PREFIX = ("dl-mtl-h256-a1.0-", "dl-cat-cafe-dense-512-p20-")


def is_blocked(run_id: str) -> bool:
    return run_id in BLOCKED_EXACT or run_id.startswith(BLOCKED_PREFIX)


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
        m_cat, m_sal = load("best.pt"), load("best_sal.pt")
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", required=True, help="run_id, phẩy-cách")
    ap.add_argument("--dataset", required=True, choices=["dev", "test"])
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
    for r in [r for r in runs if is_blocked(r)]:
        print(f"skip {r}: hàng chờ tác giả", flush=True)
    kinds = {r: run_kind(C.ARTIFACT_DIR / r) for r in runs if not is_blocked(r)}
    labels = sorted(set(D.load_split("train")["category"].dropna()))
    df = split_frame(args.dataset, args.limit)

    groups = defaultdict(list)
    for r, k in kinds.items():
        groups[(k["encoder"], k["max_len"], family_of(k["task"]), k["head"])].append(r)
    for (enc, L, fam, head), rs in groups.items():
        if head == "rnn":
            raise SystemExit("head rnn: chưa có (bước sau)")
        X = np.load(cache_path_t12(args.dataset, fam, enc, L))[:len(df)]
        if len(X) != len(df):
            raise SystemExit(f"cache {enc} {fam} {L}: {len(X)} dòng ≠ {len(df)}")
        for r in rs:
            fn, mu, sigma = load_predictor(C.ARTIFACT_DIR / r, kinds[r], len(labels), device)
            proba, sal = predict_dense(fn, X, mu, sigma, device)
            res = score_frame(df, proba, sal, labels)
            save(r, args.dataset, args.device, df, res)
            print(r, {t: {k: round(v, 4) for k, v in m.items() if k in ("f1_macro", "accuracy", "mae_trieu", "r2_log")}
                      for t, m in res.items()}, flush=True)


if __name__ == "__main__":
    main()
