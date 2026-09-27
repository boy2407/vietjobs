"""Phân tích lỗi phân loại ngành trên `dev` (lược đồ v2) và lấy mẫu để đọc tay nhãn (T2.3).

Ba việc, không huấn luyện gì mới và không ghi vào ``docs/04-results.md``:

1. Đọc ``predictions_dev.parquet`` của các run đã có → per-class P/R/F1, top cặp
   nhầm, ma trận nhầm lẫn, tỉ lệ lỗi chung giữa các run → ``artifacts/eda/errors-v2.json``.
2. File dự đoán không lưu xác suất, nên với run Dense đầu tiên trong ``--runs``
   (phải là Dense) tính lại softmax từ ``best.pt`` + ``scaler.npz`` trên cache
   ``dev`` — vài giây trên CPU. Argmax phải khớp ``y_pred`` đã lưu 100 %.
3. Lấy ~100 tin mà **mọi** run đều đoán sai và run Dense đoán sai với độ tự tin
   ≥ ``--min-conf`` → ``data/review/label-audit-100.csv`` để người đọc tay.

Chỉ đọc split ``dev``; ``test`` không được mở (docs/03-protocol.md).

    PYTHONPATH=src .venv-dl/bin/python scripts/analyze_errors.py
    PYTHONPATH=src .venv/bin/python scripts/analyze_errors.py --plot-only   # vẽ hình (cần matplotlib)
    PYTHONPATH=src .venv-dl/bin/python scripts/analyze_errors.py --audit data/review/label-audit-100.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vietjobs import config as C  # noqa: E402
from vietjobs import evaluate as E  # noqa: E402

RUNS = "dl-cat-ce-cpu-0920,dl-cat-focal-cpu-0920,dl-cat-rnn-focal"
OUT_JSON = C.ARTIFACT_DIR / "eda" / "errors-v2.json"
AUDIT_JSON = C.ARTIFACT_DIR / "eda" / "label-audit.json"
SAMPLE_CSV = C.ROOT / "data" / "review" / "label-audit-100.csv"
FIG = C.DOCS_DIR / "figures" / "cat-confusion-v2.png"
VERDICTS = ("goc", "du_doan", "ca_hai", "khong_ro")


# ---------------------------------------------------------------------------
# Hàm thuần — có test trong tests/test_analyze_errors.py
# ---------------------------------------------------------------------------

def stratified_sample(err: pd.DataFrame, pairs: list[tuple[str, str]], *, n: int = 100,
                      n_pairs: int = 70, n_junk: int = 15, seed: int = C.RANDOM_SEED) -> pd.DataFrame:
    """Chọn ``n`` dòng từ ``err`` (cột ``nhan_goc``, ``du_doan``, ``p_du_doan``).

    ~``n_pairs`` chia đều cho các cặp trong ``pairs``, ~``n_junk`` có nhãn gốc là lớp
    rác, phần còn lại ngẫu nhiên trong các lỗi khác. Ô nào thiếu mẫu thì phần thiếu
    chuyển sang nhóm "khác", nên tổng luôn là ``min(n, len(err))``. Tất định theo ``seed``.
    """
    rng = np.random.default_rng(seed)
    taken: list[int] = []

    def pick(pool: pd.DataFrame, k: int, tag: str) -> None:
        pool = pool[~pool.index.isin(taken)]
        k = min(k, len(pool))
        if k <= 0:
            return
        idx = rng.choice(pool.index.to_numpy(), size=k, replace=False)
        err.loc[idx, "nhom_mau"] = tag
        taken.extend(idx.tolist())

    err = err.copy()
    err["nhom_mau"] = ""
    per_pair = n_pairs // max(len(pairs), 1)
    for a, b in pairs:
        pick(err[(err["nhan_goc"] == a) & (err["du_doan"] == b)], per_pair, f"cap:{a}->{b}")
    pick(err[err["nhan_goc"] == C.JUNK_CATEGORY], n_junk, "rac")
    pick(err, min(n, len(err)) - len(taken), "khac")
    return err.loc[taken].sort_values(["nhom_mau", "p_du_doan"], ascending=[True, False])


def audit_summary(df: pd.DataFrame, col: str = "nhan_dung") -> dict:
    """Tỉ lệ từng kết luận đọc tay, tổng và theo ``nhom_mau``. Dòng trống bị bỏ qua."""
    filled = df[df[col].astype(str).str.strip().isin(VERDICTS)]

    def rates(g: pd.DataFrame) -> dict:
        c = g[col].value_counts()
        return {"n": int(len(g)), **{v: round(float(c.get(v, 0)) / max(len(g), 1), 4) for v in VERDICTS}}

    out = {"total": rates(filled),
           "by_group": {k: rates(g) for k, g in filled.groupby("nhom_mau")}}
    other = f"{col}_2"
    if other in df.columns:  # người đọc thứ hai → Cohen κ
        from sklearn.metrics import cohen_kappa_score
        both = df[df[col].isin(VERDICTS) & df[other].isin(VERDICTS)]
        out["kappa"] = {"n": int(len(both)),
                        "value": round(float(cohen_kappa_score(both[col], both[other])), 4)
                        if len(both) else None}
    return out


# ---------------------------------------------------------------------------
# Phần chạy trên dữ liệu thật
# ---------------------------------------------------------------------------

def dense_probs(run_dir: Path, labels: list[str]) -> np.ndarray:
    import torch
    from vietjobs.dl import encode as ENC
    from vietjobs.dl.heads import DenseHead
    from vietjobs.dl.train_dl import load_task

    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if cfg.get("head", "dense") != "dense":
        raise SystemExit(f"{run_dir.name}: tính lại xác suất chỉ hỗ trợ head dense")
    X, _, _, _ = load_task("dev", C.TASK_CATEGORY, cfg["max_len"], "dense")
    sc = np.load(run_dir / "scaler.npz")
    x = torch.tensor((X - sc["mean"]) / np.clip(sc["std"], 1e-6, None), dtype=torch.float32)
    model = DenseHead(ENC.HIDDEN, cfg["hidden"], len(labels), cfg["dropout"])
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu"))
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(x), dim=1).numpy()


def diagnose(args) -> None:
    from vietjobs import dataset as D

    runs = args.runs.split(",")
    preds = {r: pd.read_parquet(C.ARTIFACT_DIR / r / "predictions_dev.parquet") for r in runs}
    dev = D.load_split("dev")
    dev = dev[dev["category"].notna()].reset_index(drop=True)  # cùng thứ tự với train_dl.load_task
    y = dev["category"].to_numpy()
    for r, p in preds.items():
        if len(p) != len(dev) or not np.array_equal(p["y_true"].to_numpy(), y):
            raise SystemExit(f"{r}: y_true không khớp split dev hiện tại")
    labels = sorted(set(y.tolist()))

    report = {"n": int(len(y)), "labels": labels, "runs": {}}
    for r, p in preds.items():
        yp = p["y_pred"].to_numpy()
        top = E.top_confusions(y, yp, labels, k=10)
        support = pd.Series(y).value_counts()
        for t in top:
            t["pct_of_true"] = round(t["n"] / int(support[t["true"]]), 4)
        report["runs"][r] = {"metrics": E.classification_metrics(y, yp, labels),
                             "per_class": E.per_class_report(y, yp, labels),
                             "top_confusions": top,
                             "confusion": E.confusion(y, yp, labels)}
    wrong = np.stack([preds[r]["y_pred"].to_numpy() != y for r in runs])
    report["shared_errors"] = {"any": int(wrong.any(0).sum()), "all": int(wrong.all(0).sum()),
                               "by_count": {int(k): int((wrong.sum(0) == k).sum())
                                            for k in range(len(runs) + 1)}}

    base = runs[0]
    prob = dense_probs(C.ARTIFACT_DIR / base, labels)
    argmax = np.array(labels)[prob.argmax(1)]
    match = float((argmax == preds[base]["y_pred"].to_numpy()).mean())
    report["recomputed_probs"] = {"run": base, "argmax_match": match,
                                  "f1_macro": E.classification_metrics(y, argmax, labels)["f1_macro"]}
    if match < 1.0:
        raise SystemExit(f"{base}: xác suất tính lại chỉ khớp {match:.4f} — dừng")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  → {OUT_JSON.relative_to(C.ROOT)}  (argmax khớp {match:.0%}, "
          f"macroF1 tính lại {report['recomputed_probs']['f1_macro']:.4f})")

    to_id = {lab: i for i, lab in enumerate(labels)}
    err = pd.DataFrame({
        "dev_row": np.arange(len(y)), "group_id": dev["group_id"].to_numpy(),
        "job_title": dev["job_title"].to_numpy(),
        "trich_mo_ta": dev["description"].fillna("").str.slice(0, 300).to_numpy(),
        "nhan_goc": y, "du_doan": argmax,
        "p_du_doan": prob.max(1).round(3),
        "p_nhan_goc": prob[np.arange(len(y)), [to_id[v] for v in y]].round(3),
        "rnn_du_doan": preds[runs[-1]]["y_pred"].to_numpy(),
    })
    err = err[wrong.all(0) & (err["p_du_doan"] >= args.min_conf)]
    pairs = [(t["true"], t["predicted"]) for t in report["runs"][base]["top_confusions"][:5]]
    sample = stratified_sample(err, pairs, n=args.sample, seed=args.seed)
    sample["nhan_dung"] = ""
    sample["ghi_chu"] = ""
    SAMPLE_CSV.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(SAMPLE_CSV, index=False, encoding="utf-8-sig")
    print(f"  → {SAMPLE_CSV.relative_to(C.ROOT)}  ({len(sample)} dòng từ {len(err)} lỗi chung, "
          f"p ≥ {args.min_conf})")


def plot() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rep = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    base = next(iter(rep["runs"]))
    cm = np.array(rep["runs"][base]["confusion"], dtype=float)
    cm = cm / cm.sum(1, keepdims=True).clip(min=1)
    short = [lab.split("_")[0] + ("_" + lab.split("_")[1] if lab.count("_") else "")
             for lab in rep["labels"]]
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(cm)):
        for j in range(len(cm)):
            if cm[i, j] >= 0.05:
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if cm[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(short)), short, rotation=70, ha="right", fontsize=7)
    ax.set_yticks(range(len(short)), short, fontsize=7)
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Nhãn gốc")
    ax.set_title(f"Ma trận nhầm lẫn chuẩn hoá theo hàng — {base}, dev n={rep['n']}", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=200)
    print(f"  → {FIG.relative_to(C.ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=RUNS, help="run đầu tiên phải là head dense")
    ap.add_argument("--sample", type=int, default=100)
    ap.add_argument("--min-conf", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=C.RANDOM_SEED)
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--audit", type=Path, default=None, help="CSV đã đọc tay → label-audit.json")
    args = ap.parse_args()
    if args.audit:
        summary = audit_summary(pd.read_csv(args.audit, encoding="utf-8-sig", dtype=str).fillna(""))
        AUDIT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary["total"], ensure_ascii=False))
        return
    if not args.plot_only:
        diagnose(args)
    try:
        plot()
    except ModuleNotFoundError:
        print("  (không có matplotlib ở môi trường này — vẽ hình bằng: "
              "PYTHONPATH=src .venv/bin/python scripts/analyze_errors.py --plot-only)")


if __name__ == "__main__":
    main()
