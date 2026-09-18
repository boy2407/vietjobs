"""Đánh giá chéo mô hình `category` trên VietJobs-37K — tập ngoài, không train.

Câu hỏi: mô hình học được *nghề* hay chỉ học *giọng* của VietJobs.csv? Một tập
tin tuyển dụng thu thập ở thời điểm khác, tiền xử lý khác, nhãn do người khác
gán, trả lời được câu đó — với ba điều kiện được làm ở đây: khử tin trùng với
dữ liệu của ta, ánh xạ 60 nhãn → 16 lớp, và chấm theo hai quy ước (strict:
chỉ tin đơn nhãn; lenient: trúng bất kỳ nhãn nào sau ánh xạ).

    PYTHONPATH=src .venv-dl/bin/python scripts/eval_external.py --run-id dl-cat-s2 --subset gold,test

Ghi ``artifacts/<run_id>/external_37k/<subset>/`` và một dòng ``docs/04-results.md``
cho mỗi subset (trừ khi ``--no-log``). Cache vector ở
``artifacts/embeddings/ext37k-<subset>-raw-len<max_len>.npy``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vietjobs import config as C  # noqa: E402
from vietjobs import dataset as D  # noqa: E402
from vietjobs import evaluate as E  # noqa: E402
from vietjobs import external as X  # noqa: E402
from vietjobs import vitext as V  # noqa: E402
from vietjobs.dl import encode as ENC  # noqa: E402
from vietjobs.dl import text as T  # noqa: E402


def prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Đúng ba bước `dataset.clean` làm với văn bản: preprocess → segment."""
    out = df.copy()
    for col in ("job_title", "description", "requirements_text"):
        out[col] = out[col].map(lambda s: V.preprocess(s, tone=True, abbrev=True, mask=False))
    for src, dst in [("job_title", "job_title_seg"), ("description", "description_seg"),
                     ("requirements_text", "requirements_seg")]:
        out[dst] = V.segment_many(out[src])
    return out


def embed(df: pd.DataFrame, subset: str, max_len: int, force: bool) -> np.ndarray:
    path = ENC.CACHE_DIR / f"ext37k-{subset}-raw-len{max_len}.npy"
    if path.exists() and not force:
        emb = np.load(path)
        if len(emb) == len(df):
            return emb
        print(f"  cache {path.name} có {len(emb)} dòng ≠ {len(df)} — tính lại")
    texts = T.build_text(df, task=C.TASK_CATEGORY, segmented=True)
    emb = ENC.encode_texts(texts, max_len=max_len)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, emb)
    return emb


def load_head(run_dir: Path, out_dim: int):
    import torch
    from vietjobs.dl.heads import DenseHead

    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    sc = np.load(run_dir / "scaler.npz")
    head = DenseHead(ENC.HIDDEN, cfg["hidden"], out_dim, cfg["dropout"])
    head.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu"))
    head.eval()
    return head, sc["mean"], sc["std"], cfg


def predict(head, X_: np.ndarray, mu, sigma) -> np.ndarray:
    import torch

    with torch.no_grad():
        logits = head(torch.tensor((X_ - mu) / sigma, dtype=torch.float32))
        return torch.softmax(logits, dim=1).numpy()


def score(df: pd.DataFrame, proba: np.ndarray, labels: list, n_boot: int) -> dict:
    pred = np.array(labels)[proba.argmax(1)]
    out = {}

    # strict: single external label, mapped — ordinary metrics + CI
    s = df["strict_target"].notna().to_numpy()
    if s.sum():
        y = df.loc[s, "strict_target"].to_numpy()
        m = E.classification_metrics(y, pred[s], proba[s], labels)
        idx = E.bootstrap_indices(int(s.sum()), n_boot=n_boot)
        m["f1_macro_ci"] = E.bootstrap_summary(E.bootstrap_scores(y, pred[s], idx))
        m["per_class"] = E.per_class_report(y, pred[s], labels)
        m["top_confusions"] = E.top_confusions(y, pred[s], labels, k=15)
        out["strict"] = m

    # lenient: every scorable row, credit if prediction ∈ targets
    y_len = X.lenient_truth(pred, df["targets"].tolist())
    m = E.classification_metrics(y_len, pred, proba, labels)
    m["hit_any_target"] = float(np.mean([p in t for p, t in zip(pred, df["targets"])]))
    out["lenient"] = m
    out["pred_distribution"] = pd.Series(pred).value_counts().to_dict()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", default="dl-cat-s2")
    ap.add_argument("--subset", default="gold,test",
                    help="phẩy-cách trong: gold,test,dev,train")
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--keep-overlap", action="store_true",
                    help="KHÔNG khử trùng — chỉ để đo xem trùng thổi điểm bao nhiêu")
    ap.add_argument("--force-encode", action="store_true")
    ap.add_argument("--no-log", action="store_true", help="không ghi vào 04-results.md")
    args = ap.parse_args()

    run_dir = C.ARTIFACT_DIR / args.run_id
    if not (run_dir / "best.pt").exists():
        raise SystemExit(f"không thấy {run_dir}/best.pt")

    cw = X.load_crosswalk()
    if str(cw.get("_status", "")).startswith("draft"):
        print(f"!! crosswalk còn ở trạng thái '{cw['_status']}' — số liệu chỉ để tham khảo,"
              " chưa được đưa vào báo cáo", flush=True)

    labels = sorted(set(D.load_split("train")["category"].dropna()))
    head, mu, sigma, cfg = load_head(run_dir, len(labels))
    if cfg.get("max_len", 256) != args.max_len:
        print(f"!! run được train với max_len={cfg['max_len']}, đang chấm với {args.max_len}")
    index = X.OurIndex()

    for subset in [s.strip() for s in args.subset.split(",") if s.strip()]:
        t0 = time.time()
        out_dir = run_dir / "external_37k" / subset
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[{subset}] đọc {X.SUBSETS[subset].name}", flush=True)

        df = X.load_37k(X.SUBSETS[subset])
        n_raw = len(df)
        mask, dedup = X.overlap_mask(df, index)
        if not args.keep_overlap:
            df = df[~mask].reset_index(drop=True)
        df, cw_report = X.apply_crosswalk(df, cw)
        df = df[df["n_targets"] > 0].reset_index(drop=True)
        print(f"  {n_raw:,} tin → trùng {dedup['overlap_total']:,} ({dedup['overlap_pct']}%)"
              f" → không ánh xạ được {cw_report['unscorable_all_null']:,} → chấm {len(df):,}"
              f" (strict {int(df['strict_target'].notna().sum()):,})", flush=True)

        # embedding cache is keyed on the *raw* subset; keep it aligned with ids
        full = X.load_37k(X.SUBSETS[subset])
        emb_full = embed(prepare_frame(full), subset, args.max_len, args.force_encode)
        pos = pd.Series(range(len(full)), index=full["id"])
        emb = emb_full[pos.loc[df["id"]].to_numpy()]

        proba = predict(head, emb, mu, sigma)
        res = score(df, proba, labels, args.n_boot)
        seconds = time.time() - t0

        pred = np.array(labels)[proba.argmax(1)]
        pd.DataFrame({"id": df["id"], "source": df["source"], "labels": df["labels"].map("|".join),
                      "targets": df["targets"].map("|".join), "strict_target": df["strict_target"],
                      "pred": pred, "pred_conf": proba.max(1)}
                     ).to_parquet(out_dir / "predictions.parquet")
        (out_dir / "dedup_report.json").write_text(json.dumps(
            {"n_raw": n_raw, "overlap": dedup, "crosswalk": cw_report, "kept_overlap": args.keep_overlap,
             "n_scored": int(len(df))}, indent=2, ensure_ascii=False), encoding="utf-8")
        (out_dir / "metrics.json").write_text(json.dumps(
            {"run_id": args.run_id, "subset": subset, "n_raw": n_raw, "n_scored": int(len(df)),
             "crosswalk_status": cw.get("_status"), "max_len": args.max_len, "seconds": seconds,
             **res}, indent=2, ensure_ascii=False), encoding="utf-8")
        if "strict" in res:
            pd.DataFrame(res["strict"]["per_class"]).T.to_csv(out_dir / "per_class_strict.csv")
            pd.DataFrame(res["strict"]["top_confusions"]).to_csv(out_dir / "confusions_strict.csv",
                                                                index=False)

        st = res.get("strict")
        le = res["lenient"]
        headline = ""
        if st:
            ci = st["f1_macro_ci"]
            headline += (f"strict: macroF1={st['f1_macro']:.4f} [{ci['ci95_lo']:.3f},{ci['ci95_hi']:.3f}]"
                         f" · acc={st['accuracy']:.4f} · top3={st.get('top3_accuracy', float('nan')):.4f}"
                         f" (n={st['n']}) · ")
        headline += f"lenient: hit={le['hit_any_target']:.4f} · macroF1={le['f1_macro']:.4f} (n={le['n']})"
        print(f"  {headline}  ({seconds:.0f}s)")

        if not args.no_log:
            with C.RESULTS_LOG.open("a", encoding="utf-8") as fh:
                fh.write(f"| {args.run_id}-ext37k-{subset} | "
                         f"{datetime.now(timezone.utc).strftime('%m-%d %H:%M')} | category "
                         f"| phobert-frozen+dense(h={cfg['hidden']}) scored on VietJobs-37K "
                         f"| title+desc+req | segment+dedup+crosswalk({cw.get('_status')}) "
                         f"| ext37k-{subset} | {len(df)} | {headline} | {seconds:.1f}s | eval-only |\n")


if __name__ == "__main__":
    main()
