#!/usr/bin/env bash
"""Đánh giá toàn bộ 5 mô hình Đa nhiệm (MTL RNN & MTL Dense) trên tập TEST và EXT37K.

Tính toán F1-macro, Accuracy, MAE, RMSE, R2, Spearman và tự động ghi:
- artifacts/<run_id>/metrics_test.json
- artifacts/<run_id>/external_37k/metrics.json
- artifacts/eval_mtl_results_summary.json
"""
import ast
import json
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vietjobs import config as C
from vietjobs import dataset as D
from vietjobs import evaluate as E
from vietjobs import external as X
from vietjobs import vitext as V
from vietjobs.dl import encode as ENC
from vietjobs.dl import text as T
from vietjobs.dl.heads import MultiTaskRnn, MultiTaskHead

MTL_RNN_MODELS = [
    ("dl-mtl-pho-rnn-b01-cw-s42", "phobert", 256, True),
    ("dl-mtl-caf-rnn-b01-cw-s42", "cafebert", 512, False),
    ("dl-mtl-vis-rnn-b01-cw-s42", "visobert", 512, False),
    ("dl-mtl-xlm-rnn-b01-cw-s42", "xlmr", 512, False),
    ("dl-mtl-bam-rnn-b01-cw-s42", "bamibert", 1024, False),
]

def load_data():
    print("Loading Test and Ext37K data...")
    df_test = D.load_split("test")
    print(f"Test split: {len(df_test):,} rows")

    sal_file = C.ROOT / "data" / "external" / "vietjobs37k" / "ext37k-sal.csv"
    full = pd.read_csv(sal_file, keep_default_na=False, na_values=[""])
    for col in ("labels", "label_ids"):
        full[col] = full[col].map(ast.literal_eval)
    overlap, dedup = X.overlap_mask(full, X.OurIndex())
    keep = ~np.asarray(overlap)
    # strict_target không nằm trong CSV: nó đến từ crosswalk 60 → 16, như eval_external_mtl.py.
    df_ext, _ = X.apply_crosswalk(full[keep].reset_index(drop=True), X.load_crosswalk())
    print(f"Ext37K: {len(df_ext):,} rows after dedup overlap")

    # Đảm bảo phân đoạn cho PhoBERT nếu cần
    for col in ("job_title_masked", "description_masked", "requirements_masked"):
        dst = f"{col}_seg"
        if dst not in df_ext.columns:
            print(f"Segmenting {col} for Ext37K...")
            df_ext[dst] = V.segment_many(df_ext[col])

    return df_test, df_ext


def eval_rnn_on_df(m_cat, m_sal, df, task_type, encoder, max_len, segmented, mu, sigma, device, batch=32, layers=1):
    texts = T.build_text(df, task=C.TASK_SALARY, segmented=segmented)
    tok, enc, dev_hf = ENC._load_model("cuda" if torch.cuda.is_available() else "cpu", encoder)
    
    probas, sals = [], []
    t0 = time.time()
    n_total = len(texts)
    with torch.no_grad():
        for i in range(0, n_total, batch):
            e = tok(texts[i:i + batch], padding="max_length", truncation=True,
                    max_length=max_len, return_tensors="pt").to(dev_hf)
            m = e["attention_mask"]
            # Cùng hàm với encode_tokens lúc train (Rule 4): layers=4 khi run dùng last4.
            h = (ENC._hidden(enc, e, layers) * m.unsqueeze(-1)).to(torch.float16).cpu().numpy()
            x = torch.from_numpy((h.astype(np.float32) - mu) / sigma).to(device)
            mk = m.to(torch.uint8).to(device)
            
            p_cat = torch.softmax(m_cat(x, mk)[0], dim=1).cpu().numpy()
            s_sal = m_sal(x, mk)[1].squeeze(-1).cpu().numpy()
            probas.append(p_cat)
            sals.append(s_sal)
            if (i // batch) % 50 == 0 and i > 0:
                print(f"    [{i:>5}/{n_total}] inference in {time.time() - t0:.1f}s...")

    proba_cat = np.concatenate(probas, axis=0)
    pred_sals_log = np.concatenate(sals, axis=0)
    return proba_cat, pred_sals_log


def evaluate_all():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== ĐÁNH GIÁ MTL RNN TRÊN THIẾT BỊ: {device} ===")
    
    df_test, df_ext = load_data()
    train_df = D.load_split("train")
    labels = sorted(set(train_df["category"].dropna()))
    n_classes = len(labels)

    # Ground truth test
    y_true_cat_test = df_test["category"].to_numpy()
    keep_sal_test = ((df_test["salary_disclosed"] == 1) & df_test["salary_mid"].notna()).to_numpy()
    y_true_sal_test = df_test.loc[keep_sal_test, "salary_mid"].to_numpy(dtype=float)
    y_true_log_test = np.log1p(y_true_sal_test)

    # Ground truth ext37k
    keep_cat_ext = df_ext["strict_target"].notna().to_numpy()
    y_true_cat_ext = df_ext.loc[keep_cat_ext, "strict_target"].to_numpy()
    y_true_sal_ext = df_ext["salary_mid"].to_numpy(dtype=float)
    y_true_log_ext = np.log1p(y_true_sal_ext)

    results_summary = []

    # ONLY="run_a,run_b": chỉ chấm các run này — mỗi run phải chấm trên đúng máy/môi trường đã train nó.
    only = [r for r in os.environ.get("ONLY", "").split(",") if r]
    # RUN_SUFFIX="-v2": chấm bộ run train lại với cấu hình loss mới, không đụng run gốc.
    suffix = os.environ.get("RUN_SUFFIX", "")
    for run_id, encoder, max_len, segmented in MTL_RNN_MODELS:
        run_id += suffix
        if only and run_id not in only:
            continue
        run_dir = C.ARTIFACT_DIR / run_id
        if not (run_dir / "best.pt").exists():
            print(f"Skipping {run_id}: chưa có checkpoint {run_dir / 'best.pt'}")
            continue

        print(f"\n=======================================================")
        print(f"Đánh giá mô hình: {run_id} ({encoder})")
        print(f"=======================================================")
        cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        sc = np.load(run_dir / "scaler.npz")
        mu, sigma = sc["mean"], sc["std"]

        in_dim = ENC.hidden_of(encoder)
        m_cat = MultiTaskRnn(in_dim, cfg["hidden"], n_classes, cfg["dropout"],
                             spatial_dropout=cfg.get("spatial_dropout", 0.0),
                             conv_channels=cfg["conv_channels"],
                             branches=cfg.get("branches", "both")).to(device)
        m_cat.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
        m_cat.eval()

        sal_ckpt = run_dir / "best_sal.pt" if (run_dir / "best_sal.pt").exists() else run_dir / "best.pt"
        m_sal = MultiTaskRnn(in_dim, cfg["hidden"], n_classes, cfg["dropout"],
                             spatial_dropout=cfg.get("spatial_dropout", 0.0),
                             conv_channels=cfg["conv_channels"],
                             branches=cfg.get("branches", "both")).to(device)
        m_sal.load_state_dict(torch.load(sal_ckpt, map_location=device))
        m_sal.eval()

        # 1. Chấm trên Test split
        print(f"  [1/2] Đang chấm Test set (9.541 tin)...")
        proba_test, pred_log_test = eval_rnn_on_df(m_cat, m_sal, df_test, C.TASK_SALARY,
                                                    encoder, max_len, segmented, mu, sigma, device,
                                                    layers=4 if cfg.get("last4") else 1)
        pred_cat_test = np.array(labels)[proba_test.argmax(1)]
        pred_sal_test = np.expm1(pred_log_test)

        res_cat_test = E.classification_metrics(y_true_cat_test, pred_cat_test, labels)
        
        y_eval_sal_test = pred_sal_test[keep_sal_test]
        y_eval_log_test = pred_log_test[keep_sal_test]
        res_sal_test = E.regression_metrics(y_true_log_test, y_eval_log_test)
        res_sal_test["rmse_trieu"] = float(np.sqrt(np.mean((y_eval_sal_test - y_true_sal_test) ** 2)))
        res_sal_test["within_3_trieu"] = float(np.mean(np.abs(y_eval_sal_test - y_true_sal_test) <= 3.0))
        res_sal_test["within_20pct"] = float(np.mean(np.abs(y_eval_sal_test - y_true_sal_test) <= 0.20 * y_true_sal_test))

        (run_dir / "metrics_test.json").write_text(json.dumps({
            "run_id": run_id, "eval": "test", "n_test": len(df_test),
            "metrics_cat": res_cat_test, "metrics_sal": res_sal_test
        }, indent=2, ensure_ascii=False), encoding="utf-8")

        # 2. Chấm trên Ext37K
        print(f"  [2/2] Đang chấm Ext37K (6.111 tin)...")
        proba_ext, pred_log_ext = eval_rnn_on_df(m_cat, m_sal, df_ext, C.TASK_SALARY,
                                                  encoder, max_len, segmented, mu, sigma, device,
                                                    layers=4 if cfg.get("last4") else 1)
        pred_cat_ext = np.array(labels)[proba_ext.argmax(1)]
        pred_sal_ext = np.expm1(pred_log_ext)

        res_cat_ext = E.classification_metrics(y_true_cat_ext, pred_cat_ext[keep_cat_ext], labels)
        res_sal_ext = E.regression_metrics(y_true_log_ext, pred_log_ext)
        res_sal_ext["rmse_trieu"] = float(np.sqrt(np.mean((pred_sal_ext - y_true_sal_ext) ** 2)))
        res_sal_ext["spearman"] = float(spearmanr(y_true_sal_ext, pred_sal_ext).statistic)
        res_sal_ext["within_3_trieu"] = float(np.mean(np.abs(pred_sal_ext - y_true_sal_ext) <= 3.0))
        res_sal_ext["within_20pct"] = float(np.mean(np.abs(pred_sal_ext - y_true_sal_ext) <= 0.20 * y_true_sal_ext))

        ext_out = run_dir / "external_37k"
        ext_out.mkdir(parents=True, exist_ok=True)
        (ext_out / "metrics.json").write_text(json.dumps({
            "run_id": run_id, "metrics_cat_strict": res_cat_ext, "metrics_sal": res_sal_ext
        }, indent=2, ensure_ascii=False), encoding="utf-8")

        row = {
            "run_id": run_id,
            "encoder": encoder,
            "test_f1": res_cat_test["f1_macro"],
            "test_acc": res_cat_test["accuracy"],
            "test_mae": res_sal_test["mae_trieu"],
            "test_rmse": res_sal_test["rmse_trieu"],
            "test_r2_log": res_sal_test["r2_log"],
            "ext_f1": res_cat_ext["f1_macro"],
            "ext_mae": res_sal_ext["mae_trieu"],
            "ext_rmse": res_sal_ext["rmse_trieu"],
            "ext_spearman": res_sal_ext["spearman"]
        }
        results_summary.append(row)
        print(f"  -> Kết quả: Test F1={row['test_f1']:.4f}, Test MAE={row['test_mae']:.3f}tr | Ext F1={row['ext_f1']:.4f}, Ext MAE={row['ext_mae']:.3f}tr, Spearman={row['ext_spearman']:.4f}")

    with open(C.ARTIFACT_DIR / "eval_mtl_results_summary.json", "w") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    print("\n=== HOÀN TẤT CHẤM ĐIỂM MTL RNN ===")

if __name__ == "__main__":
    evaluate_all()
