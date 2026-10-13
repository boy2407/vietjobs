"""Chấm một run đa nhiệm (dense) trên ``ext37k-sal.csv`` — cả hai đầu, tạm thời.

Đầu ``category``: strict/lenient như ``eval_external.py`` (crosswalk 60 → 16,
còn draft). Đầu ``salary``: lương silver đọc từ văn bản 37K (T9/T10) — **không
cùng đại lượng** với lương nền tảng của VietJobs.csv (T10.1: ~80 % khớp đúng ở
tin có ghi lương), nên ngoài MAE còn báo Spearman (thứ hạng). Văn bản đọc là
các cột ``*_masked`` của tệp, dựng giống hệt lúc train. Không ghi
``04-results.md`` (quyết định của tác giả, 2026-10-12).

    PYTHONPATH=src .venv-dl/bin/python scripts/eval_external_mtl.py \\
        --run-id dl-mtl-cafe512-a5-e80-s42,dl-mtl-cafe512-a5-e80-s43,dl-mtl-cafe512-a5-e80-s44

``--check-dev N``: mã hoá lại N tin ``dev`` theo đúng đường này và so với cache
train — kiểm tra đầu vào khớp lúc train trước khi tin số trên 37K.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vietjobs import config as C  # noqa: E402
from vietjobs import dataset as D  # noqa: E402
from vietjobs import evaluate as E  # noqa: E402
from vietjobs import external as X  # noqa: E402
from vietjobs.dl import encode as ENC  # noqa: E402
from vietjobs.dl import text as T  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_external import score as score_category  # noqa: E402

SAL_FILE = X.EXTERNAL_DIR / "ext37k-sal.csv"


def texts_for(df: pd.DataFrame, encoder: str) -> list:
    return T.build_text(df, task=C.TASK_SALARY, segmented=ENC.ENCODERS[encoder]["segmented"])


def embed(df: pd.DataFrame, encoder: str, max_len: int, force: bool) -> np.ndarray:
    path = ENC.CACHE_DIR / f"{encoder}-ext37k-sal-masked-len{max_len}.npy"
    if path.exists() and not force:
        emb = np.load(path)
        if len(emb) == len(df):
            return emb
    emb = ENC.encode_texts(texts_for(df, encoder), max_len=max_len, encoder=encoder)
    np.save(path, emb)
    return emb


def load_model(run_dir: Path, n_classes: int, which: str):
    import torch
    from vietjobs.dl.heads import MultiTaskHead, MultiTaskRnn

    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if cfg.get("head", "dense") == "rnn":
        model = MultiTaskRnn(ENC.hidden_of(cfg["encoder"]), cfg["hidden"], n_classes, cfg["dropout"],
                             spatial_dropout=cfg["spatial_dropout"],
                             conv_channels=cfg["conv_channels"], branches=cfg["branches"])
    else:
        model = MultiTaskHead(ENC.hidden_of(cfg["encoder"]), cfg["hidden"], n_classes,
                              cfg["dropout"], pre_hidden=cfg.get("pre_hidden", 0))
    model.load_state_dict(torch.load(run_dir / which, map_location="cpu"))
    model.eval()
    sc = np.load(run_dir / "scaler.npz")
    return model, sc["mean"], sc["std"], cfg


def predict(model, X_: np.ndarray, mu, sigma):
    import torch

    with torch.no_grad():
        cat, sal = model(torch.tensor((X_ - mu) / sigma, dtype=torch.float32))
    return torch.softmax(cat, dim=1).numpy(), sal.squeeze(-1).numpy()


def predict_rnn(models, texts: list, encoder: str, max_len: int, mu, sigma, batch: int = 16):
    """Head rnn: token states theo từng batch, dựng đúng như cache memmap lúc train
    (float16, pad tới ``max_len`` bằng 0, mask uint8) — không ghi cache ~6 GB."""
    import torch

    tok, enc, dev = ENC._load_model("auto", encoder)
    m_cat, m_sal = models
    probas, sals = [], []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            e = tok(texts[i:i + batch], padding="max_length", truncation=True,
                    max_length=max_len, return_tensors="pt").to(dev)
            m = e["attention_mask"]
            h = (enc(**e).last_hidden_state * m.unsqueeze(-1)).to(torch.float16).cpu().numpy()
            x = torch.from_numpy((h.astype(np.float32) - mu) / sigma)
            mk = m.to(torch.uint8).cpu()
            probas.append(torch.softmax(m_cat(x, mk)[0], dim=1).numpy())
            sals.append(m_sal(x, mk)[1].squeeze(-1).numpy())
            if (i // batch) % 20 == 0:
                print(f"  rnn {i + len(h):,}/{len(texts):,}", flush=True)
    return np.concatenate(probas), np.concatenate(sals)


def salary_metrics(y_true: np.ndarray, pred_log: np.ndarray) -> dict:
    from scipy.stats import spearmanr

    m = E.regression_metrics(np.log1p(y_true), pred_log)
    pred = np.expm1(pred_log)
    m["spearman"] = float(spearmanr(y_true, pred).statistic)
    for name, sel in (("lt30", y_true < 30), ("ge30", y_true >= 30)):
        m[f"mae_{name}"] = float(np.abs(pred[sel] - y_true[sel]).mean()) if sel.any() else None
        m[f"n_{name}"] = int(sel.sum())
    return m


def check_dev(encoder: str, max_len: int, n: int) -> None:
    dev = D.load_split("dev").head(n)
    fresh = ENC.encode_texts(texts_for(dev, encoder), max_len=max_len, encoder=encoder)
    cached = ENC.embeddings_for("dev", C.TASK_SALARY, max_len=max_len, encoder=encoder)[:n]
    cos = (fresh * cached).sum(1) / (np.linalg.norm(fresh, axis=1) * np.linalg.norm(cached, axis=1))
    print(f"check-dev: {n} tin, cosine fresh vs cache min={cos.min():.5f} mean={cos.mean():.5f}")


def check_dev_rnn(run: str, encoder: str, max_len: int, n: int) -> None:
    """Không có token cache dev ở máy này: so dự đoán với ``predictions_dev.parquet``."""
    run_dir = C.ARTIFACT_DIR / run
    dev = D.load_split("dev").head(n)
    labels = sorted(set(D.load_split("train")["category"].dropna()))
    m_cat, mu, sigma, _ = load_model(run_dir, len(labels), "best.pt")
    m_sal, _, _, _ = load_model(run_dir, len(labels), "best_sal.pt")
    proba, sal = predict_rnn((m_cat, m_sal), texts_for(dev, encoder), encoder, max_len, mu, sigma)
    ref = pd.read_parquet(run_dir / "predictions_dev.parquet").head(n)
    agree = (np.array(labels)[proba.argmax(1)] == ref["y_pred_cat"].to_numpy()).mean()
    print(f"check-dev rnn: {n} tin, category trùng predictions_dev {agree:.1%}")
    if "y_pred_sal" in ref:
        print(f"  |Δ salary| max = {np.abs(np.expm1(sal) - ref['y_pred_sal'].to_numpy()).max():.4f} tr")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True, help="một hay nhiều run, phẩy-cách")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--force-encode", action="store_true")
    ap.add_argument("--check-dev", type=int, default=0)
    args = ap.parse_args()
    runs = [r.strip() for r in args.run_id.split(",") if r.strip()]

    cfg0 = json.loads((C.ARTIFACT_DIR / runs[0] / "config.json").read_text(encoding="utf-8"))
    encoder, max_len = cfg0["encoder"], cfg0["max_len"]
    is_rnn = cfg0.get("head", "dense") == "rnn"
    if args.check_dev:
        if is_rnn:
            check_dev_rnn(runs[0], encoder, max_len, args.check_dev)
        else:
            check_dev(encoder, max_len, args.check_dev)
        return

    cw = X.load_crosswalk()
    print(f"!! crosswalk '{cw.get('_status')}', lương 37K là silver — số liệu tạm thời", flush=True)

    full = pd.read_csv(SAL_FILE, keep_default_na=False, na_values=[""])
    for col in ("labels", "label_ids"):
        full[col] = full[col].map(ast.literal_eval)
    emb_full = None if is_rnn else embed(full, encoder, max_len, args.force_encode)

    overlap, dedup = X.overlap_mask(full, X.OurIndex())
    keep = ~np.asarray(overlap)
    df, cw_report = X.apply_crosswalk(full[keep].reset_index(drop=True), cw)
    emb = None if is_rnn else emb_full[keep]
    print(f"{len(full):,} tin → trùng {int((~keep).sum())} → chấm lương {len(df):,}"
          f" · category scorable {int((df['n_targets'] > 0).sum()):,}"
          f" (strict {int(df['strict_target'].notna().sum()):,})", flush=True)

    labels = sorted(set(D.load_split("train")["category"].dropna()))
    cat_rows = (df["n_targets"] > 0).to_numpy()
    y_sal = df["salary_mid"].to_numpy(dtype=float)
    summary = {}
    for run in runs:
        run_dir = C.ARTIFACT_DIR / run
        cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        if (cfg["encoder"], cfg["max_len"]) != (encoder, max_len):
            raise SystemExit(f"{run}: encoder/max_len khác run đầu")
        m_cat, mu, sigma, _ = load_model(run_dir, len(labels), "best.pt")
        m_sal, _, _, _ = load_model(run_dir, len(labels), "best_sal.pt")
        if is_rnn:
            proba, pred_log = predict_rnn((m_cat, m_sal), texts_for(df, encoder), encoder,
                                          max_len, mu, sigma)
        else:
            proba, _ = predict(m_cat, emb, mu, sigma)
            _, pred_log = predict(m_sal, emb, mu, sigma)

        res_cat = score_category(df[cat_rows].reset_index(drop=True), proba[cat_rows],
                                 labels, args.n_boot)
        res_sal = salary_metrics(y_sal, pred_log)
        out_dir = run_dir / "external_37k" / "sal"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metrics.json").write_text(json.dumps(
            {"run_id": run, "file": SAL_FILE.name, "n_file": len(full), "dedup": dedup,
             "crosswalk": cw_report, "category": res_cat, "salary": res_sal},
            indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        pd.DataFrame({"id": df["id"], "source": df["source"],
                      "targets": df["targets"].map("|".join), "strict_target": df["strict_target"],
                      "pred_cat": np.array(labels)[proba.argmax(1)],
                      "salary_true": y_sal, "salary_pred": np.expm1(pred_log),
                      "salary_rule": df["salary_rule"]}).to_parquet(out_dir / "predictions.parquet")
        st, le = res_cat.get("strict", {}), res_cat["lenient"]
        summary[run] = {"strict_macroF1": st.get("f1_macro"), "strict_acc": st.get("accuracy"),
                        "lenient_hit": le["hit_any_target"], "sal_mae": res_sal["mae_trieu"],
                        "sal_r2log": res_sal["r2_log"], "sal_spearman": res_sal["spearman"]}
        print(f"  {run}: " + " · ".join(f"{k}={v:.4f}" for k, v in summary[run].items()), flush=True)

    s = pd.DataFrame(summary).T
    print("\nmean ± sd:\n" + (s.mean().round(4).astype(str) + " ± " + s.std().round(4).astype(str)).to_string())


if __name__ == "__main__":
    main()
