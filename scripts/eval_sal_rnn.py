"""Chấm một run salary RNN đơn nhiệm trên `test` (9.541) và `ext37k-sal` (6.111).

    PYTHONPATH=src python scripts/eval_sal_rnn.py dl-sal-phobert-rnn-last4-b01-cw-s42-v2

Cổng trước: chấm lại `dev` bằng đúng đường chấm này và so MAE với metrics.json (±0,001).
Lệch thì thoát mã 1, không chạm `test` — chấm khác môi trường lúc train làm số phồng (Đợt 17, 22).
Ghi artifacts/<run_id>/metrics_test.json và external_37k/metrics.json (test-informed, Rule 5 mục 6).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_mtl_all import load_data  # noqa: E402  (cùng test + ext37k đã khử trùng, cùng crosswalk)
from vietjobs import config as C  # noqa: E402
from vietjobs import dataset as D  # noqa: E402
from vietjobs import evaluate as E  # noqa: E402
from vietjobs.dl import encode as ENC  # noqa: E402
from vietjobs.dl import text as T  # noqa: E402
from vietjobs.dl.heads import BiGruLstmCnn  # noqa: E402


def predict_log(model, df, encoder, max_len, layers, mu, sigma, device, batch=32):
    texts = T.build_text(df, task=C.TASK_SALARY, segmented=ENC.ENCODERS[encoder]["segmented"])
    tok, enc, dev_hf = ENC._load_model(str(device), encoder)
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            e = tok(texts[i:i + batch], padding="max_length", truncation=True,
                    max_length=max_len, return_tensors="pt").to(dev_hf)
            m = e["attention_mask"]
            # Cùng phép tính với encode_tokens lúc train (Rule 4): nhân mask, ép float16.
            h = (ENC._hidden(enc, e, layers) * m.unsqueeze(-1)).to(torch.float16).cpu().numpy()
            x = torch.from_numpy((h.astype(np.float32) - mu) / sigma).to(device)
            out.append(model(x, m.to(torch.uint8).to(device)).squeeze(-1).cpu().numpy())
    return np.concatenate(out)


def sal_metrics(y_true, pred_log):
    pred = np.expm1(pred_log)
    r = E.regression_metrics(np.log1p(y_true), pred_log)
    r["rmse_trieu"] = float(np.sqrt(np.mean((pred - y_true) ** 2)))
    r["spearman"] = float(spearmanr(y_true, pred).statistic)
    r["within_3_trieu"] = float(np.mean(np.abs(pred - y_true) <= 3.0))
    r["within_20pct"] = float(np.mean(np.abs(pred - y_true) <= 0.20 * y_true))
    return r


def main(run_id: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = C.ARTIFACT_DIR / run_id
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    sc = np.load(run_dir / "scaler.npz")
    mu, sigma = sc["mean"], sc["std"]
    enc, max_len = cfg["encoder"], cfg["max_len"]
    layers = 4 if cfg.get("last4") else cfg.get("layers", 1)
    model = BiGruLstmCnn(ENC.hidden_of(enc), cfg["hidden"], 1, cfg["dropout"],
                         spatial_dropout=cfg["spatial_dropout"], conv_channels=cfg["conv_channels"],
                         branches=cfg["branches"]).to(device)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    model.eval()

    def keep_sal(df):
        return ((df["salary_disclosed"] == 1) & df["salary_mid"].notna()).to_numpy()

    dev = D.load_split("dev")
    k = keep_sal(dev)
    mae = sal_metrics(dev.loc[k, "salary_mid"].to_numpy(float),
                      predict_log(model, dev[k], enc, max_len, layers, mu, sigma, device))["mae_trieu"]
    want = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))["metrics"]["mae_trieu"]
    ok = abs(mae - want) <= 1e-3
    print(f"{run_id} DEV cham lai MAE={mae:.4f} | luc train MAE={want:.4f} -> {'KHOP' if ok else 'LECH'}")
    if not ok:
        sys.exit(1)

    df_test, df_ext = load_data()
    k = keep_sal(df_test)
    res_test = sal_metrics(df_test.loc[k, "salary_mid"].to_numpy(float),
                           predict_log(model, df_test[k], enc, max_len, layers, mu, sigma, device))
    (run_dir / "metrics_test.json").write_text(json.dumps({
        "run_id": run_id, "eval": "test", "n_test": int(k.sum()), "metrics_sal": res_test,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    res_ext = sal_metrics(df_ext["salary_mid"].to_numpy(float),
                          predict_log(model, df_ext, enc, max_len, layers, mu, sigma, device))
    (run_dir / "external_37k").mkdir(exist_ok=True)
    (run_dir / "external_37k" / "metrics.json").write_text(json.dumps({
        "run_id": run_id, "n": len(df_ext), "metrics_sal": res_ext,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{run_id} TEST MAE={res_test['mae_trieu']:.3f} RMSE={res_test['rmse_trieu']:.3f} "
          f"R2log={res_test['r2_log']:.4f} | EXT MAE={res_ext['mae_trieu']:.3f} "
          f"RMSE={res_ext['rmse_trieu']:.3f} R2log={res_ext['r2_log']:.4f} rho={res_ext['spearman']:.4f}")


if __name__ == "__main__":
    main(sys.argv[1])
