"""Tuần 9 (B7): rút 1.000 tin từ ext37k-sal, chấm một run MTL RNN, vẽ ma trận nhầm lẫn
và biểu đồ lương, so với cùng run trên ``dev``.

Đường dự đoán giống hệt ``eval_mtl_all.py`` (cùng ``eval_rnn_on_df``, Rule 4).
Lương 37K là silver và crosswalk 60 → 16 còn draft: số liệu chỉ để phân tích.

    PYTHONPATH=src .venv-dl/bin/python scripts/analyze_ext37k_1k.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vietjobs import config as C  # noqa: E402
from vietjobs import dataset as D  # noqa: E402
from vietjobs import evaluate as E  # noqa: E402
from vietjobs.dl import encode as ENC  # noqa: E402
from vietjobs.dl.heads import MultiTaskRnn  # noqa: E402
from eval_mtl_all import load_data, eval_rnn_on_df  # noqa: E402

RUN = "dl-mtl-caf-rnn-b01-cw-s42-v2"
N = 1000
OUT = C.ARTIFACT_DIR / RUN / "external_37k" / "sample1k"


def load(run_dir: Path, ckpt: str, n_classes: int, cfg: dict) -> MultiTaskRnn:
    m = MultiTaskRnn(ENC.hidden_of(cfg["encoder"]), cfg["hidden"], n_classes, cfg["dropout"],
                     spatial_dropout=cfg.get("spatial_dropout", 0.0),
                     conv_channels=cfg["conv_channels"], branches=cfg.get("branches", "both"))
    m.load_state_dict(torch.load(run_dir / ckpt, map_location="cpu"))
    return m.eval()


def sal_block(y: np.ndarray, p: np.ndarray) -> dict:
    m = E.regression_metrics(np.log1p(y), np.log1p(p))
    m["rmse_trieu"] = float(np.sqrt(np.mean((p - y) ** 2)))
    m["within_20pct"] = float(np.mean(np.abs(p - y) <= 0.2 * y))
    m["bias_trieu"] = float(np.mean(p - y))
    bins = [0, 10, 20, 30, np.inf]
    m["by_band"] = {f"{lo}-{hi}": {"n": int(s.sum()), "mae": float(np.abs(p[s] - y[s]).mean()),
                                   "bias": float((p[s] - y[s]).mean())}
                    for lo, hi in zip(bins, bins[1:]) if (s := (y >= lo) & (y < hi)).any()}
    return m


def main() -> None:
    run_dir = C.ARTIFACT_DIR / RUN
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    labels = sorted(set(D.load_split("train")["category"].dropna()))
    _, df_ext = load_data()
    df_ext = df_ext[df_ext["strict_target"].notna()]  # đủ cả nhãn nghề lẫn lương: 2.812 tin
    df = df_ext.sample(n=N, random_state=C.SPLIT_SEED).reset_index(drop=True)

    proba, pred_log = _predict(run_dir, cfg, df, labels)
    OUT.mkdir(parents=True, exist_ok=True)
    pred = pd.DataFrame({"id": df["id"], "job_title": df["job_title"],
                         "strict_target": df["strict_target"],
                         "y_pred_cat": np.array(labels)[proba.argmax(1)],
                         "y_true_sal": df["salary_mid"].astype(float),
                         "y_pred_sal": np.expm1(pred_log)})
    pred.to_parquet(OUT / "predictions.parquet")

    dev = pd.read_parquet(run_dir / "predictions_dev.parquet")
    s = pred["strict_target"].notna()
    dv = dev["y_true_sal"].notna()
    res = {"run_id": RUN, "n_sample": N, "seed": C.SPLIT_SEED,
           "ext_cat": E.classification_metrics(pred.loc[s, "strict_target"].to_numpy(),
                                               pred.loc[s, "y_pred_cat"].to_numpy(), labels),
           "dev_cat": E.classification_metrics(dev["y_true_cat"].to_numpy(),
                                               dev["y_pred_cat"].to_numpy(), labels),
           "ext_sal": sal_block(pred["y_true_sal"].to_numpy(), pred["y_pred_sal"].to_numpy()),
           "dev_sal": sal_block(dev.loc[dv, "y_true_sal"].to_numpy(),
                                dev.loc[dv, "y_pred_sal"].to_numpy())}
    res["ext_cat"]["n"], res["dev_cat"]["n"] = int(s.sum()), len(dev)
    (OUT / "metrics.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float),
                                      encoding="utf-8")
    print(json.dumps({k: {m: v for m, v in r.items() if not isinstance(v, (dict, list))}
                      for k, r in res.items() if isinstance(r, dict)}, indent=1, default=float))


def _predict(run_dir: Path, cfg: dict, df: pd.DataFrame, labels: list):
    sc = np.load(run_dir / "scaler.npz")
    return eval_rnn_on_df(load(run_dir, "best.pt", len(labels), cfg),
                          load(run_dir, "best_sal.pt", len(labels), cfg),
                          df, C.TASK_SALARY, cfg["encoder"], cfg["max_len"],
                          ENC.ENCODERS[cfg["encoder"]]["segmented"], sc["mean"], sc["std"],
                          torch.device("cpu"), layers=4 if cfg.get("last4") else 1)


if __name__ == "__main__":
    main()
