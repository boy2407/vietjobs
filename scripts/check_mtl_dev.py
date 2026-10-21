"""Chấm lại một run đa nhiệm trên ``dev`` bằng đúng đường chấm của ``eval_mtl_all.py`` /
``eval_dense_all.py`` và so với ``metrics.json`` lúc train.

Lý do: Đợt 17 (2026-10-21) cho thấy một ``best.pt`` train ở môi trường này rồi chấm ở môi
trường khác cho số ``test`` phồng (PhoBERT MTL Dense: dev 0.5809 lúc train, 0.6215 khi
encode lại). Script này là cổng chặn: chỉ chấm ``test`` khi ``dev`` khớp lúc train.

    PYTHONPATH=src python scripts/check_mtl_dev.py <run_id>     # exit 1 nếu lệch
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import eval_mtl_all as M  # noqa: E402
from vietjobs import config as C, dataset as D, evaluate as E  # noqa: E402
from vietjobs.dl import encode as ENC, text as T  # noqa: E402
from vietjobs.dl.heads import MultiTaskHead, MultiTaskRnn  # noqa: E402

TOL = 1e-3  # F1 / MAE (triệu) lệch quá mức này là khác môi trường, không phải nhiễu số học


def main(rid: str) -> int:
    rd = C.ARTIFACT_DIR / rid
    cfg = json.loads((rd / "config.json").read_text(encoding="utf-8"))
    enc, L = cfg["encoder"], cfg["max_len"]
    seg = ENC.ENCODERS[enc]["segmented"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = D.load_split("dev")
    labels = sorted(set(D.load_split("train")["category"].dropna()))
    sc = np.load(rd / "scaler.npz")
    mu, sd = sc["mean"], sc["std"]

    def load(ck):
        if cfg["head"] == "rnn":
            m = MultiTaskRnn(ENC.hidden_of(enc), cfg["hidden"], len(labels), cfg["dropout"],
                             spatial_dropout=cfg.get("spatial_dropout", 0.0),
                             conv_channels=cfg["conv_channels"], branches=cfg.get("branches", "both"))
        else:
            m = MultiTaskHead(ENC.hidden_of(enc), cfg["hidden"], len(labels), cfg["dropout"],
                              pre_hidden=cfg.get("pre_hidden", 0))
        m.load_state_dict(torch.load(rd / ck, map_location=dev))
        return m.to(dev).eval()

    mc, ms = load("best.pt"), load("best_sal.pt")
    if cfg["head"] == "rnn":
        proba, lg = M.eval_rnn_on_df(mc, ms, df, C.TASK_SALARY, enc, L, seg, mu, sd, dev,
                                     layers=4 if cfg.get("last4") else 1)
    else:
        X = ENC.encode_texts(T.build_text(df, task=C.TASK_SALARY, segmented=seg), max_len=L,
                             batch=32, device=str(dev), encoder=enc, last4=bool(cfg.get("last4")))
        x = torch.from_numpy(((X - mu) / sd).astype(np.float32)).to(dev)
        with torch.no_grad():
            proba = mc(x)[0].cpu().numpy()
            lg = ms(x)[1].squeeze(-1).cpu().numpy()

    f1 = E.classification_metrics(df["category"].to_numpy(), np.array(labels)[proba.argmax(1)], labels)["f1_macro"]
    k = ((df["salary_disclosed"] == 1) & df["salary_mid"].notna()).to_numpy()
    mae = E.regression_metrics(np.log1p(df.loc[k, "salary_mid"].to_numpy(float)), lg[k])["mae_trieu"]
    old = json.loads((rd / "metrics.json").read_text(encoding="utf-8"))
    f1_0, mae_0 = old["metrics_cat"]["f1_macro"], old["metrics_sal"]["mae_trieu"]
    ok = abs(f1 - f1_0) <= TOL and abs(mae - mae_0) <= TOL
    print(f"{rid} DEV cham lai F1={f1:.4f} MAE={mae:.3f} | luc train F1={f1_0:.4f} MAE={mae_0:.3f} "
          f"-> {'KHOP' if ok else 'LECH'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
