"""Chấm lại mô hình chốt trên ``test`` và tách phần "sạch" (không có bản gần trùng trong train).

Không train, không chọn mô hình, không sửa ``test.csv`` (Rule 5). Dùng đúng
``eval_rnn_on_df`` của ``eval_mtl_all.py`` (Rule 4). Dự đoán từng tin được lưu lại để
khỏi phải chấm lần nữa. Cần ``artifacts/eda/near_dup_test_best.npy`` từ
``scripts/near_dup_test_train.py``.

    PYTHONPATH=src .venv-dl/bin/python scripts/score_test_clean.py [dev|test]

``dev`` dùng lại ``predictions_dev.parquet`` có sẵn, không chạy mô hình. Với ``test``,
``clean_train_dev`` là phần không có bản gần trùng trong cả train lẫn dev.
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
from eval_mtl_all import eval_rnn_on_df  # noqa: E402

RUN = "dl-mtl-caf-rnn-b01-cw-s42-v2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD = 0.9  # Jaccard >= 0.9 với một tin train thì coi là "gần trùng"


def load_head(run_dir: Path, ckpt: str, cfg: dict, n_classes: int) -> MultiTaskRnn:
    m = MultiTaskRnn(ENC.hidden_of("cafebert"), cfg["hidden"], n_classes, cfg["dropout"],
                     spatial_dropout=cfg.get("spatial_dropout", 0.0),
                     conv_channels=cfg["conv_channels"], branches=cfg.get("branches", "both"))
    m.load_state_dict(torch.load(run_dir / ckpt, map_location="cpu"))
    return m.eval().to(DEVICE)


def main() -> None:
    split = sys.argv[1] if len(sys.argv) > 1 else "test"
    run_dir = C.ARTIFACT_DIR / RUN
    pred_path = run_dir / f"predictions_{split}.parquet"
    df = D.load_split(split)
    labels = sorted(set(D.load_split("train")["category"]))

    if split == "test" and not pred_path.exists():
        cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        sc = np.load(run_dir / "scaler.npz")
        m_cat = load_head(run_dir, "best.pt", cfg, len(labels))
        m_sal = load_head(run_dir, "best_sal.pt", cfg, len(labels))
        proba, pred_log = eval_rnn_on_df(m_cat, m_sal, df, C.TASK_SALARY, "cafebert",
                                         cfg["max_len"], False, sc["mean"], sc["std"],
                                         DEVICE, layers=4 if cfg["last4"] else 1)
        pd.DataFrame({"pred_cat": np.array(labels)[proba.argmax(1)],
                      "pred_log": pred_log}).to_parquet(pred_path)
    pred = pd.read_parquet(pred_path)
    if split == "dev":  # cột của train_mtl khác cột ghi ở trên
        pred = pd.DataFrame({"pred_cat": pred["y_pred_cat"], "pred_log": np.log1p(pred["y_pred_sal"])})

    near = np.load(C.ARTIFACT_DIR / "eda" / f"near_dup_{split}_best.npy") >= THRESHOLD - 1e-9
    has_sal = ((df["salary_disclosed"] == 1) & df["salary_mid"].notna()).to_numpy()
    out = {"run_id": RUN, "split": split, "jaccard_threshold": THRESHOLD}
    subsets = [("all", np.ones(len(df), bool)), ("clean", ~near), ("near_dup", near)]
    if split == "test":
        near_td = np.load(C.ARTIFACT_DIR / "eda" / "near_dup_test_best_traindev.npy") >= THRESHOLD - 1e-9
        subsets.append(("clean_train_dev", ~near_td))
    for name, keep in subsets:
        y = df["salary_mid"].to_numpy(dtype=float)
        s = keep & has_sal
        cat = E.classification_metrics(df["category"].to_numpy()[keep],
                                       pred["pred_cat"].to_numpy()[keep], labels)
        out[name] = {
            "n_cat": int(keep.sum()), "f1_macro": cat["f1_macro"],
            "n_sal": int(s.sum()),
            "mae_trieu": float(np.mean(np.abs(np.expm1(pred["pred_log"].to_numpy()[s]) - y[s]))),
        }
    (run_dir / f"metrics_{split}_clean.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
