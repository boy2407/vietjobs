"""Dựng bảng kết quả từ những gì đã đo — không nhập tay số nào.

    .venv/bin/python scripts/export_runs.py

Nguồn là ``docs/04-results.md`` (nhật ký append-only: mọi lần chạy đều có một
dòng ở đó) làm khung, bổ sung chi tiết từ ``artifacts/<run_id>/`` khi có:
hàm loss, gamma, cân bằng lớp, thiết bị, epoch tốt nhất, phiên bản thư viện.
Probe không sinh artifact nên chỉ có phần đọc từ nhật ký — đủ cho bảng.

Vì dựng lại từ nguồn nên workbook **không thể lệch** khỏi số thật; chạy lại bất
cứ lúc nào, và không bao giờ sửa tay tệp .xlsx (Rule 13).

Ghi ra ``data/eda_xlsx/ket_qua_chay.xlsx``:

    00_Runs      mọi lần chạy, đầy đủ cột
    01_Phan_Lop  bảng báo cáo bài phân lớp, mỗi khối một hàm loss
    02_Luong     bảng báo cáo bài ước lượng lương
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vietjobs import config as C  # noqa: E402
from vietjobs import dataset as D  # noqa: E402
from vietjobs import evaluate as E  # noqa: E402

OUT = C.ROOT / "data" / "eda_xlsx" / "ket_qua_chay.xlsx"

# Bảng báo cáo hiển thị đúng những lần chạy này. Danh sách viết tay là có chủ ý:
# bảng trong luận văn phải là một lựa chọn có trách nhiệm, không phải "lần nào
# chạy sau cùng thì lấy lần đó". Đổi baseline nghĩa là sửa đúng chỗ này.
OFFICIAL_CLS = [
    ("CrossEntropyLoss", "dl-cat-ce-cpu-0920", "dl-cat-rnn-ce", "probe-cat-0920"),
    ("FocalLoss (γ=2)", "dl-cat-focal-cpu-0920", "dl-cat-rnn-focal", "probe-cat-0920"),
]
OFFICIAL_REG = ("SmoothL1Loss (Huber)", "dl-sal-cpu-0920", "dl-sal-rnn", "probe-sal-0920")
# T8.5 — ablation nhánh, cùng cấu hình dl-cat-rnn-ce trừ --branches.
OFFICIAL_ABLATION = [
    ("cả hai nhánh", "dl-cat-rnn-ce"),
    ("chỉ Bi-GRU", "dl-cat-rnn-gru"),
    ("chỉ Bi-LSTM", "dl-cat-rnn-lstm"),
]

# Nhãn cột trong headline -> tên cột trong bảng.
_HEAD_KEYS = {
    "macroF1": "macro_F1", "F1": "F1", "acc": "acc",
    "MAE": "MAE_trieu", "RMSE": "RMSE_trieu",
    "R2log": "R2log", "R2raw": "R2raw",
    # Hai khoá dưới đây đã bị bỏ khỏi evaluate.py ngày 2026-09-20; giữ để đọc
    # được các dòng nhật ký cũ, không tính cho lần chạy mới.
    "balAcc": "balAcc_cu", "top3": "top3_cu", "MedAE": "MedAE_cu",
}


def parse_headline(text: str) -> dict:
    """``macroF1=0.6030 · F1=0.6413 · acc=0.6501`` -> dict số."""
    out = {}
    for part in text.split("·"):
        if "=" not in part:
            continue
        key, _, val = part.strip().partition("=")
        col = _HEAD_KEYS.get(key.strip())
        if col is None:
            continue
        num = re.sub(r"[^0-9.\-]", "", val.split()[0])
        if num not in ("", "-", "."):
            out[col] = float(num)
    return out


def read_log() -> list[dict]:
    rows = []
    for line in C.RESULTS_LOG.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 11 or cells[0] in ("RunID", "") or set(cells[0]) <= {"-"}:
            continue
        rows.append({
            "run_id": cells[0], "utc": cells[1], "task": cells[2], "model": cells[3],
            "eval": cells[6], "n": cells[7], "headline": cells[8],
            "seconds": cells[9], "commit": cells[10],
            **parse_headline(cells[8]),
        })
    return rows


def enrich(row: dict) -> dict:
    """Bổ sung từ artifacts/<run_id>/ — chỉ dense run mới có."""
    d = C.ARTIFACT_DIR / row["run_id"]
    try:
        met = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return row
    cfg, env = met.get("config", {}), met.get("env", {})
    m = met.get("metrics", {})
    # Bài lương luôn dùng Huber; cờ --loss bị từ chối cho task này nhưng argparse
    # vẫn lưu giá trị mặc định "ce" vào config.json, nên phải suy từ task.
    is_reg = met.get("task") == C.TASK_SALARY
    loss = "smooth_l1" if is_reg else cfg.get("loss", "ce")
    row.update({
        "loss": loss,
        "gamma": cfg.get("focal_gamma") if loss == "focal" else None,
        "class_weight": cfg.get("class_weight"),
        "device": env.get("device"), "lr": cfg.get("lr"), "hidden": cfg.get("hidden"),
        "best_epoch": met.get("best_epoch"), "epochs_run": met.get("epochs_run"),
        "torch": env.get("torch"), "numpy": env.get("numpy"),
        "pandas": env.get("pandas"), "scikit_learn": env.get("scikit_learn"),
    })
    for key, col in (("f1_macro", "macro_F1"), ("f1_weighted", "F1"),
                     ("accuracy", "acc"), ("f1_macro_no_junk", "f1_macro_no_junk"),
                     ("mae_trieu", "MAE_trieu"), ("rmse_trieu", "RMSE_trieu"),
                     ("r2_log", "R2log"), ("r2_raw", "R2raw"),
                     ("within_20pct", "within_20pct")):
        if key in m:
            row[col] = m[key]          # số đầy đủ, không phải bản làm tròn ở headline
    return row


def floors() -> dict:
    """Sàn không dùng mô hình. F1 của sàn phân lớp không nằm trong summary.json
    nên tính lại tại chỗ, bằng đúng hàm mà mọi lần chạy dùng."""
    s = json.loads((C.ARTIFACT_DIR / "eda" / "summary.json").read_text(encoding="utf-8"))
    f = s["floors"]
    dev = D.load_split("dev")
    y = dev["category"].dropna().to_numpy()
    maj = D.load_split("train")["category"].value_counts().idxmax()
    m = E.classification_metrics(y, np.full(len(y), maj), sorted(set(y)))
    return {
        "cls": {"macro-F1": m["f1_macro"], "F1": m["f1_weighted"], "acc": m["accuracy"]},
        "reg": {"MAE_trieu": f["salary_global_median"]["mae"],
                "RMSE_trieu": f["salary_global_median"]["rmse"],
                "R2log": f["salary_global_median"]["r2"]},
    }


def main() -> None:
    runs = pd.DataFrame(enrich(r) for r in read_log())
    by_id = {r["run_id"]: r for r in runs.to_dict("records")}
    fl = floors()

    cls_rows = []
    for loss_name, dense_id, rnn_id, probe_id in OFFICIAL_CLS:
        cls_rows.append({"Mô hình": loss_name, "macro-F1": None, "F1": None, "acc": None})
        cls_rows.append({"Mô hình": "Luôn đoán lớp đông nhất", **fl["cls"]})
        for label, rid in (("LogReg trên cùng bộ vector", probe_id),
                           (f"Dense ({dense_id})", dense_id),
                           (f"Bi-GRU‖Bi-LSTM→CNN ({rnn_id})", rnn_id)):
            r = by_id.get(rid, {})
            cls_rows.append({"Mô hình": label, "macro-F1": r.get("macro_F1"),
                             "F1": r.get("F1"), "acc": r.get("acc")})
        cls_rows.append({})

    loss_name, dense_id, rnn_id, probe_id = OFFICIAL_REG
    reg_rows = [{"Mô hình": loss_name}, {"Mô hình": "Đoán trung vị (13 triệu)", **fl["reg"]}]
    for label, rid in (("Ridge trên cùng bộ vector", probe_id),
                       (f"Dense ({dense_id})", dense_id),
                       (f"Bi-GRU‖Bi-LSTM→CNN ({rnn_id})", rnn_id)):
        r = by_id.get(rid, {})
        reg_rows.append({"Mô hình": label, "MAE_trieu": r.get("MAE_trieu"),
                         "RMSE_trieu": r.get("RMSE_trieu"), "R2log": r.get("R2log")})

    abl_rows = []
    for label, rid in OFFICIAL_ABLATION:
        r = by_id.get(rid, {})
        abl_rows.append({"Nhánh": label, "run_id": rid, "macro-F1": r.get("macro_F1"),
                         "F1": r.get("F1"), "acc": r.get("acc"),
                         "epoch tốt nhất": r.get("best_epoch")})

    cols = ["run_id", "utc", "task", "model", "loss", "gamma", "class_weight", "device",
            "lr", "hidden", "eval", "n", "best_epoch", "epochs_run",
            "macro_F1", "F1", "acc", "f1_macro_no_junk",
            "MAE_trieu", "RMSE_trieu", "R2log", "R2raw", "within_20pct",
            "balAcc_cu", "top3_cu", "MedAE_cu",
            "torch", "numpy", "pandas", "scikit_learn", "seconds", "commit", "headline"]
    runs = runs.reindex(columns=cols)

    # Làm tròn theo đúng quy ước đang dùng trong 04-results.md: chỉ số phân lớp
    # 4 chữ số, tiền (triệu đồng) 2 chữ số, R² 3 chữ số.
    cls = pd.DataFrame(cls_rows).round({"macro-F1": 4, "F1": 4, "acc": 4})
    reg = (pd.DataFrame(reg_rows)
           .round({"MAE_trieu": 2, "RMSE_trieu": 2, "R2log": 3})
           .rename(columns={"MAE_trieu": "MAE (triệu)", "RMSE_trieu": "RMSE",
                            "R2log": "R² (thang log)"}))
    abl = pd.DataFrame(abl_rows).round({"macro-F1": 4, "F1": 4, "acc": 4})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT, engine="openpyxl") as xl:
        runs.to_excel(xl, sheet_name="00_Runs", index=False)
        cls.to_excel(xl, sheet_name="01_Phan_Lop", index=False)
        reg.to_excel(xl, sheet_name="02_Luong", index=False)
        abl.to_excel(xl, sheet_name="03_Ablation_Nhanh", index=False)

    print(f"-> {OUT.relative_to(C.ROOT)}  ({len(runs)} lần chạy)")


if __name__ == "__main__":
    main()
