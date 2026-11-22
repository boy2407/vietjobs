"""Hình cho ``analyze_ext37k_1k.py``: ma trận nhầm lẫn (1k ext37k và dev) và lương thật ↔ dự đoán.

    .venv/bin/python scripts/plot_ext37k_1k.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vietjobs import config as C  # noqa: E402

RUN = "dl-mtl-caf-rnn-b01-cw-s42-v2"
OUT = C.ROOT / "docs" / "figures"


def short(c: str) -> str:
    return "_".join(c.split("_")[:2])


def cm(y, p, labels, title, path):
    m = pd.crosstab(pd.Categorical(y, labels), pd.Categorical(p, labels), dropna=False).to_numpy()
    norm = m / np.maximum(m.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7, 6.3))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if m[i, j]:
                ax.text(j, i, m[i, j], ha="center", va="center", fontsize=6,
                        color="white" if norm[i, j] > 0.5 else "black")
    names = [short(c) for c in labels]
    ax.set_xticks(range(len(labels)), names, rotation=90, fontsize=7)
    ax.set_yticks(range(len(labels)), [f"{n} ({m[i].sum()})" for i, n in enumerate(names)], fontsize=7)
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Nhãn thật (n)")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def sal(y, p, scope, path):
    # Cùng kiểu với các hình sal-line-* đã có: tin xếp theo lương thật, chấm = dự báo.
    o = np.argsort(y, kind="stable")
    x = np.arange(len(y))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(x, p[o], s=4, alpha=0.4, color="tab:orange", label="Dự báo của mô hình")
    ax.plot(x, y[o], color="tab:blue", lw=1.5, label="Lương thật của tin")
    ax.set_yscale("log")
    ax.set_xlabel("Tin tuyển dụng (sắp xếp theo lương thật tăng dần)")
    ax.set_ylabel("Lương (triệu đồng/tháng, thang log)")
    mae = f"{np.abs(p - y).mean():.2f}".replace(".", ",")
    ax.set_title(f"Lương thật và dự báo — {RUN}, {scope} n={len(y)}, MAE={mae}", fontsize=9)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    run_dir = C.ARTIFACT_DIR / RUN
    ext = pd.read_parquet(run_dir / "external_37k" / "sample1k" / "predictions.parquet")
    dev = pd.read_parquet(run_dir / "predictions_dev.parquet")
    labels = sorted(dev["y_true_cat"].unique())
    s = ext["strict_target"].notna()
    cm(ext.loc[s, "strict_target"], ext.loc[s, "y_pred_cat"], labels,
       f"Ext37K mẫu 1.000 tin — {s.sum()} tin có nhãn strict", OUT / "t19-cm-ext37k-1k.png")
    cm(dev["y_true_cat"], dev["y_pred_cat"], labels, f"dev — {len(dev)} tin", OUT / "t19-cm-dev.png")
    sal(ext["y_true_sal"].to_numpy(), ext["y_pred_sal"].to_numpy(),
        "Ext37K mẫu 1k", OUT / "eda" / "t19-sal-line-ext37k-1k.png")
    d = dev[dev["y_true_sal"].notna()]
    sal(d["y_true_sal"].to_numpy(), d["y_pred_sal"].to_numpy(),
        "dev", OUT / "eda" / "t19-sal-line-dev.png")


if __name__ == "__main__":
    main()
