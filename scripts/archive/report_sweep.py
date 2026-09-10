"""Turn the sweep artifacts into the tables for docs/10-so-sanh-mo-hinh.md.

    PYTHONPATH=src python scripts/report_sweep.py > /tmp/report.md

Prints markdown to stdout and writes nothing into docs/. That follows
measure_vitext.py: 04-results.md is the only file a program appends to, and
every other note is edited by a person who has read the numbers.

The comparison is *paired*. One bootstrap index matrix is built once and every
model is scored through it, so "does A beat B" is answered by resampling the
same val rows for both and looking at the sign of the difference. Comparing
independent error bars instead would ask a much weaker question, because both
models make most of their mistakes on the same ambiguous postings.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vietjobs import config as C          # noqa: E402
from vietjobs import evaluate as E        # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
N_BOOT = 1000

# Configuration #1 of these six reproduces a row that already existed before the
# sweep. If one of them misses, something changed underneath and the whole sweep
# is suspect. Families added later have no prior row to anchor against, so they
# are simply absent here — the model list itself is discovered from artifacts,
# never hard-coded, so a new family joins every table by being run.
ANCHORS = {"knn": 0.4059, "svm": 0.6050, "logreg": 0.6038,
           "rf": 0.5630, "lgbm": 0.5555, "xgb": 0.5861}


def load_runs() -> list[dict]:
    runs = []
    for d in sorted(ARTIFACTS.glob("cat-SW-*")):
        mj, pz = d / "metrics.json", d / "predictions.npz"
        if not (mj.exists() and pz.exists()):
            continue
        meta = json.loads(mj.read_text(encoding="utf-8"))
        npz = np.load(pz, allow_pickle=True)
        runs.append({
            "run_id": meta["run_id"],
            "model": meta["model"],
            "n": int(meta["run_id"].rsplit("-", 1)[1]),
            "label": meta.get("model_label", meta["model"]),
            "cfg": ",".join(f"{k}={v}" for k, v in meta.get("overrides", {}).items()),
            "f1": meta["metrics"]["f1_macro"],
            "f1_nojunk": meta["metrics"].get("f1_macro_no_junk"),
            "bal": meta["metrics"]["balanced_accuracy"],
            "acc": meta["metrics"]["accuracy"],
            "top3": meta["metrics"].get("top3_accuracy"),
            "fit": meta["fit_seconds"],
            "row_index": npz["row_index"],
            "y_true": npz["y_true"],
            "y_pred": npz["y_pred"],
        })
    return runs


def check_alignment(runs) -> None:
    """Two runs may only be subtracted if they scored the same rows."""
    ref = runs[0]
    for r in runs[1:]:
        if not np.array_equal(r["row_index"], ref["row_index"]):
            raise SystemExit(f"{r['run_id']} scored different rows than {ref['run_id']}")
        if not np.array_equal(r["y_true"], ref["y_true"]):
            raise SystemExit(f"{r['run_id']} has different labels than {ref['run_id']}")


def num(x, nd=4):
    """Vietnamese decimal comma — the convention every table in docs/ uses."""
    return "—" if x is None else f"{x:.{nd}f}".replace(".", ",")


def signed(x, nd=4):
    return f"{x:+.{nd}f}".replace(".", ",")


def secs(s):
    return f"{s / 60:.1f} ph".replace(".", ",") if s >= 90 else f"{s:.0f}s"


def main() -> None:
    runs = load_runs()
    if not runs:
        raise SystemExit("no cat-SW-* runs with predictions.npz found")
    check_alignment(runs)
    print(f"<!-- {len(runs)} runs · bootstrap {N_BOOT} · shared index matrix -->\n")

    idx = E.bootstrap_indices(len(runs[0]["y_true"]), n_boot=N_BOOT, seed=C.RANDOM_SEED)
    for r in runs:
        r["boot"] = E.bootstrap_scores(r["y_true"], r["y_pred"], idx)
        r["sigma"] = E.bootstrap_summary(r["boot"])["std"]

    # Discovered from what actually ran, ordered by best score so the tables read
    # top-down. A family that was never run simply does not appear.
    names = sorted({r["model"] for r in runs},
                   key=lambda m: -max(r["f1"] for r in runs if r["model"] == m))
    by_model = {m: sorted([r for r in runs if r["model"] == m], key=lambda r: r["n"])
                for m in names}

    # --- 0 · reproducibility ------------------------------------------------
    print("## Bảng 0 — tái lập\n")
    print("| Mô hình | mỏ neo (dòng cũ) | đo lại | lệch |")
    print("|---|---|---|---|")
    for m, rs in by_model.items():
        first = next((r for r in rs if r["n"] == 1), None)
        if first is None or m not in ANCHORS:
            continue  # families added after the anchors were set
        d = first["f1"] - ANCHORS[m]
        flag = " ✅" if abs(d) < 5e-4 else " ⚠️"
        print(f"| `{m}` | {num(ANCHORS[m])} | {num(first['f1'])} | {signed(d)}{flag} |")
    print()

    # --- 1 · every run ------------------------------------------------------
    print("## Bảng 1 — toàn bộ cấu hình\n")
    print("| Mô hình | Cấu hình | macro-F1 | σ | F1@15 | balAcc | top3 | fit |")
    print("|---|---|---|---|---|---|---|---|")
    for m, rs in by_model.items():
        best = max(r["f1"] for r in rs)
        for r in rs:
            b = "**" if r["f1"] == best else ""
            print(f"| `{m}` | `{r['cfg']}` | {b}{num(r['f1'])}{b} | ±{num(r['sigma'])} | "
                  f"{num(r['f1_nojunk'])} | {num(r['bal'])} | {num(r['top3'])} | {secs(r['fit'])} |")
    print()

    # --- 2 · champions ------------------------------------------------------
    champs = {m: max(rs, key=lambda r: r["f1"]) for m, rs in by_model.items()}
    order = sorted(champs.values(), key=lambda r: -r["f1"])
    king = order[0]
    print("## Bảng 2 — nhà vô địch từng mô hình\n")
    print("| Mô hình | Cấu hình thắng | macro-F1 | Δ so quán quân | P(Δ>0) | fit |")
    print("|---|---|---|---|---|---|")
    for r in order:
        if r is king:
            print(f"| **`{r['model']}`** | `{r['cfg']}` | **{num(r['f1'])}** | — | — | {secs(r['fit'])} |")
            continue
        d = E.paired_delta(r["boot"], king["boot"])
        print(f"| `{r['model']}` | `{r['cfg']}` | {num(r['f1'])} | {signed(d['mean'])} | "
              f"{num(d['p_gt_0'], 3)} | {secs(r['fit'])} |")
    print()

    # --- 3 · pairwise matrix ------------------------------------------------
    print("## Bảng 3 — P(hàng > cột), paired bootstrap\n")
    names = [r["model"] for r in order]
    print("| | " + " | ".join(f"`{n}`" for n in names) + " |")
    print("|---" * (len(names) + 1) + "|")
    for a in order:
        cells = []
        for b in order:
            cells.append("—" if a is b
                         else num(E.paired_delta(a["boot"], b["boot"])["p_gt_0"], 3))
        print(f"| `{a['model']}` | " + " | ".join(cells) + " |")
    print("\n> 0,500 = không phân biệt được. >0,975 hoặc <0,025 = khác biệt rõ.\n")

    # --- 4 · score per unit of cost ----------------------------------------
    print("## Bảng 4 — điểm trên mỗi phút huấn luyện\n")
    print("| Mô hình | macro-F1 | fit | F1 / phút | so với rẻ nhất |")
    print("|---|---|---|---|---|")
    cheapest = min(order, key=lambda r: r["fit"])
    for r in sorted(order, key=lambda r: r["fit"]):
        print(f"| `{r['model']}` | {num(r['f1'])} | {secs(r['fit'])} | "
              f"{num(r['f1'] / (r['fit'] / 60), 3)} | {num(r['fit'] / cheapest['fit'], 1)}× |")
    print()

    # --- 5 · sensitivity ----------------------------------------------------
    print("## Bảng 5 — độ nhạy siêu tham số\n")
    print("Biên độ trong 6 cấu hình. Biên độ rộng nghĩa là mô hình phụ thuộc nặng")
    print("vào việc chỉnh tay — và con số của nhà vô địch lạc quan hơn tương ứng,")
    print("vì nó là cái tốt nhất trong 6 lần thử **chọn trên chính tập val**.\n")
    print("| Mô hình | tệ nhất | trung vị | tốt nhất | biên độ |")
    print("|---|---|---|---|---|")
    for m, rs in by_model.items():
        f = sorted(r["f1"] for r in rs)
        med = f[len(f) // 2] if len(f) % 2 else (f[len(f) // 2 - 1] + f[len(f) // 2]) / 2
        print(f"| `{m}` | {num(f[0])} | {num(med)} | {num(f[-1])} | {num(f[-1] - f[0])} |")
    print()

    # --- 6 · sigma reference ------------------------------------------------
    print("## Bảng 6 — σ đối chiếu\n")
    svm1 = next((r for r in by_model.get("svm", []) if r["n"] == 1), None)
    if svm1:
        s = E.bootstrap_summary(svm1["boot"])
        print(f"`svm C=0.02`: σ = **{num(s['std'])}** · "
              f"CI95 [{num(s['ci95_lo'])}; {num(s['ci95_hi'])}] · {N_BOOT} lần lấy lại mẫu.\n")
    sig = [r["sigma"] for r in runs]
    print(f"σ trên toàn bộ {len(runs)} run: từ {num(min(sig))} tới {num(max(sig))}, "
          f"trung vị {num(float(np.median(sig)))}.\n")
    print("`docs/` hiện viện dẫn σ ≈ 0,009 ở năm chỗ mà chưa có code nào sinh ra nó.")
    print("Con số trên là lần đầu nó được tính trong repo.\n")
    print("> **Nhưng đừng dùng σ để so hai mô hình.** σ trả lời 'điểm này dao động")
    print("> bao nhiêu nếu đổi tập val', còn câu cần hỏi là 'A có hơn B trên cùng")
    print("> những dòng đó không'. Bảng 3 trả lời câu thứ hai, và nó nhạy hơn nhiều")
    print("> vì hai mô hình sai ở phần lớn cùng những tin nhập nhằng.")


if __name__ == "__main__":
    main()
