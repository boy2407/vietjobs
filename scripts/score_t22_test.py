"""T22 — chấm `test` một lần bằng best.pt đã lưu của mọi run T22 (không train lại).

Ngoại lệ tác giả 2026-11-23 (Rule 5 mục 6): số này là test-informed, không dùng để chọn
cấu hình. Vector `test` được nhúng trong lần chạy này và chỉ lưu vào thư mục tạm
`--tmp` để chạy lại khi lỗi giữa chừng — không vào `artifacts/embeddings/`.

    PYTHONPATH=src .venv-dl/bin/python scripts/score_t22_test.py --confirm-test --tmp <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from vietjobs import config as C
from vietjobs import dataset as D
from vietjobs import evaluate as E
from vietjobs.dl import encode as ENC
from vietjobs.dl import extra as XF
from vietjobs.dl import text as T
from vietjobs.dl.heads import DenseHead


def vectors(tmp: Path, name: str, texts: list[str], max_len: int) -> np.ndarray:
    path = tmp / f"test-{name}-len{max_len}.npy"
    if path.exists():
        return np.load(path)
    print(f"[test] nhúng {name} · {len(texts):,} tin · len{max_len}", flush=True)
    emb = ENC.encode_texts(texts, max_len=max_len, batch=32, device="mps", encoder="cafebert")
    np.save(path, emb)
    return emb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm-test", action="store_true")
    ap.add_argument("--tmp", required=True)
    args = ap.parse_args()
    if not args.confirm_test:
        raise SystemExit("cần --confirm-test (docs/03-protocol.md, điều 2)")
    tmp = Path(args.tmp)
    tmp.mkdir(parents=True, exist_ok=True)

    df = D.load_split("test")
    base = {task: vectors(tmp, ENC.family(task), T.build_text(df, task=task, segmented=False), 512)
            for task in (C.TASK_CATEGORY, C.TASK_SALARY)}
    txt = {c: vectors(tmp, f"xf-{c}", df[c].astype(str).str.strip().tolist(), XF.TEXT_MAX_LEN)
           for c in XF.TEXT_COLUMNS}
    disclosed = ((df["salary_disclosed"] == 1) & df["salary_mid"].notna()).to_numpy()

    rows = []
    for run in sorted(C.ARTIFACT_DIR.glob("dl-*-t22-r*-s*")):
        m = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        cfg, is_reg = m["config"], m["task"] == C.TASK_SALARY
        X = base[m["task"]]
        if cfg.get("extra"):
            S, names = XF.structured(df, m["t22"])
            text_cols, prefixes = XF.EXTRA_SETS[cfg.get("extra_set", "all")]
            keep = XF.select(names, prefixes)
            assert [names[i] for i in keep] == m["t22"]["structured_columns"]
            t = np.concatenate([txt[c] for c in text_cols], axis=1)
            X = np.concatenate([X, t, S[:, keep]], axis=1)
        sc = np.load(run / "scaler.npz")
        Xs = torch.tensor((X - sc["mean"]) / sc["std"], dtype=torch.float32)
        out_dim = 1 if is_reg else len(m["labels"])
        model = DenseHead(Xs.shape[1], cfg["hidden"], out_dim, cfg["dropout"],
                          pre_hidden=cfg["pre_hidden"])
        model.load_state_dict(torch.load(run / "best.pt", map_location="cpu"))
        model.eval()
        with torch.no_grad():
            logits = model(Xs, None).numpy()
        res = {"run_id": m["run_id"], "task": m["task"]}
        if is_reg:
            y = np.log1p(df.loc[disclosed, "salary_mid"].to_numpy(float))
            res["disclosed"] = E.regression_metrics(y, logits[disclosed, 0])
        else:
            pred = np.array(m["labels"])[logits.argmax(1)]
            y = df["category"].to_numpy()
            res["all"] = E.classification_metrics(y, pred, m["labels"])
            res["disclosed"] = E.classification_metrics(y[disclosed], pred[disclosed], m["labels"])
        (run / "metrics_test.json").write_text(json.dumps(res, indent=2, ensure_ascii=False),
                                               encoding="utf-8")
        rows.append(res)
        key = "mae_trieu" if is_reg else "f1_macro"
        print(m["run_id"], {k: round(v[key], 4) for k, v in res.items() if isinstance(v, dict)},
              flush=True)


if __name__ == "__main__":
    main()
