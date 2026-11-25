"""T22 R6-L4 — chấm `test` một lần bằng best.pt đã lưu của 25 run R6-L4 (không train lại).

Ngoại lệ tác giả 2026-11-23 (Rule 5 mục 6): số này là test-informed. Hai bước:

    # Mac (cần bộ tách từ cho PhoBERT): dựng text test, ghi ra thư mục tạm, test.csv không đổi
    PYTHONPATH=src .venv/bin/python scripts/score_r6l4_test.py prep --confirm-test --tmp <dir>
    # máy GPU: nhúng 4 lớp cuối từng encoder rồi chấm từng run
    PYTHONPATH=src python scripts/score_r6l4_test.py score --confirm-test --tmp <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from vietjobs import config as C
from vietjobs import dataset as D

SPECS = {"pho": ("phobert", 256, 64, True), "cafe": ("cafebert", 512, 32, False),
         "visobert": ("visobert", 512, 32, False), "xlmr": ("xlmr", 512, 32, False),
         "bamibert": ("bamibert", 1024, 16, False)}


def prep(tmp: Path) -> None:
    from build_r6text_split import extra_text
    from vietjobs import vitext as V
    from vietjobs.dl import text as T
    df = D.load_split("test")
    add = extra_text(df.fillna("").astype(str))
    add_seg = V.segment_many(add.tolist())
    out = {}
    for seg, extra in ((False, add.tolist()), (True, add_seg)):
        base = T.build_text(df, task=C.TASK_SALARY, segmented=seg)
        out["seg" if seg else "raw"] = [f"{b} . {a}" if a else b for b, a in zip(base, extra)]
    keep = ((df["salary_disclosed"] == 1) & df["salary_mid"].notna()).to_numpy()
    out["keep"] = keep.tolist()
    out["salary_mid"] = df["salary_mid"].fillna(0).astype(float).tolist()
    (tmp / "test_r6text.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"test: {len(df):,} tin · {keep.sum():,} có lương")


def score(tmp: Path) -> None:
    import torch
    from vietjobs import evaluate as E
    from vietjobs.dl import encode as ENC
    from vietjobs.dl.heads import DenseHead
    d = json.loads((tmp / "test_r6text.json").read_text(encoding="utf-8"))
    keep = np.array(d["keep"])
    y = np.log1p(np.array(d["salary_mid"])[keep])
    for tag, (enc, max_len, batch, seg) in SPECS.items():
        path = tmp / f"test-{enc}-len{max_len}-l4.npy"
        if not path.exists():
            np.save(path, ENC.encode_texts(d["seg" if seg else "raw"], max_len=max_len,
                                           batch=batch, device="cuda", encoder=enc, layers=4))
        X = np.load(path)
        for s in range(42, 47):
            run = C.ARTIFACT_DIR / f"dl-sal-t22-r6l4-{tag}-s{s}"
            if tag == "cafe":
                run = C.ARTIFACT_DIR / f"dl-sal-t22-r6l4-s{s}"
            m = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
            cfg, sc = m["config"], np.load(run / "scaler.npz")
            head = DenseHead(X.shape[1], cfg["hidden"], 1, cfg["dropout"], pre_hidden=cfg["pre_hidden"])
            head.load_state_dict(torch.load(run / "best.pt", map_location="cpu"))
            head.eval()
            Xs = torch.tensor((X - sc["mean"]) / sc["std"], dtype=torch.float32)
            with torch.no_grad():
                pred = head(Xs, None).numpy()[:, 0]
            res = {"run_id": m["run_id"], "task": m["task"], "eval": "test",
                   "disclosed": E.regression_metrics(y, pred[keep])}
            (run / "metrics_test.json").write_text(json.dumps(res, indent=2, ensure_ascii=False),
                                                   encoding="utf-8")
            print(m["run_id"], round(res["disclosed"]["mae_trieu"], 4), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("prep", "score"))
    ap.add_argument("--confirm-test", action="store_true")
    ap.add_argument("--tmp", required=True)
    args = ap.parse_args()
    if not args.confirm_test:
        raise SystemExit("cần --confirm-test (docs/03-protocol.md, điều 2)")
    tmp = Path(args.tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    prep(tmp) if args.step == "prep" else score(tmp)


if __name__ == "__main__":
    main()
