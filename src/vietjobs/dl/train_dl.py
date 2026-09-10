"""Baseline học sâu — PhoBERT đóng băng + dense, mỗi bài toán một mạng riêng.

    python -m vietjobs.dl.train_dl --task category
    python -m vietjobs.dl.train_dl --task salary

Chạy trên vector đã cache (``dl/encode.py``), nên một epoch tính bằng giây. Mỗi
lần chạy ghi ``artifacts/<run_id>/`` gồm:

    config.json      mọi siêu tham số + seed, đủ để chạy lại
    env.json         thiết bị, phiên bản torch/transformers
    history.jsonl    một dòng mỗi epoch — đường học nằm ở đây, không ở metrics
    metrics.json     số cuối cùng, cùng khoá với các lần chạy học máy
    best.pt          trọng số ở epoch có val tốt nhất
    predictions_dev.parquet

và **một** dòng trong ``docs/04-results.md``. Không bao giờ ghi từng epoch vào đó.

Luật không đổi: chọn cấu hình trên ``val``; ``test`` chỉ chạm với ``--confirm-test``.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from torch import nn

from .. import config as C
from .. import dataset as D
from .. import evaluate as E
from . import encode as ENC
from . import text as T
from .heads import DenseHead


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=C.ROOT,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return ""


def environment(device) -> dict:
    import transformers

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": str(device),
        "encoder": ENC.MODEL_NAME,
        "git_sha": _git("rev-parse", "HEAD") or "uncommitted",
        "git_dirty": bool(_git("status", "--porcelain")),
    }


def load_task(split: str, task: str, max_len: int):
    """Trả về (X, y, frame) — frame giữ lại để cắt lát kết quả theo ngành."""
    df = D.load_split(split)
    X = ENC.embeddings_for(split, task, max_len=max_len)
    if len(X) != len(df):
        raise SystemExit(f"cache {split} có {len(X)} dòng nhưng split có {len(df)} "
                         "— chạy lại `python -m vietjobs.dl.encode --force`")
    if task == C.TASK_CATEGORY:
        keep = df["category"].notna().to_numpy()
        return X[keep], df.loc[keep, "category"].to_numpy(), df[keep]
    if task == C.TASK_SALARY:
        keep = ((df["salary_disclosed"] == 1) & df["salary_mid"].notna()).to_numpy()
        y = np.log1p(df.loc[keep, "salary_mid"].to_numpy(dtype=np.float32))
        return X[keep], y, df[keep]
    keep = df["salary_disclosed"].notna().to_numpy()
    return X[keep], df.loc[keep, "salary_disclosed"].to_numpy(dtype=np.int64), df[keep]


def batches(n: int, size: int, shuffle: bool, gen: torch.Generator):
    idx = torch.randperm(n, generator=gen) if shuffle else torch.arange(n)
    for i in range(0, n, size):
        yield idx[i:i + size]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True, choices=list(C.TASKS))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--clip", type=float, default=1.0,
                    help="chuẩn gradient tối đa; 0 = tắt")
    ap.add_argument("--no-standardize", action="store_true",
                    help="bỏ chuẩn hoá đặc trưng — chỉ để đo lại vì sao cần nó")
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--class-weight", action="store_true",
                    help="cân bằng lớp cho bài phân lớp (lệch 27:1)")
    ap.add_argument("--eval", default="dev", choices=["dev", "test"])
    ap.add_argument("--confirm-test", action="store_true")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--seed", type=int, default=C.RANDOM_SEED)
    args = ap.parse_args()

    if args.eval == "test" and not args.confirm_test:
        raise SystemExit("chấm test cần --confirm-test (docs/03-protocol.md, điều 2)")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = ENC.pick_device("auto")
    is_reg = args.task == C.TASK_SALARY

    Xtr, ytr, _ = load_task("train", args.task, args.max_len)
    Xev, yev, ev_df = load_task(args.eval, args.task, args.max_len)

    labels = None
    if not is_reg:
        labels = sorted(set(ytr.tolist()))
        to_id = {lab: i for i, lab in enumerate(labels)}
        ytr_t = torch.tensor([to_id[v] for v in ytr], dtype=torch.long)
        yev_ids = np.array([to_id.get(v, -1) for v in yev])
        out_dim = len(labels)
    else:
        ytr_t = torch.tensor(ytr, dtype=torch.float32).unsqueeze(1)
        out_dim = 1

    # Vector PhoBERT rất bất đẳng hướng: cosine trung vị giữa hai tin bất kỳ là
    # 0,896 — mọi tin nằm trong một nón hẹp, và các chiều lệch tâm khác nhau.
    # Chuẩn hoá từng chiều theo thống kê của `train` đưa chúng về cùng thang; thiếu
    # bước này vòng huấn luyện phân kỳ (đo được: loss 2,6 → 321 → 1.941 ở run
    # `dl-cat-h256`). Trung bình và độ lệch chuẩn được lưu cùng run vì đường suy
    # luận sau này phải dùng đúng con số đó (quy tắc số 4).
    mu = Xtr.mean(0, keepdims=True)
    sigma = Xtr.std(0, keepdims=True)
    if args.no_standardize:
        mu, sigma = np.zeros_like(mu), np.ones_like(sigma)
    sigma = np.clip(sigma, 1e-6, None)

    Xtr_t = torch.tensor((Xtr - mu) / sigma, dtype=torch.float32)
    Xev_t = torch.tensor((Xev - mu) / sigma, dtype=torch.float32).to(device)

    model = DenseHead(Xtr.shape[1], args.hidden, out_dim, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if is_reg:
        loss_fn = nn.SmoothL1Loss()  # đuôi lương rất dài (skew 11,8) — huber chịu đuôi tốt hơn MSE
    elif args.class_weight:
        counts = np.bincount(ytr_t.numpy(), minlength=out_dim).astype(np.float32)
        w = torch.tensor(counts.sum() / (out_dim * np.clip(counts, 1, None)))
        loss_fn = nn.CrossEntropyLoss(weight=w.to(device))
    else:
        loss_fn = nn.CrossEntropyLoss()

    run_id = args.run_id or (
        f"dl-{'reg' if is_reg else args.task[:3]}-h{args.hidden}"
        f"{'-cw' if args.class_weight else ''}-{datetime.now().strftime('%m%d-%H%M')}"
    )
    out_dir = C.ARTIFACT_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    hist_path = out_dir / "history.jsonl"
    hist_path.write_text("", encoding="utf-8")

    gen = torch.Generator().manual_seed(args.seed)
    best_score, best_epoch, best_state, waited = -np.inf, -1, None, 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total, seen = 0.0, 0
        for idx in batches(len(Xtr_t), args.batch, True, gen):
            xb = Xtr_t[idx].to(device)
            yb = ytr_t[idx].to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            if args.clip:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            total += float(loss) * len(idx)
            seen += len(idx)

        model.eval()
        with torch.no_grad():
            logits = model(Xev_t).cpu()
        if is_reg:
            pred_log = logits.squeeze(1).numpy()
            m = E.regression_metrics(yev, pred_log)
            score = -m["mae_trieu"]          # càng nhỏ càng tốt → đổi dấu
            line = {"val_mae_trieu": m["mae_trieu"], "val_r2_log": m["r2_log"]}
        else:
            proba = torch.softmax(logits, dim=1).numpy()
            pred = np.array(labels)[proba.argmax(1)]
            m = E.classification_metrics(yev, pred, proba, labels)
            score = m["f1_macro"]
            line = {"val_macro_f1": m["f1_macro"], "val_accuracy": m["accuracy"]}

        with hist_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"epoch": epoch, "train_loss": total / seen,
                                 "seconds": round(time.time() - t0, 1), **line},
                                ensure_ascii=False) + "\n")
        print(f"  epoch {epoch:>3}  loss {total / seen:.4f}  "
              + "  ".join(f"{k} {v:.4f}" for k, v in line.items()), flush=True)

        if score > best_score:
            best_score, best_epoch, waited = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics, best_pred = m, (pred_log if is_reg else pred)
        else:
            waited += 1
            if waited >= args.patience:
                print(f"  dừng sớm ở epoch {epoch} (tốt nhất: {best_epoch})")
                break

    seconds = time.time() - t0
    model.load_state_dict(best_state)
    torch.save(best_state, out_dir / "best.pt")
    np.savez(out_dir / "scaler.npz", mean=mu, std=sigma)

    pred_frame = {"y_true": yev if not is_reg else np.expm1(yev),
                  "y_pred": best_pred if not is_reg else np.expm1(best_pred),
                  "category": ev_df["category"].to_numpy()}
    pd.DataFrame(pred_frame).to_parquet(out_dir / f"predictions_{args.eval}.parquet")

    env = environment(device)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    (out_dir / "env.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps({
        "run_id": run_id, "task": args.task, "model": "phobert-frozen+dense",
        "eval": args.eval, "best_epoch": best_epoch, "epochs_run": epoch,
        "seconds": seconds, "metrics": best_metrics, "config": vars(args), "env": env,
        "columns": T.columns_for(args.task),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    if is_reg:
        headline = (f"MAE={best_metrics['mae_trieu']:.2f}tr · "
                    f"MedAE={best_metrics['median_ae_trieu']:.2f}tr · "
                    f"R2log={best_metrics['r2_log']:.3f} · "
                    f"±20%={best_metrics['within_20pct'] * 100:.1f}%")
    else:
        headline = (f"macroF1={best_metrics['f1_macro']:.4f} · "
                    f"acc={best_metrics['accuracy']:.4f} · "
                    f"balAcc={best_metrics['balanced_accuracy']:.4f}"
                    + (f" · top3={best_metrics['top3_accuracy']:.4f}"
                       if "top3_accuracy" in best_metrics else ""))

    path = C.RESULTS_LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(
            "# Experiment log (append-only)\n\n"
            "Selection uses `val`. `test` is scored once, at the end.\n\n"
            "| RunID | UTC | Task | Model | Scope | Prep | Eval | n | Headline | Time | Commit |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|\n",
            encoding="utf-8")
    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"| {run_id} | {datetime.now(timezone.utc).strftime('%m-%d %H:%M')} "
                 f"| {args.task} | phobert-frozen+dense(h={args.hidden}"
                 f"{',cw' if args.class_weight else ''}) | title+desc+req | segment "
                 f"| {args.eval} | {best_metrics['n']} | {headline} | {seconds:.1f}s "
                 f"| {env['git_sha'][:8]}{'+dirty' if env['git_dirty'] else ''} |\n")

    print(f"\n[{run_id}] {args.task} · best epoch {best_epoch}/{epoch} · {headline}")
    print(f"  -> artifacts/{run_id}/")


if __name__ == "__main__":
    main()
