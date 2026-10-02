"""Multitask Learning - Shared trunk, two heads (Category & Salary).

    python -m vietjobs.dl.train_mtl

Chạy trên vector PhoBERT đã được cache (`masked`). 
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
from .focal_loss import FocalLoss
from .heads import MultiTaskHead, MultiTaskRnn
from .train_dl import token_stats


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=C.ROOT,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return ""


def environment(device, encoder: str = ENC.DEFAULT_ENCODER) -> dict:
    import sklearn
    import transformers
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "device": str(device),
        "encoder": ENC.ENCODERS[encoder]["name"],
        "git_sha": _git("rev-parse", "HEAD") or "uncommitted",
        "git_dirty": bool(_git("status", "--porcelain")),
    }


def load_mtl(split: str, max_len: int, encoder: str = ENC.DEFAULT_ENCODER, head: str = "dense"):
    df = D.load_split(split)
    # Trunk reads 'masked' family (use task 'salary' to get masked cache).
    # head="rnn": X là memmap float16 [n, T, dim] chưa gộp, mask [n, T] uint8.
    if head == "rnn":
        X, mask = ENC.token_cache_for(split, "salary", max_len=max_len, encoder=encoder)
    else:
        X, mask = ENC.embeddings_for(split, "salary", max_len=max_len, encoder=encoder), None
    
    if len(X) != len(df):
        raise SystemExit(f"cache {split} có {len(X)} dòng nhưng split có {len(df)}")
        
    y_cat = df["category"].to_numpy()
    
    keep_sal = ((df["salary_disclosed"] == 1) & df["salary_mid"].notna()).to_numpy()
    y_sal = np.where(keep_sal, np.log1p(df["salary_mid"].to_numpy(dtype=np.float32)), np.nan)
    
    return X, mask, y_cat, y_sal, df, keep_sal


def batches(n: int, size: int, shuffle: bool, gen: torch.Generator):
    idx = torch.randperm(n, generator=gen) if shuffle else torch.arange(n)
    for i in range(0, n, size):
        yield idx[i:i + size]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--pre-hidden", type=int, default=0)
    ap.add_argument("--head", default="dense", choices=["dense", "rnn"],
                    help="trunk: dense trên vector đã gộp, hoặc rnn trên token cache (--pooling none)")
    ap.add_argument("--branches", default="both", choices=["both", "gru", "lstm"])
    ap.add_argument("--spatial-dropout", type=float, default=0.2)
    ap.add_argument("--conv-channels", type=int, default=64)
    ap.add_argument("--encoder", default=ENC.DEFAULT_ENCODER, choices=list(ENC.ENCODERS))
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--class-weight", action="store_true")
    ap.add_argument("--alpha-sal", type=float, default=1.0, help="Weight for salary loss")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--eval", default="dev", choices=["dev", "test"])
    ap.add_argument("--confirm-test", action="store_true")
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--seed", type=int, default=C.RANDOM_SEED)
    return ap


def main() -> None:
    run(build_parser().parse_args())


def run(args: argparse.Namespace) -> dict:
    if args.eval == "test" and not args.confirm_test:
        raise SystemExit("chấm test cần --confirm-test")

    is_rnn = args.head == "rnn"
    if args.pre_hidden and is_rnn:
        raise SystemExit("--pre-hidden chỉ dùng với --head dense")
    if is_rnn and args.batch > 64 and args.device != "cuda":
        print(f"  --head rnn trên {args.device}: hạ batch {args.batch} → 64")
        args.batch = 64

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = ENC.pick_device(args.device)

    Xtr, mtr, y_cat_tr, y_sal_tr, df_tr, keep_sal_tr = load_mtl("train", args.max_len, args.encoder, args.head)
    Xev, mev, y_cat_ev, y_sal_ev, df_ev, keep_sal_ev = load_mtl(args.eval, args.max_len, args.encoder, args.head)

    n_train = len(y_cat_tr)
    labels = sorted(set(y_cat_tr.tolist()))
    to_id = {lab: i for i, lab in enumerate(labels)}
    y_cat_tr_t = torch.tensor([to_id[v] for v in y_cat_tr], dtype=torch.long)
    y_cat_ev_t = torch.tensor([to_id.get(v, -1) for v in y_cat_ev], dtype=torch.long)

    y_sal_tr_t = torch.tensor(np.nan_to_num(y_sal_tr), dtype=torch.float32).unsqueeze(1)
    mask_sal_tr_t = torch.tensor(keep_sal_tr, dtype=torch.float32).unsqueeze(1)
    
    out_dim_cat = len(labels)

    if is_rnn:
        mu, sigma = token_stats(Xtr, np.arange(len(Xtr)), mtr)
    else:
        mu, sigma = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True)
    sigma = np.clip(sigma, 1e-6, None)

    if is_rnn:
        # Như train_dl: đọc từng batch từ memmap, chuẩn hoá rồi mới đẩy lên device.
        def take(X, mask, idx):
            order = idx.numpy() if torch.is_tensor(idx) else idx
            xb = (np.asarray(X[order], dtype=np.float32) - mu) / sigma
            return (torch.from_numpy(xb).to(device),
                    torch.from_numpy(mask[order].astype(np.uint8)).to(device))
        in_dim = ENC.hidden_of(args.encoder)
        model = MultiTaskRnn(in_dim, args.hidden, out_dim_cat, args.dropout,
                             spatial_dropout=args.spatial_dropout,
                             conv_channels=args.conv_channels,
                             branches=args.branches).to(device)
    else:
        Xtr = torch.tensor((Xtr - mu) / sigma, dtype=torch.float32)
        Xev = torch.tensor((Xev - mu) / sigma, dtype=torch.float32).to(device)

        def take(X, mask, idx):
            return X[idx].to(device), None
        in_dim = Xtr.shape[1]
        model = MultiTaskHead(in_dim, args.hidden, out_dim_cat, args.dropout,
                              pre_hidden=args.pre_hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    loss_fn_sal = nn.SmoothL1Loss(reduction='none')
    w = None
    if args.class_weight:
        counts = np.bincount(y_cat_tr_t.numpy(), minlength=out_dim_cat).astype(np.float32)
        w = torch.tensor(counts.sum() / (out_dim_cat * np.clip(counts, 1, None))).to(device)
    loss_fn_cat = nn.CrossEntropyLoss(weight=w)

    run_id = args.run_id or (
        f"dl-mtl{'-rnn' if is_rnn else ''}-h{args.hidden}"
        f"{'-cw' if args.class_weight else ''}"
        f"-a{args.alpha_sal}"
        f"-{datetime.now().strftime('%m%d-%H%M')}"
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
        total_loss, seen = 0.0, 0
        for idx in batches(n_train, args.batch, True, gen):
            xb, mb = take(Xtr, mtr, idx)
            yc_b = y_cat_tr_t[idx].to(device)
            ys_b = y_sal_tr_t[idx].to(device)
            ms_b = mask_sal_tr_t[idx].to(device)
            
            opt.zero_grad()
            logits_cat, logits_sal = model(xb, mb)
            
            l_cat = loss_fn_cat(logits_cat, yc_b)
            l_sal_all = loss_fn_sal(logits_sal, ys_b)
            l_sal = (l_sal_all * ms_b).sum() / (ms_b.sum() + 1e-6)
            
            loss = l_cat + args.alpha_sal * l_sal
            loss.backward()
            
            if args.clip:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            
            total_loss += loss.detach().item() * len(idx)
            seen += len(idx)

        model.eval()
        with torch.no_grad():
            if is_rnn:
                outs = [tuple(o.cpu() for o in model(*take(Xev, mev, idx)))
                        for idx in batches(len(y_cat_ev), args.batch, False, gen)]
                logits_cat_ev = torch.cat([o[0] for o in outs])
                logits_sal_ev = torch.cat([o[1] for o in outs])
            else:
                logits_cat_ev, logits_sal_ev = (o.cpu() for o in model(Xev, None))
            
        pred_log_sal = logits_sal_ev.squeeze(1).numpy()
        proba_cat = torch.softmax(logits_cat_ev, dim=1).numpy()
        pred_cat = np.array(labels)[proba_cat.argmax(1)]
        
        m_cat = E.classification_metrics(y_cat_ev, pred_cat, labels)
        
        # Only evaluate salary on disclosed ones
        yev_sal_valid = y_sal_ev[keep_sal_ev]
        pred_log_sal_valid = pred_log_sal[keep_sal_ev]
        m_sal = E.regression_metrics(yev_sal_valid, pred_log_sal_valid)
        
        # Score is combined: macro_f1 (higher is better) + somewhat normalized R2 or we just use F1
        score = m_cat["f1_macro"] # Focus on cat F1 to select best epoch, or average
        
        line = {
            "val_macro_f1": m_cat["f1_macro"],
            "val_mae_trieu": m_sal["mae_trieu"],
            "val_r2_log": m_sal["r2_log"]
        }

        with hist_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"epoch": epoch, "train_loss": total_loss / seen,
                                 "seconds": round(time.time() - t0, 1), **line},
                                ensure_ascii=False) + "\n")
        print(f"  epoch {epoch:>3}  loss {total_loss / seen:.4f}  "
              + "  ".join(f"{k} {v:.4f}" for k, v in line.items()), flush=True)

        if score > best_score:
            best_score, best_epoch, waited = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_m_cat = m_cat
            best_m_sal = m_sal
            best_pred_cat = pred_cat
            best_pred_sal = pred_log_sal
            best_proba_cat = proba_cat
        else:
            waited += 1
            if waited >= args.patience:
                print(f"  dừng sớm ở epoch {epoch} (tốt nhất: {best_epoch})")
                break

    seconds = time.time() - t0
    model.load_state_dict(best_state)
    torch.save(best_state, out_dir / "best.pt")
    np.savez(out_dir / "scaler.npz", mean=mu, std=sigma)

    pred_frame = {
        "y_true_cat": y_cat_ev,
        "y_pred_cat": best_pred_cat,
        "y_true_sal": np.where(keep_sal_ev, np.expm1(y_sal_ev), np.nan),
        "y_pred_sal": np.expm1(best_pred_sal)
    }
    pred_frame["proba_cat"] = [row.tolist() for row in best_proba_cat.astype(np.float32)]
    pd.DataFrame(pred_frame).to_parquet(out_dir / f"predictions_{args.eval}.parquet")

    env = environment(device, args.encoder)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps({
        "run_id": run_id, "task": "mtl", "model": f"{args.encoder}-frozen+{args.head}-mtl",
        "eval": args.eval, "best_epoch": best_epoch, "epochs_run": epoch,
        "seconds": seconds, "metrics_cat": best_m_cat, "metrics_sal": best_m_sal,
        "labels": labels, "config": vars(args), "env": env, "encoder": args.encoder
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    headline = (f"macroF1={best_m_cat['f1_macro']:.4f} · "
                f"MAE={best_m_sal['mae_trieu']:.2f}tr · "
                f"R2log={best_m_sal['r2_log']:.3f}")

    if not args.no_log:
        path = C.RESULTS_LOG
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(f"| {run_id} | {datetime.now(timezone.utc).strftime('%m-%d %H:%M')} "
                     f"| mtl | {args.encoder}-frozen+{args.head}-mtl(h={args.hidden}) "
                     f"| title+desc+req | mask+nosegment | {args.eval} | {len(y_cat_ev)} | {headline} | {seconds:.1f}s |\n")

    print(f"\n[{run_id}] MTL · best epoch {best_epoch}/{epoch} · {headline}")
    return {"run_id": run_id, "out_dir": out_dir}

if __name__ == "__main__":
    main()
