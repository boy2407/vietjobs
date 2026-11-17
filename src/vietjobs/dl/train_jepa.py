"""T19 — JEPA hai giai đoạn trên tin tuyển dụng.

    PYTHONPATH=src python -m vietjobs.dl.train_jepa pretrain --run-id jepa-pre-s42
    PYTHONPATH=src python -m vietjobs.dl.train_jepa finetune --init jepa --from-run jepa-pre-s42
    PYTHONPATH=src python -m vietjobs.dl.train_jepa finetune --init random      # đối chứng

Giai đoạn 1 (``pretrain``) chỉ đọc text ``train`` họ ``masked`` — không nhãn; ``dev``
chỉ để theo dõi loss và phát hiện sụp đổ. Giai đoạn 2 (``finetune``) đóng băng
encoder rồi học ``MultiTaskRnn`` như ``train_mtl --head rnn``.
"""
from __future__ import annotations

import argparse
import json
import math
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
from .heads import MultiTaskRnn, class_weights
from .jepa import JepaEncoder, Predictor, block_mask, ema_update, jepa_step
from .train_dl import compute_salary_bin_weights
from .train_mtl import batches, environment, save_atomic

JEPA_DIR = C.ARTIFACT_DIR / "jepa"
SOURCE = "cafebert"                      # tokenizer + bảng embedding từ vựng


def token_ids(split: str, max_len: int):
    """``(ids int32 [n, T], attn uint8 [n, T])`` của text họ masked, cache dưới ``artifacts/jepa``."""
    if split not in ("train", "dev"):
        raise SystemExit(f"JEPA chỉ đọc train/dev, không đọc {split!r} (Rule 5)")
    path = JEPA_DIR / f"tok-{split}-len{max_len}.npz"
    if not path.exists():
        from transformers import AutoTokenizer
        spec = ENC.ENCODERS[SOURCE]
        tok = AutoTokenizer.from_pretrained(spec["name"], use_fast=spec["use_fast"])
        texts = T.build_text(D.load_split(split), task="salary", segmented=spec["segmented"])
        enc = tok(texts, max_length=max_len, truncation=True, padding="max_length")
        JEPA_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(path, ids=np.asarray(enc["input_ids"], dtype=np.int32),
                 attn=np.asarray(enc["attention_mask"], dtype=np.uint8))
    z = np.load(path)
    return z["ids"], z["attn"]


def word_table() -> torch.Tensor:
    """Bảng embedding từ vựng của CafeBERT ``[vocab, 1024]`` — trước mọi tầng attention."""
    path = JEPA_DIR / f"{SOURCE}-wordemb.f16.npy"
    if not path.exists():
        from transformers import AutoModel
        model = AutoModel.from_pretrained(ENC.ENCODERS[SOURCE]["name"])
        w = model.get_input_embeddings().weight.detach().to(torch.float16).numpy()
        JEPA_DIR.mkdir(parents=True, exist_ok=True)
        np.save(path, w)
    return torch.from_numpy(np.load(path))


def make_encoder(table: torch.Tensor, args) -> JepaEncoder:
    return JepaEncoder(table, d=args.dim, layers=args.layers, heads=args.heads,
                       max_len=args.max_len, dropout=args.enc_dropout)


@torch.no_grad()
def collapse_std(enc: JepaEncoder, ids, attn, device, batch: int = 64) -> float:
    """Độ lệch chuẩn giữa các tin của vector trung bình, lấy trung bình theo chiều.
    Gần 0 = mọi tin cùng một vector (sụp đổ)."""
    enc.eval()
    pooled = []
    for i in range(0, len(ids), batch):
        x = torch.from_numpy(ids[i:i + batch]).long().to(device)
        m = torch.from_numpy(attn[i:i + batch]).bool().to(device)
        h = enc(x, m)
        pooled.append(((h * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True)).float().cpu())
    return float(torch.cat(pooled).std(0).mean())


def pretrain(args) -> None:
    torch.manual_seed(args.seed)
    device = ENC.pick_device(args.device)
    ids, attn = token_ids("train", args.max_len)
    ids_ev, attn_ev = token_ids("dev", args.max_len)
    ids_ev, attn_ev = ids_ev[:args.dev_rows], attn_ev[:args.dev_rows]
    if args.limit:
        ids, attn = ids[:args.limit], attn[:args.limit]
    table = word_table().to(device)
    enc, tgt = make_encoder(table, args).to(device), make_encoder(table, args).to(device)
    tgt.load_state_dict(enc.state_dict())
    tgt.eval()                                          # đích không dropout
    for p in tgt.parameters():
        p.requires_grad_(False)
    pred = Predictor(args.dim, args.pred_dim, args.pred_layers, args.heads,
                     args.max_len, args.enc_dropout).to(device)
    params = list(enc.parameters()) + list(pred.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    steps = args.epochs * math.ceil(len(ids) / args.batch)
    warm = max(1, int(args.warmup * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm)
                                              * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    use_amp = device.type == "cuda"

    out = C.ARTIFACT_DIR / args.run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    env = environment(device, SOURCE)
    env["encoder"] = f"jepa(d={args.dim},L={args.layers}) on {env['encoder']} word table"
    (out / "env.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    hist = out / "history.jsonl"
    hist.write_text("", encoding="utf-8")

    gen = torch.Generator().manual_seed(args.seed)
    ev_masks = block_mask(torch.from_numpy(attn_ev), gen=torch.Generator().manual_seed(0))
    best, step, t0 = math.inf, 0, time.time()
    for epoch in range(1, args.epochs + 1):
        enc.train(), pred.train()
        tot, seen = 0.0, 0
        for idx in batches(len(ids), args.batch, True, gen):
            a = torch.from_numpy(attn[idx.numpy()])
            masked = block_mask(a, gen=gen).to(device)
            x = torch.from_numpy(ids[idx.numpy()]).long().to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss = jepa_step(enc, tgt, pred, x, a.to(device), masked)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.clip)
            opt.step(), sched.step()
            tau = args.tau + (1.0 - args.tau) * min(1.0, step / steps)
            ema_update(tgt, enc, tau)
            step += 1
            tot, seen = tot + loss.item() * len(idx), seen + len(idx)

        enc.eval(), pred.eval()
        with torch.no_grad():
            dl = [jepa_step(enc, tgt, pred,
                            torch.from_numpy(ids_ev[i:i + args.batch]).long().to(device),
                            torch.from_numpy(attn_ev[i:i + args.batch]).to(device),
                            ev_masks[i:i + args.batch].to(device)).item()
                  for i in range(0, len(ids_ev), args.batch)]
        dev_loss, std = float(np.mean(dl)), collapse_std(tgt, ids_ev, attn_ev, device)
        line = {"epoch": epoch, "train_loss": tot / seen, "dev_loss": dev_loss,
                "collapse_std": std, "tau": tau, "seconds": round(time.time() - t0, 1)}
        with hist.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line) + "\n")
        print("  " + "  ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}"
                               for k, v in line.items()), flush=True)
        if std < args.min_std:
            print(f"  sụp đổ: collapse_std {std:.4f} < {args.min_std} — dừng")
            break
        if dev_loss < best:
            best = dev_loss
            save_atomic({k: v.cpu() for k, v in tgt.state_dict().items()}, out / "encoder.pt")
    print(f"[{args.run_id}] pretrain xong · dev_loss tốt nhất {best:.4f} · {out}")


def labels_for(split: str):
    """Nhãn như ``train_mtl.load_mtl``: lương là log1p, chỉ ở tin đã công bố."""
    df = D.load_split(split)
    keep = ((df["salary_disclosed"] == 1) & df["salary_mid"].notna()).to_numpy()
    y_sal = np.where(keep, np.log1p(df["salary_mid"].to_numpy(dtype=np.float32)), np.nan)
    return df["category"].to_numpy(), y_sal, keep


def finetune(args) -> None:
    """Giai đoạn 2: encoder đóng băng → ``MultiTaskRnn``, cùng loss và cách chọn epoch như ``train_mtl``."""
    if args.init == "jepa":
        if not args.from_run:
            raise SystemExit("--init jepa cần --from-run <run_id của pretrain>")
        pre = json.loads((C.ARTIFACT_DIR / args.from_run / "config.json").read_text())
        for k in ("dim", "layers", "heads", "max_len", "enc_dropout"):
            setattr(args, k, pre[k])                    # kiến trúc phải khớp bản đã học
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = ENC.pick_device(args.device)
    ids, attn = token_ids("train", args.max_len)
    ids_ev, attn_ev = token_ids("dev", args.max_len)
    yc, ys, ks = labels_for("train")
    yc_ev, ys_ev, ks_ev = labels_for("dev")
    if args.limit:
        ids, attn, yc, ys, ks = (a[:args.limit] for a in (ids, attn, yc, ys, ks))

    enc = make_encoder(word_table().to(device), args)
    if args.init == "jepa":
        enc.load_state_dict(torch.load(C.ARTIFACT_DIR / args.from_run / "encoder.pt",
                                       map_location="cpu"))
    enc.to(device).eval()
    for p in enc.parameters():
        p.requires_grad_(False)

    labels = sorted(set(yc.tolist()))
    to_id = {lab: i for i, lab in enumerate(labels)}
    yc_t = torch.tensor([to_id[v] for v in yc])
    ys_t = torch.tensor(np.nan_to_num(ys), dtype=torch.float32).unsqueeze(1)
    ks_t = torch.tensor(ks, dtype=torch.float32).unsqueeze(1)
    w_sal = np.ones(len(yc), dtype=np.float32)
    w_sal[ks] = compute_salary_bin_weights(ys[ks])
    w_sal_t = torch.tensor(w_sal).unsqueeze(1)
    counts = np.bincount(yc_t.numpy(), minlength=len(labels)).astype(np.float32)
    w_cat = torch.tensor(class_weights(counts, "effective", args.cw_beta)).to(device)
    loss_cat = nn.CrossEntropyLoss(weight=w_cat, label_smoothing=args.label_smoothing)
    loss_sal = nn.SmoothL1Loss(reduction="none", beta=args.beta_sal)
    head = MultiTaskRnn(args.dim, args.hidden, len(labels), args.dropout,
                        spatial_dropout=args.spatial_dropout,
                        conv_channels=args.conv_channels).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def feats(I, A, idx):
        m = torch.from_numpy(A[idx]).bool().to(device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                             enabled=device.type == "cuda"):
            h = enc(torch.from_numpy(I[idx]).long().to(device), m)
        return h.float(), m

    out = C.ARTIFACT_DIR / args.run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    hist = out / "history.jsonl"
    hist.write_text("", encoding="utf-8")
    gen = torch.Generator().manual_seed(args.seed)
    best_f1, best_mae, waited, t0 = -np.inf, np.inf, 0, time.time()
    for epoch in range(1, args.epochs + 1):
        head.train()
        tot, seen = 0.0, 0
        for idx in batches(len(yc), args.batch, True, gen):
            i = idx.numpy()
            lc, ls_ = head(*feats(ids, attn, i))
            l_sal = (loss_sal(ls_, ys_t[i].to(device)) * w_sal_t[i].to(device) * ks_t[i].to(device))
            loss = (loss_cat(lc, yc_t[i].to(device))
                    + args.alpha_sal * l_sal.sum() / (ks_t[i].sum().to(device) + 1e-6))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), args.clip)
            opt.step()
            tot, seen = tot + loss.item() * len(i), seen + len(i)

        head.eval()
        with torch.no_grad():
            outs = [head(*feats(ids_ev, attn_ev, idx.numpy()))
                    for idx in batches(len(yc_ev), args.batch, False, gen)]
        proba = torch.softmax(torch.cat([o[0] for o in outs]), 1).cpu().numpy()
        p_sal = torch.cat([o[1] for o in outs]).squeeze(1).cpu().numpy()
        pred_cat = np.array(labels)[proba.argmax(1)]
        m_cat = E.classification_metrics(yc_ev, pred_cat, labels)
        m_sal = E.regression_metrics(ys_ev[ks_ev], p_sal[ks_ev])
        line = {"epoch": epoch, "train_loss": tot / seen, "val_macro_f1": m_cat["f1_macro"],
                "val_mae_trieu": m_sal["mae_trieu"], "val_r2_log": m_sal["r2_log"],
                "seconds": round(time.time() - t0, 1)}
        with hist.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")
        print("  " + "  ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}"
                               for k, v in line.items()), flush=True)
        improved = False
        if m_cat["f1_macro"] > best_f1:
            best_f1, best_ep, best_m_cat, best_cat, best_proba = (m_cat["f1_macro"], epoch, m_cat,
                                                                  pred_cat, proba)
            save_atomic(head.state_dict(), out / "best.pt")
            improved = True
        if m_sal["mae_trieu"] < best_mae:
            best_mae, best_ep_sal, best_m_sal, best_sal = m_sal["mae_trieu"], epoch, m_sal, p_sal
            save_atomic(head.state_dict(), out / "best_sal.pt")
            improved = True
        waited = 0 if improved else waited + 1
        if waited >= args.patience:
            print(f"  dừng sớm ở epoch {epoch}")
            break
    seconds = time.time() - t0

    pd.DataFrame({"y_true_cat": yc_ev, "y_pred_cat": best_cat,
                  "y_true_sal": np.where(ks_ev, np.expm1(ys_ev), np.nan),
                  "y_pred_sal": np.expm1(best_sal),
                  "proba_cat": [r.tolist() for r in best_proba.astype(np.float32)]}
                 ).to_parquet(out / "predictions_dev.parquet")
    env = environment(device, SOURCE)
    src = f"jepa[{args.from_run}]" if args.init == "jepa" else "random-init"
    env["encoder"] = f"{src}(d={args.dim},L={args.layers}) on {env['encoder']} word table"
    model = (f"{src}-frozen(d={args.dim},L={args.layers})+rnn-mtl(h={args.hidden},"
             f"c={args.conv_channels},cw(effective,b={args.cw_beta:g}),"
             f"ls={args.label_smoothing:g},huber(b={args.beta_sal:g}),a={args.alpha_sal:g})")
    (out / "env.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps({
        "run_id": args.run_id, "task": "mtl", "model": model, "eval": "dev",
        "best_epoch": best_ep, "best_epoch_sal": best_ep_sal, "epochs_run": epoch,
        "seconds": seconds, "metrics_cat": best_m_cat, "metrics_sal": best_m_sal,
        "labels": labels, "config": vars(args), "env": env, "encoder": src,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    headline = (f"macroF1={best_m_cat['f1_macro']:.4f} · MAE={best_m_sal['mae_trieu']:.2f}tr · "
                f"R2log={best_m_sal['r2_log']:.3f}")
    if not args.no_log:
        with C.RESULTS_LOG.open("a", encoding="utf-8") as fh:
            fh.write(f"| {args.run_id} | {datetime.now(timezone.utc).strftime('%m-%d %H:%M')} "
                     f"| mtl | {model} | title+desc+req | mask+nosegment | dev | {len(yc_ev)} "
                     f"| {headline} | {seconds:.1f}s |\n")
    print(f"[{args.run_id}] best epoch cat {best_ep} · sal {best_ep_sal} / {epoch} · {headline}")


def _common(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0, help="chỉ lấy N dòng train (thử nhanh)")
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--enc-dropout", type=float, default=0.1)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pretrain")
    _common(p)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.04)
    p.add_argument("--warmup", type=float, default=0.05, help="tỉ lệ bước warmup")
    p.add_argument("--tau", type=float, default=0.996, help="EMA đầu, tăng tuyến tính tới 1")
    p.add_argument("--pred-dim", type=int, default=128)
    p.add_argument("--pred-layers", type=int, default=2)
    p.add_argument("--dev-rows", type=int, default=1024)
    p.add_argument("--min-std", type=float, default=0.01)

    f = sub.add_parser("finetune")
    _common(f)
    f.add_argument("--init", choices=["jepa", "random"], required=True)
    f.add_argument("--from-run", default=None, help="run_id của pretrain (với --init jepa)")
    # Mặc định = cấu hình dl-mtl-caf-rnn-b01-cw-s42-v2, để chỉ encoder là khác.
    f.add_argument("--lr", type=float, default=3e-4)
    f.add_argument("--weight-decay", type=float, default=1e-2)
    f.add_argument("--hidden", type=int, default=256)
    f.add_argument("--conv-channels", type=int, default=64)
    f.add_argument("--dropout", type=float, default=0.3)
    f.add_argument("--spatial-dropout", type=float, default=0.2)
    f.add_argument("--cw-beta", type=float, default=0.999)
    f.add_argument("--label-smoothing", type=float, default=0.05)
    f.add_argument("--beta-sal", type=float, default=0.1)
    f.add_argument("--alpha-sal", type=float, default=5.0)
    f.add_argument("--patience", type=int, default=12)
    f.add_argument("--no-log", action="store_true", help="không ghi dòng vào 04-results.md")
    f.set_defaults(epochs=80)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    {"pretrain": pretrain, "finetune": finetune}[args.cmd](args)


if __name__ == "__main__":
    main()
