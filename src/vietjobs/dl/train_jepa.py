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

import numpy as np
import torch

from .. import config as C
from .. import dataset as D
from . import encode as ENC
from . import text as T
from .jepa import JepaEncoder, Predictor, block_mask, ema_update, jepa_step
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
    return ap


def main() -> None:
    args = build_parser().parse_args()
    {"pretrain": pretrain}[args.cmd](args)


if __name__ == "__main__":
    main()
