"""Baseline học sâu — PhoBERT đóng băng + dense, mỗi bài toán một mạng riêng.

    python -m vietjobs.dl.train_dl --task category
    python -m vietjobs.dl.train_dl --task salary

Chạy trên vector đã cache (``dl/encode.py``), nên một epoch tính bằng giây. Mỗi
lần chạy ghi ``artifacts/<run_id>/`` gồm:

    config.json      mọi siêu tham số + seed, đủ để chạy lại
    env.json         thiết bị, phiên bản torch/transformers
    history.jsonl    một dòng mỗi epoch — đường học nằm ở đây, không ở metrics
    metrics.json     số cuối cùng, cùng bộ khoá cho mọi lần chạy
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
from .focal_loss import FocalLoss  # bản của itakurah/focal-loss-pytorch
from .heads import BiGruLstmCnn, DenseHead


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=C.ROOT,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return ""


def environment(device) -> dict:
    # numpy/pandas/sklearn nằm trong danh sách này vì một lý do đo được: ngày
    # 2026-09-20 `dl-cat-s2` (chạy 09-15) không tái lập được — 0,6025 → 0,6013,
    # khoảng 5/3812 tin đổi nhãn — trong khi mã nguồn cũ chạy lại hôm nay cũng
    # ra 0,6013, tức code vô can. Không truy được nguyên nhân vì đúng ba thư
    # viện này không được ghi lại. Xem TASKS.md §5, mục 2026-09-20.
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
        "encoder": ENC.MODEL_NAME,
        "git_sha": _git("rev-parse", "HEAD") or "uncommitted",
        "git_dirty": bool(_git("status", "--porcelain")),
    }


def load_task(split: str, task: str, max_len: int, head: str = "dense"):
    """Trả về ``(X, mask, y, frame)`` — frame giữ lại để cắt lát kết quả theo ngành.

    ``head="dense"``  → ``X`` là ma trận ``[n, 768]`` đã gộp, ``mask`` là ``None``.
    ``head="rnn"``    → ``X`` là **memmap** ``[n, T, 768]`` float16 chưa gộp và
    ``mask`` là ``[n, T]`` uint8. Memmap không được index bằng mảng boolean (sẽ
    nạp cả 13,5 GB vào RAM), nên ở đây chỉ trả về **chỉ số dòng được giữ**; vòng
    huấn luyện đọc theo batch qua chỉ số đó.
    """
    df = D.load_split(split)
    if head == "rnn":
        X, mask = ENC.token_cache_for(split, task, max_len=max_len)
    else:
        X, mask = ENC.embeddings_for(split, task, max_len=max_len), None
    if len(X) != len(df):
        raise SystemExit(f"cache {split} có {len(X)} dòng nhưng split có {len(df)} "
                         "— chạy lại `python -m vietjobs.dl.encode --force`")
    if task == C.TASK_CATEGORY:
        keep = df["category"].notna().to_numpy()
        y = df.loc[keep, "category"].to_numpy()
    elif task == C.TASK_SALARY:
        keep = ((df["salary_disclosed"] == 1) & df["salary_mid"].notna()).to_numpy()
        y = np.log1p(df.loc[keep, "salary_mid"].to_numpy(dtype=np.float32))
    else:
        keep = df["salary_disclosed"].notna().to_numpy()
        y = df.loc[keep, "salary_disclosed"].to_numpy(dtype=np.int64)
    if head == "rnn":
        rows = np.flatnonzero(keep)
        return (X, rows), mask[rows], y, df[keep]
    return X[keep], None, y, df[keep]


def token_stats(X, rows: np.ndarray, mask: np.ndarray, chunk: int = 512):
    """μ/σ theo từng chiều, tính trên **token thật** của `train`, đọc memmap theo khối.

    Không gọi ``X.mean(0)``: mảng 13,5 GB sẽ được nạp trọn vào RAM. Một lượt cộng
    dồn ``sum`` và ``sum²`` là đủ, và cho đúng con số mà một lượt đầy đủ sẽ cho.
    """
    total = np.zeros(ENC.HIDDEN, dtype=np.float64)
    total_sq = np.zeros(ENC.HIDDEN, dtype=np.float64)
    n_tok = 0
    t0 = time.time()
    for i in range(0, len(rows), chunk):
        if i and (i // chunk) % 10 == 0:
            # Lượt này đọc tuần tự 13,5 GB và chạy trước epoch đầu tiên. Không in
            # gì thì người chạy tưởng treo — đã xảy ra thật trên Colab 2026-09-21.
            print(f"  thống kê chuẩn hoá: {i:,}/{len(rows):,} dòng "
                  f"({100 * i / len(rows):4.1f}%, {time.time() - t0:5.0f}s)", flush=True)
        idx = rows[i:i + chunk]
        xb = np.asarray(X[idx], dtype=np.float32)          # [c, T, 768]
        mb = mask[i:i + chunk][..., None].astype(np.float32)
        total += (xb * mb).sum((0, 1))
        total_sq += ((xb ** 2) * mb).sum((0, 1))
        n_tok += int(mb.sum())
    mu = total / max(n_tok, 1)
    var = np.clip(total_sq / max(n_tok, 1) - mu ** 2, 0, None)
    return mu.astype(np.float32)[None, None, :], np.sqrt(var).astype(np.float32)[None, None, :]


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
    ap.add_argument("--head", default="dense", choices=["dense", "rnn"],
                    help="dense: đọc vector đã gộp trung bình (mặc định); "
                         "rnn: đọc từng token từ cache --pooling none, Bi-GRU ‖ Bi-LSTM "
                         "→ CNN → pooling có mặt nạ (T8)")
    ap.add_argument("--branches", default="both", choices=["both", "gru", "lstm"],
                    help="chỉ dùng với --head rnn; ablation T8.5")
    ap.add_argument("--spatial-dropout", type=float, default=0.2,
                    help="chỉ dùng với --head rnn: tắt cả một kênh trên mọi bước thời gian")
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--class-weight", action="store_true",
                    help="cân bằng lớp cho bài phân lớp (lệch 27:1)")
    ap.add_argument("--loss", default="ce", choices=["ce", "focal"],
                    help="chỉ áp dụng cho phân lớp; focal cân theo độ khó từng mẫu "
                         "thay vì độ hiếm của lớp — dùng cùng --class-weight được")
    ap.add_argument("--focal-gamma", type=float, default=2.0,
                    help="gamma của focal loss; gamma=0 trùng CrossEntropyLoss")
    # Mặc định cpu, không phải auto: head dense chỉ 768→256→128→16 nên mps không
    # nhanh hơn (12-16s cả hai), nhưng mps **không tái lập được** — đo 2026-09-20,
    # cùng seed, 8 lần chạy trải 0,6012-0,6041 macro-F1, trong khi cpu cho đúng
    # một số mỗi lần. Đổi mặc định là cái giá rẻ nhất để mua lại tính tái lập.
    ap.add_argument("--device", default="cpu",
                    help="cpu|mps|cuda|auto — mặc định cpu vì tái lập được; "
                         "mps cho số khác nhau giữa các lần chạy")
    ap.add_argument("--eval", default="dev", choices=["dev", "test"])
    ap.add_argument("--confirm-test", action="store_true")
    ap.add_argument("--no-log", action="store_true",
                    help="không ghi vào 04-results.md — dành cho lần chạy thử "
                         "tái lập, nhật ký chỉ nhận lần chạy thật")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--seed", type=int, default=C.RANDOM_SEED)
    args = ap.parse_args()

    if args.eval == "test" and not args.confirm_test:
        raise SystemExit("chấm test cần --confirm-test (docs/03-protocol.md, điều 2)")
    if args.loss == "focal" and args.task == C.TASK_SALARY:
        raise SystemExit("--loss focal chỉ dùng cho phân lớp, salary đã dùng Huber")
    if args.head == "rnn" and args.batch > 64 and args.device != "cuda":
        # 256 × 768 float32 = 786 KB mỗi tin; batch 256 là 200 MB chỉ riêng đầu vào,
        # chưa kể trạng thái RNN — nặng cho CPU/MPS. Trên GPU (16 GB+) 200 MB không
        # đáng lo, và hạ batch chỉ làm chậm do giảm độ song song — không cap ở đó.
        print(f"  --head rnn trên {args.device}: hạ batch {args.batch} → 64 (mỗi tin nặng gấp 256 lần)")
        args.batch = 64

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = ENC.pick_device(args.device)
    is_reg = args.task == C.TASK_SALARY

    Xtr, mtr, ytr, _ = load_task("train", args.task, args.max_len, args.head)
    Xev, mev, yev, ev_df = load_task(args.eval, args.task, args.max_len, args.head)
    is_rnn = args.head == "rnn"
    n_train = len(ytr)

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
    if is_rnn:
        mu, sigma = token_stats(*Xtr, mtr)
    else:
        mu, sigma = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True)
    if args.no_standardize:
        mu, sigma = np.zeros_like(mu), np.ones_like(sigma)
    sigma = np.clip(sigma, 1e-6, None)

    if is_rnn:
        # Không dựng sẵn tensor: 13,5 GB không nằm trong RAM. `take` đọc đúng các
        # dòng của batch từ memmap, chuẩn hoá rồi mới đẩy lên device.
        def take(X, mask, idx):
            arr, rows = X
            order = idx.numpy() if torch.is_tensor(idx) else idx
            xb = (np.asarray(arr[rows[order]], dtype=np.float32) - mu) / sigma
            return (torch.from_numpy(xb).to(device),
                    torch.from_numpy(mask[order].astype(np.uint8)).to(device))
        in_dim = ENC.HIDDEN
    else:
        in_dim = Xtr.shape[1]
        Xtr_t = torch.tensor((Xtr - mu) / sigma, dtype=torch.float32)
        Xev_t = torch.tensor((Xev - mu) / sigma, dtype=torch.float32).to(device)
        # Vòng lặp gọi take(Xtr, ...) chung cho cả hai head, nên ở nhánh này Xtr/Xev
        # phải là tensor đã chuẩn hoá — không phải mảng numpy thô. Thiếu dòng này
        # thì `.to(device)` gãy trên numpy (bắt được bằng smoke run 2026-09-21).
        Xtr, Xev = Xtr_t, Xev_t

        def take(X, mask, idx):
            return X[idx].to(device), None

    if is_rnn:
        model = BiGruLstmCnn(in_dim, args.hidden, out_dim, args.dropout,
                             spatial_dropout=args.spatial_dropout,
                             branches=args.branches).to(device)
    else:
        model = DenseHead(in_dim, args.hidden, out_dim, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if is_reg:
        loss_fn = nn.SmoothL1Loss()  # đuôi lương rất dài (skew 11,8) — huber chịu đuôi tốt hơn MSE
    else:
        w = None
        if args.class_weight:
            counts = np.bincount(ytr_t.numpy(), minlength=out_dim).astype(np.float32)
            w = torch.tensor(counts.sum() / (out_dim * np.clip(counts, 1, None))).to(device)
        if args.loss == "focal":
            loss_fn = FocalLoss(gamma=args.focal_gamma, alpha=w, task_type="multi-class",
                                num_classes=out_dim)
        else:
            loss_fn = nn.CrossEntropyLoss(weight=w)

    run_id = args.run_id or (
        f"dl-{'reg' if is_reg else args.task[:3]}-h{args.hidden}"
        f"{'-cw' if args.class_weight else ''}"
        f"{f'-focal-g{args.focal_gamma:g}' if args.loss == 'focal' else ''}"
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
        total, seen = 0.0, 0
        for idx in batches(n_train, args.batch, True, gen):
            xb, mb = take(Xtr, mtr, idx)
            yb = ytr_t[idx].to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb, mb), yb)
            loss.backward()
            if args.clip:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            total += loss.detach().item() * len(idx)
            seen += len(idx)

        model.eval()
        with torch.no_grad():
            if is_rnn:
                # Đánh giá cũng theo batch: 3.812 × 256 × 768 float32 là 3 GB.
                logits = torch.cat([model(*take(Xev, mev, idx)).cpu()
                                    for idx in batches(len(yev), args.batch, False, gen)])
            else:
                logits = model(Xev_t, None).cpu()
        if is_reg:
            pred_log = logits.squeeze(1).numpy()
            m = E.regression_metrics(yev, pred_log)
            score = -m["mae_trieu"]          # càng nhỏ càng tốt → đổi dấu
            line = {"val_mae_trieu": m["mae_trieu"], "val_r2_log": m["r2_log"]}
        else:
            proba = torch.softmax(logits, dim=1).numpy()
            pred = np.array(labels)[proba.argmax(1)]
            m = E.classification_metrics(yev, pred, labels)
            score = m["f1_macro"]
            # macro-F1 là độ đo chọn mô hình (lệch lớp 27:1); f1_weighted đi kèm để
            # thấy ngay mô hình có đang bỏ rơi lớp nhỏ không — hai số càng xa nhau
            # thì phần đuôi càng bị bỏ. `f1_macro_no_junk` chỉ lưu ở metrics.json.
            line = {"val_macro_f1": m["f1_macro"], "val_f1_micro": m["f1_micro"],
                    "val_f1_weighted": m["f1_weighted"], "val_accuracy": m["accuracy"]}

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
    arch = (f"phobert-frozen+bigru-lstm-cnn(h={args.hidden},{args.branches})" if is_rnn
            else "phobert-frozen+dense")
    # F1 "chuẩn" 2·P·R/(P+R) chỉ định nghĩa cho MỘT lớp; với 16 lớp là 16 con số,
    # và macro/micro/weighted ở `metrics` chỉ là ba cách gộp chúng lại. Lưu cả 16
    # (kèm precision/recall/support) để báo cáo đọc được từng lớp mà không phải
    # chạy lại — trước 2026-09-21 `per_class_report` có sẵn nhưng không ai gọi.
    per_class = None if is_reg else E.per_class_report(yev, best_pred, labels)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    (out_dir / "env.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps({
        "run_id": run_id, "task": args.task, "model": arch,
        "pooling": "none (token-level)" if is_rnn else "masked_mean",
        "eval": args.eval, "best_epoch": best_epoch, "epochs_run": epoch,
        "seconds": seconds, "metrics": best_metrics, "per_class": per_class,
        "config": vars(args), "env": env,
        "columns": T.columns_for(args.task),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    if is_reg:
        # RMSE và R2raw (thang triệu) là hai số bài báo gốc của bộ dữ liệu công bố
        # (arXiv 2603.05262) — ghi ngay trên dòng để so được mà không mở metrics.json.
        headline = (f"MAE={best_metrics['mae_trieu']:.2f}tr · "
                    f"RMSE={best_metrics['rmse_trieu']:.2f}tr · "
                    f"R2log={best_metrics['r2_log']:.3f} · "
                    f"R2raw={best_metrics['r2_raw']:.3f} · "
                    f"±20%={best_metrics['within_20pct'] * 100:.1f}%")
    else:
        # microF1 = accuracy = F1 của arXiv:2112.11052 — ghi cả hai tên để so được
        # với bài báo mà không phải tra lại đẳng thức (xem evaluate.py).
        headline = (f"macroF1={best_metrics['f1_macro']:.4f} · "
                    f"microF1={best_metrics['f1_micro']:.4f} · "
                    f"wF1={best_metrics['f1_weighted']:.4f} · "
                    f"acc={best_metrics['accuracy']:.4f}")

    if not args.no_log:
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
            loss_tag = f",focal(g={args.focal_gamma:g})" if args.loss == "focal" else ""
            base = (f"phobert-frozen+bigru-lstm-cnn(h={args.hidden},{args.branches}" if is_rnn
                    else f"phobert-frozen+dense(h={args.hidden}")
            prep = "segment+token-level" if is_rnn else "segment"
            fh.write(f"| {run_id} | {datetime.now(timezone.utc).strftime('%m-%d %H:%M')} "
                     f"| {args.task} | {base}"
                     f"{',cw' if args.class_weight else ''}{loss_tag}) | title+desc+req | {prep} "
                     f"| {args.eval} | {best_metrics['n']} | {headline} | {seconds:.1f}s "
                     f"| {env['git_sha'][:8]}{'+dirty' if env['git_dirty'] else ''} |\n")

    print(f"\n[{run_id}] {args.task} · best epoch {best_epoch}/{epoch} · {headline}")
    print(f"  -> artifacts/{run_id}/")


if __name__ == "__main__":
    main()
