#!/usr/bin/env bash
# Đợt 16 Vast.ai: ViSoBERT encode + train Dense last4, và chấm điểm Dense last4 (CafeBERT, ViSoBERT, BamiBERT, XLM-R)
# Chạy trong tmux: bash scripts/vast_run_dot16.sh 2>&1 | tee logs/dot16-all.txt

set -euo pipefail
export PYTHONPATH=src
mkdir -p logs artifacts/embeddings

if [ -f /venv/main/bin/activate ]; then
    source /venv/main/bin/activate
elif [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

echo "=== KIỂM TRA GPU ==="
nvidia-smi

echo "=== BƯỚC 1: NHÚNG VISOBERT TRAIN RAW LEN512 LAST4 ==="
python -m vietjobs.dl.encode --encoder visobert --max-len 512 --layers 4 --splits train dev --task category --batch 64 --device cuda

echo "=== BƯỚC 2: TRAIN 5 SEED DENSE VISOBERT LAST4 ==="
for s in 42 43 44 45 46; do
    echo "Training ViSoBERT dense seed $s..."
    python -m vietjobs.dl.train_dl --task category --encoder visobert --max-len 512 --layers 4 \
        --hidden 256 --class-weight --cw-mode effective --cw-beta 0.999 --label-smoothing 0.05 \
        --device cpu --seed $s --run-id dl-cat-visobert-dense-last4-cweff999-ls05-s$s
done

echo "=== BƯỚC 3: CHẤM ĐIỂM CAFEBERT DENSE LAST4 (TEST & EXT37K) ==="
python scripts/score_cafebert_dense_last4.py

echo "=== BƯỚC 4: CHẤM ĐIỂM VISOBERT DENSE LAST4 (TEST & EXT37K) ==="
cat << 'PYSCRIPT' > scripts/score_visobert_dense_last4.py
import sys, json, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vietjobs import config as C
from vietjobs import dataset as D
from vietjobs.dl import encode as ENC
from vietjobs.dl import text as T
from score_t12 import run_kind, load_predictor, score_frame, save
from encode_t12_cpu import ext_sal_frame
from vietjobs import external as X
from eval_external import score as score_category

def predict_dense_batch(preds, texts, encoder, max_len, device, layers=4, batch=64):
    tok, enc, dev = ENC._load_model(str(device), encoder)
    outs = {r: [] for r in preds}
    n = len(texts)
    print(f"Encoding {n} texts with {encoder} (layers={layers}, batch={batch}) on {device}...", flush=True)
    t0 = time.time()
    with torch.inference_mode():
        for i in range(0, n, batch):
            batch_texts = texts[i:i + batch]
            e = tok(batch_texts, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt").to(dev)
            m = e["attention_mask"].unsqueeze(-1)
            h = ENC._hidden(enc, e, layers=layers)
            pooled = ((h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)).float()
            for r, (fn, mu, sg) in preds.items():
                mu_t = torch.tensor(mu, dtype=torch.float32, device=dev)
                sg_t = torch.tensor(sg, dtype=torch.float32, device=dev)
                x = (pooled - mu_t) / sg_t
                logits, _ = fn(x, None)
                outs[r].append(torch.softmax(logits, dim=1).cpu().numpy())
            if (i // batch) % 50 == 0 and i > 0:
                dt = time.time() - t0
                print(f"  {i + len(batch_texts):,}/{n:,} ({(i + len(batch_texts))/dt:.1f} texts/s)", flush=True)
    dt = time.time() - t0
    print(f"Done in {dt:.1f}s ({n/dt:.1f} texts/s)", flush=True)
    return {r: np.concatenate(outs[r]) for r in preds}

def main():
    runs = [f"dl-cat-visobert-dense-last4-cweff999-ls05-s{s}" for s in [42, 43, 44, 45, 46]]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    labels = sorted(set(D.load_split("train")["category"].dropna()))
    kinds = {r: run_kind(C.ARTIFACT_DIR / r) for r in runs}
    preds = {r: load_predictor(C.ARTIFACT_DIR / r, kinds[r], len(labels), device) for r in runs}
    
    # TEST
    print("=== SCORING TEST (ViSoBERT) ===", flush=True)
    df_test = D.load_split("test")
    texts_test = T.build_text(df_test, task="category", segmented=False)
    out_test = predict_dense_batch(preds, texts_test, "visobert", 512, device, layers=4, batch=64)
    test_f1s, test_accs = [], []
    for r in runs:
        res = score_frame(df_test, out_test[r], None, labels)
        save(r, "test", device, df_test, res)
        test_f1s.append(res["category"]["f1_macro"])
        test_accs.append(res["category"]["accuracy"])
    print(f"ViSoBERT TEST: F1 = {np.mean(test_f1s):.4f} ± {np.std(test_f1s, ddof=1):.4f}, Acc = {np.mean(test_accs):.4f} ± {np.std(test_accs, ddof=1):.4f}", flush=True)
    
    # EXT37K
    print("=== SCORING EXT37K (ViSoBERT) ===", flush=True)
    cw = X.load_crosswalk()
    base = ext_sal_frame(False)
    overlap, dedup = X.overlap_mask(base, X.OurIndex())
    keep = ~np.asarray(overlap)
    df_ext, cw_report = X.apply_crosswalk(ext_sal_frame(False, "raw")[keep].reset_index(drop=True), cw)
    texts_ext = T.build_text(df_ext, task="category", segmented=False)
    out_ext = predict_dense_batch(preds, texts_ext, "visobert", 512, device, layers=4, batch=64)
    cat_rows = (df_ext["n_targets"] > 0).to_numpy()
    ext_f1s, ext_accs = [], []
    for r in runs:
        proba = out_ext[r]
        cat_metrics = score_category(df_ext[cat_rows].reset_index(drop=True), proba[cat_rows], labels, 1000)
        res = {"n_file": len(base), "dedup": dedup, "crosswalk": cw_report, "category": cat_metrics}
        out_dir = C.ARTIFACT_DIR / r / "t12_score"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "ext37k-sal.json").write_text(json.dumps(
            {"run_id": r, "dataset": "ext37k-sal", "tag": "ext37k-sal: provisional, T10.1 overridden",
             "n": len(df_ext), "device": device, **res}, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        ext_f1s.append(cat_metrics.get("strict", {}).get("f1_macro", float("nan")))
        ext_accs.append(cat_metrics.get("strict", {}).get("accuracy", float("nan")))
    print(f"ViSoBERT EXT37K: F1 = {np.mean(ext_f1s):.4f} ± {np.std(ext_f1s, ddof=1):.4f}, Acc = {np.mean(ext_accs):.4f} ± {np.std(ext_accs, ddof=1):.4f}", flush=True)

if __name__ == "__main__":
    main()
PYSCRIPT
python scripts/score_visobert_dense_last4.py

echo "=== HOÀN TẤT ĐỢT 15 ==="
echo "DONE"
