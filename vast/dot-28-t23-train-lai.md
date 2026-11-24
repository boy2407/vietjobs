# Đợt 28 — T23: train lại 6 cấu trúc last4 × 5 encoder trên `splits-sal17-11c`

**Task:** T23.2–T23.11b ([tasks/T23-train-lai.md](../tasks/T23-train-lai.md)). **Script:** `scripts/vast_run_t23.sh`.
**Trạng thái:** CHƯA THUÊ. Trần chi phí **15 USD** cho cả đợt.

**Dữ liệu.** `data/processed/splits-sal17-11c` (`VIETJOBS_SPLITS`), họ `masked`, 11 cột nối:
train 21.801 / dev 2.422 / test 6.056 (lược đồ v3, AGENTS.md Rule 5 mục 7). Không so với `splits/`.

**Cấu hình chung (cả 6 cấu trúc):** `--last4`, lr 3e-4, weight decay 0,01, dropout 0,3, 80 epoch,
`patience` 12. Dense hidden 256; RNN `--hidden 100 --conv-channels 50` (RNN đơn và MTL RNN).
`category`: `--family masked` + CE + cw effective 0,999 + ls 0,05. `salary`: SmoothL1 β 0,1 + `--class-weight`.
MTL: α = 5. Dense (cả MTL Dense) seed 42–46 trên `cpu`; RNN seed 42 trên `cuda`.

**Máy:** chưa chọn. Cần: GPU ≥ 16 GB, RAM ≥ 64 GB, ổ ≥ 150 GB (cache token BamiBERT 1024 last4
train ước ~46 GB), CPU mạnh từng lõi (kinh-nghiem.md: RNN bị chặn ở CPU một lõi).
`ssh -p PORT root@IP` — điền khi thuê.

**Ước tính (chưa đo, suy từ Đợt 27: RNN lương 34 nghìn dòng, 40/8 epoch ≈ 75 phút/run trên 5060 Ti):**

| Phần | Ước tính |
|---|---|
| Encode pooled 5 encoder (T23.2) | ~1 giờ |
| 75 run Dense trên `cpu` (T23.3–T23.5) | ~2,5 giờ |
| Encode token 5 encoder (T23.7) | ~1 giờ |
| 15 run RNN trên `cuda` (T23.8–T23.10) | ~18 giờ |
| Chấm test + ext37k (T23.6, T23.6b, T23.11, T23.11b) | ~2 giờ |
| **Tổng** | **~25 giờ → ~5–9 USD ở 0,20–0,35 USD/giờ** |

## Bước

| Bước | Lệnh / việc | Trạng thái |
|---|---|---|
| 1 | Kiểm máy: `nvidia-smi`, `nproc`, `free -g`, `df -h / /workspace`, phiên bản torch + `cuda.get_device_capability()` | chưa |
| 2 | venv: torch của image + numpy, pandas, scikit-learn, transformers 4.46.3, pyarrow, sentencepiece, protobuf, scipy; ghi phiên bản vào đây | chưa |
| 3 | Đẩy `src/`, `scripts/`, `data/processed/manifest.json`, `splits-sal17-11c/{train,dev}.csv`, `docs/04-results.md` (không đẩy `test.csv` ở bước này) | chưa |
| 4 | `tmux new -s t23` → `bash scripts/vast_run_t23.sh pooled dense` | chưa |
| 5 | Kéo 75 run Dense + log + `04-results.md` về Mac, kiểm `best.pt` | chưa |
| 6 | Chấm `test` + ext37k 75 run Dense (T23.6, T23.6b) — tác giả gõ lệnh, `--confirm-test` | chưa (script chấm chưa viết) |
| 7 | Báo chi phí đã dùng cho tác giả trước T23.7 | chưa |
| 8 | `bash scripts/vast_run_t23.sh rnn` | chưa |
| 9 | Kéo 15 run RNN về Mac, chấm `test` + ext37k (T23.11, T23.11b) | chưa |
| 10 | Checklist `vast/README.md`, tác giả destroy | chưa |

## Run

| Nhóm | run_id | Số run | Dev | Test | Ext37k |
|---|---|---|---|---|---|
| Dense cat | `dl-cat-{enc}-dense-sal17-s{42..46}` | 25 | — | — | — |
| Dense sal | `dl-sal-{enc}-dense-sal17-s{42..46}` | 25 | — | — | — |
| MTL Dense | `dl-mtl-{enc}-dense-sal17-s{42..46}` | 25 | — | — | — |
| RNN cat | `dl-cat-{enc}-rnn-sal17-s42` | 5 | — | — | — |
| RNN sal | `dl-sal-{enc}-rnn-sal17-s42` | 5 | — | — | — |
| MTL RNN | `dl-mtl-{enc}-rnn-sal17-s42` | 5 | — | — | — |

`{enc}` ∈ phobert, cafebert, xlmr, visobert, bamibert. Số điền từ `artifacts/<run_id>/metrics.json`.

## Lệnh (thay PORT, IP)

```bash
# lên
tar czf - src scripts data/processed/manifest.json data/processed/splits-sal17-11c/train.csv \
  data/processed/splits-sal17-11c/dev.csv docs/04-results.md \
  | ssh -p PORT root@IP "mkdir -p /workspace/vietjobs && tar xzf - -C /workspace/vietjobs"
# chạy (trong tmux)
ssh -p PORT root@IP "cd /workspace/vietjobs && bash scripts/vast_run_t23.sh pooled dense"
# về — giữ best.pt
ssh -p PORT root@IP "cd /workspace/vietjobs && tar czf - artifacts/dl-*-sal17-* logs docs/04-results.md" | tar xzf - -C .
```
