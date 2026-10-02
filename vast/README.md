# Chạy trên vast.ai — theo dõi

Mỗi đợt thuê máy là **một file** trong thư mục này (AGENTS.md, Rule 14).
Script chạy nằm ở `scripts/vast_run_*.sh`; kết quả gốc nằm ở `artifacts/<run_id>/`.

| Đợt | File | Task | Script | Trạng thái |
|---|---|---|---|---|
| 1 | [dot-1-phobert-rnn.md](dot-1-phobert-rnn.md) | T8.14/T8.15 | `scripts/vast_run_rnn.sh` | xong |
| 2 | [dot-2-cafebert-rnn.md](dot-2-cafebert-rnn.md) | T8.16/T8.17 | `scripts/vast_run_cafe_rnn.sh` | xong |
| 3 | [dot-3-cafebert-mtl-rnn.md](dot-3-cafebert-mtl-rnn.md) | T4.1 | `scripts/vast_run_cafe_mtl.sh` | chưa thuê |

Kinh nghiệm chung cho mọi đợt: [kinh-nghiem.md](kinh-nghiem.md).

## Lệnh dùng chung (thay PORT và IP)

```bash
ssh -p PORT root@IP
# kéo kết quả về Mac — rsync của macOS bị treo, dùng tar qua ssh, giữ cả best.pt
ssh -p PORT root@IP "cd /workspace/vietjobs && tar czf - artifacts/<run_glob> logs" | tar xzf - -C .
```

## Checklist trước khi destroy instance

1. Đủ mọi run của đợt có `metrics.json` + `history.jsonl` + log trên Mac.
2. `best.pt` của mọi run đã nằm trên Mac (Đợt 2 mất 5 file vì bỏ qua bước này).
3. Bảng của file đợt đã điền số từ `metrics.json`.
