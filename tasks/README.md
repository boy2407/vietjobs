# tasks/ — overview

Detail sheets for board sections that outgrew [TASKS.md](../TASKS.md). Each sheet holds its
own rows, status and log; status is authoritative **in the sheet**.

| ID | Task | Status | Assignee | Last updated |
|---|---|---|---|---|
| T14 | Label noise, Vietnamese text and salary follow-ups (12 stories, [sheet](T14-nhieu-nhan-va-van-ban.md)) | doing (T14.1, T14.3, T14.10 done; T14.2, T14.4 human: Nghĩa) | Claude (main session) | 2026-10-17 |
| T2.6–T2.8 | Cấu trúc nhóm train/dev, top-1 trong tập nhãn, soft target (3 stories, [thư mục](user-stories/README.md)) | done | Claude (main session) | 2026-10-17 |
| T15 | Salary on dataset v8, PhoBERT and CafeBERT, Dense and RNN, on vast.ai ([sheet](T15-v8-luong.md)) | todo (T15.1 needs the author to rent) | Claude | 2026-10-17 |
| T16 | Train lại 15 run T11 có dev lệch Mac, đuôi `-r2` ([sheet](T16-train-lai-r2.md)) | doing (Đợt 8, 9 đang chạy) | Claude | 2026-10-19 |
| T17 | Đợt 17: 4 MTL RNN còn lại + chấm Test/Ext37K ([sheet](T17-dot17-mtl-rnn-cham-diem.md)) | doing (train trên vast 5090) | Claude | 2026-10-21 |
| T18 | MTL RNN `-v2` còn lại (XLM-R, BamiBERT), checkpoint an toàn, sheet và tài liệu ([sheet](T18-mtl-rnn-v2-con-lai.md)) | todo (chờ thuê máy và `gws auth login`) | Claude | 2026-10-21 |
| T19 | Tuần 9: 1k tin ext37k-sal (CM + lương vs dev), `nhóm_nghề_khác`, tứ phân vị lương ≥ 100tr ([sheet](T19-tuan9-phan-tich.md)) | done | Claude (main session) | 2026-10-30 |
| T20 | Viết lại báo cáo theo mẫu KLTN UIT, chốt CafeBERT, xuất LaTeX ([sheet](T20-bao-cao-theo-mau-uit.md)) | doing | Claude (main session) | 2026-11-05 |
| T21 | Trùng gần test↔train mà `group_key` bỏ sót: đo, ghi báo cáo, quyết cách xử lý ([sheet](T21-trung-gan-test-train.md)) | done | Claude (main session) | 2026-11-22 |
| T22 | Salary-disclosed train, conflict filter, extra columns, oversampling — CafeBERT Dense R0–R4 ([sheet](T22-cot-moi-loc-xung-dot.md)) | doing | Claude (main session) | 2026-11-23 |
| T23 | Split `splits-sal17`: tin có lương, bỏ trùng 17 cột khác nhãn, chia lại 8:2→9:1 ([sheet](T23-split-sal17.md)) | doing | Claude (main session) | 2026-11-24 |
| T23-train | Retrain 6 structures × 5 encoders on `splits-sal17-11c` (90 runs, Dense 5 seeds, shared hyper-parameters, test + ext37k), one vast rental ≤ 15 USD ([sheet](T23-train-lai.md)) | todo (author rents rental 28) | Claude (main session) | 2026-11-24 |
