# category — 2026-10-17

🔴 **Chưa kết luận được: mô hình chưa qua `nm-eval` và bộ dữ liệu đang hạng F.**

**Cần thầy/cô cho ý kiến:** khi đoán sai nhóm nghề thì hậu quả nặng hay nhẹ, và sản phẩm có hiện 3 gợi ý hay chỉ 1? Khuyến nghị: giữ macro-F1 làm chỉ số chính cho đến khi có câu trả lời.

## Bài toán
Phân loại một tin tuyển dụng tiếng Việt vào 1 trong 16 nhóm nghề, từ tiêu đề, mô tả và yêu cầu. Nhãn là nhãn gốc của nguồn dữ liệu và được coi là chuẩn (không đổi nhãn). Dữ liệu: 47.707 tin sau khi loại trùng hệt; chia `train` 34.354 / `dev` 3.812 / `test` 9.541 (seed 20260826). Mục đích sử dụng chưa được ghi (`problem.md`).

## Kết quả (trên `dev`, chỉ để chọn mô hình; `test` chưa chấm)
| Cách làm | macro-F1 | accuracy | Nguồn |
|---|---|---|---|
| Luôn đoán lớp đông nhất | 0,0214 | 0,2062 | `artifacts/eda/summary.json`, `floors` |
| Probe tuyến tính trên vector PhoBERT đóng băng | 0,5898 | 0,6388 | `probe-cat-s2` |
| PhoBERT đóng băng + đầu Dense | 0,6030 | 0,6501 | `dl-cat-ce-cpu-0920` |
| CafeBERT 512 đóng băng + đầu Bi-GRU‖Bi-LSTM→CNN, 3 seed | 0,6152 ± 0,0084 | seed 42: 0,6679 | `dl-cat-cafe-rnn-ce-s42/43/44` |

Hơn nhau 0,0122 macro-F1 giữa mô hình tốt nhất và PhoBERT Dense, chỉ bằng khoảng 1,5 độ lệch chuẩn giữa các seed, nên chưa chắc khác biệt là thật. Các lần chạy RNN dùng `cuda`, không phải `cpu` như quy ước (Rule 13).

## Chỗ còn yếu
- `nhóm_nghề_khác`: F1 0,000, chỉ 25 dòng trên `dev`. Kèm theo `f1_macro_no_junk` = 0,6608 (seed 42).
- `du_lịch_nhà_hàng_khách_sạn_dịch_vụ`: recall 0,4628 (n = 443); `kinh_doanh_…`: recall 0,5840 (n = 786). Cặp nhầm nhiều nhất: du_lịch → kinh_doanh, 125 dòng.
- Cosine giữa tâm lớp của các cặp nhầm: thô ≥ 0,992 ở mọi cặp nên không dùng được; sau khi trừ trung bình `train` là 0,469 / 0,242 / −0,055 (cặp đối chứng −0,276). Ngưỡng chưa hiệu chuẩn cho cosine đã trừ trung bình.

## Rủi ro
- Dữ liệu hạng F, 6 phát hiện (2 nghiêm trọng, 2 lớn, 2 nhỏ): chưa ghi nguồn, giấy phép và xử lý thông tin cá nhân; `problem.md` chưa duyệt; chưa đo được độ tin cậy của nhãn (không có người gán nhãn thứ hai); lớp rác chỉ 25 dòng trên `dev`; nhãn `dev` lệch so với `train` (χ² p = 1,78e-56, PSI 0,093).
- Kiểm tra gần-trùng (MinHash) và số mẫu từng lớp trên `test` chưa chạy (Rule 5).

## Đã thử nhưng bỏ
- Chưa bỏ hướng nào trong `problem.md`.

## Việc tiếp theo & cần quyết định
1. Duyệt `problem.md` (trả lời 3 câu hỏi mở) và ghi giấy phép dữ liệu. Đây là điều kiện để dữ liệu thoát hạng F.
2. Chọn hướng thử trước. Khuyến nghị: các biến thể văn bản đầu vào của CafeBERT (`strip-company-intro`, `field-order-512`, `cafebert-preprocess-ablation`), vì nhãn gốc là chuẩn và không đổi.
3. Quyết định về `nhóm_nghề_khác` (`junk-class-decision`): khuyến nghị chỉ báo `f1_macro_no_junk` kèm macro-F1, không đổi nhãn.

<details><summary>Chi tiết kỹ thuật</summary>
Links: <a href="problem.md">problem.md</a> (draft) · <a href="data/v2-85862b06.md">data/v2-85862b06.md</a> (grade F) · eval.md: chưa có · experiment.md: chưa có.
Bảng theo `docs/04-results.md` và `artifacts/<run_id>/metrics.json`; `artifacts/eda/class-centroid-cosine.json`. Số liệu giữ nguyên như đã đo, thập phân dùng dấu phẩy theo Rule 6.
</details>
