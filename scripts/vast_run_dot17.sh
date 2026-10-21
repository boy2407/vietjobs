#!/usr/bin/env bash
# ==============================================================================
# Đợt 17: Vast.ai Run Script - Huấn luyện 4 MTL RNN & Chấm điểm toàn bộ Test/Ext37K
# ==============================================================================
set -e
cd /workspace/vietjobs
export PYTHONPATH=src
mkdir -p logs

LOG_ALL="logs/dot17-all.txt"
echo "=== BẮT ĐẦU ĐỢT 17 VAST.AI: $(date) ===" | tee -a "$LOG_ALL"

# ------------------------------------------------------------------------------
# 1. Huấn luyện 4 mô hình Multi-Task RNN còn lại
# ------------------------------------------------------------------------------
echo "--- [GIAI ĐOẠN 1] Huấn luyện 4 Multi-Task RNN (CafeBERT, ViSoBERT, XLM-R, BamiBERT) ---" | tee -a "$LOG_ALL"
bash scripts/vast_run_mtl_rnn.sh 2>&1 | tee -a "$LOG_ALL"

# ------------------------------------------------------------------------------
# 2. Chấm điểm toàn bộ 5 Multi-Task RNN trên Test & Ext37K
# ------------------------------------------------------------------------------
echo "--- [GIAI ĐOẠN 2] Chấm điểm MTL RNN trên Test (9.541 tin) & Ext37K (6.111 tin) ---" | tee -a "$LOG_ALL"
/venv/main/bin/python scripts/eval_mtl_all.py 2>&1 | tee -a "$LOG_ALL"

# ------------------------------------------------------------------------------
# 3. Chấm điểm toàn bộ các mô hình Dense (MTL Dense & Salary Dense) trên Test & Ext37K
# ------------------------------------------------------------------------------
echo "--- [GIAI ĐOẠN 3] Chấm điểm MTL Dense & Salary Dense (5 seeds) trên Test & Ext37K ---" | tee -a "$LOG_ALL"
/venv/main/bin/python scripts/eval_dense_all.py 2>&1 | tee -a "$LOG_ALL"

# ------------------------------------------------------------------------------
# 4. Xuất kết quả ra file Excel / Dashboard
# ------------------------------------------------------------------------------
echo "--- [GIAI ĐOẠN 4] Xuất kết quả tổng hợp ---" | tee -a "$LOG_ALL"
/venv/main/bin/python scripts/export_runs.py 2>&1 | tee -a "$LOG_ALL"

echo "=== ĐỢT 17 ĐÃ HOÀN TẤT TRỌN VẸN: $(date) ===" | tee -a "$LOG_ALL"
echo "Bây giờ bạn có thể chạy lệnh kéo kết quả về Mac:"
echo "ssh -p PORT root@IP 'cd /workspace/vietjobs && tar czf - artifacts/dl-mtl-* artifacts/dl-sal-* artifacts/eval_* logs' | tar xzf - -C ."
