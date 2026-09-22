# Nhật ký thí nghiệm (chỉ được nối thêm)

Việc chọn mô hình sử dụng `dev` (các dòng trước 2026-09-09 ghi `val`, tên cũ của cùng tập). `test` chỉ được chấm điểm một lần, vào lúc cuối.

> **Nhật ký này khởi động lại từ ngày 2026-09-08**, khi hướng đi chính của luận văn
> chuyển sang học sâu. Mốc để đối chiếu là hai lần chạy nền `dl-cat-s2`
> (macro-F1 **0,6025**) và `dl-sal-s2` (MAE **4,15 triệu**), cùng các sàn không
> dùng mô hình ở [06-baseline-dl.md](06-baseline-dl.md).

| RunID | UTC | Task | Model | Scope | Prep | Eval | n | Headline | Time | Commit |
|---|---|---|---|---|---|---|---|---|---|---|
| dl-cat-h256 | 09-08 19:10 | category | phobert-frozen+dense(h=256) | title+desc+req | segment | val | 7159 | macroF1=0.0420 · acc=0.2127 · balAcc=0.0752 · top3=0.4577 | 143.8s | 128ce9f4+dirty |
| dl-cat-h256-cw | 09-08 19:13 | category | phobert-frozen+dense(h=256,cw) | title+desc+req | segment | val | 7159 | macroF1=0.0747 · acc=0.1904 · balAcc=0.1117 · top3=0.3519 | 143.6s | 128ce9f4+dirty |
| dl-sal-h256 | 09-08 19:15 | salary | phobert-frozen+dense(h=256) | title+desc+req | segment | val | 5095 | MAE=15.21tr · MedAE=13.00tr · R2log=-27.306 · ±20%=0.0% | 86.9s | 128ce9f4+dirty |
| dl-cat-v2 | 09-08 20:30 | category | phobert-frozen+dense(h=256) | title+desc+req | segment | val | 7159 | macroF1=0.5987 · acc=0.6493 · balAcc=0.6048 · top3=0.9257 | 2119.8s | 128ce9f4+dirty |
| dl-cat-v2-nostd | 09-08 20:40 | category | phobert-frozen+dense(h=256) | title+desc+req | segment | val | 7159 | macroF1=0.5934 · acc=0.6466 · balAcc=0.5917 · top3=0.9250 | 580.3s | 128ce9f4+dirty |
| dl-cat-v2-cw | 09-08 21:29 | category | phobert-frozen+dense(h=256,cw) | title+desc+req | segment | val | 7159 | macroF1=0.5637 · acc=0.5913 · balAcc=0.6687 · top3=0.9012 | 2912.7s | 128ce9f4+dirty |
| dl-sal-v2 | 09-08 21:36 | salary | phobert-frozen+dense(h=256) | title+desc+req | segment | val | 5095 | MAE=4.83tr · MedAE=2.86tr · R2log=0.381 · ±20%=47.1% | 418.4s | 128ce9f4+dirty |
| dl-sal-v3-long | 09-08 22:11 | salary | phobert-frozen+dense(h=256) | title+desc+req | segment | val | 5095 | MAE=4.88tr · MedAE=2.86tr · R2log=0.357 · ±20%=45.9% | 1997.0s | 128ce9f4+dirty |
| probe-cat | 09-08 22:29 | category | logreg-probe(C=1.0) on phobert-frozen | title+desc+req | segment | val | 7159 | macroF1=0.5867 · acc=0.6423 · balAcc=0.5782 · top3=0.9225 | 1079.0s | probe |
| probe-sal | 09-08 22:29 | salary | ridge-probe(alpha=1.0) on phobert-frozen | title+desc+req | segment | val | 5095 | MAE=6.60tr · MedAE=3.26tr · R2log=-0.004 · ±20%=42.1% | 0.8s | probe |
| dl-cat-s2 | 09-15 14:22 | category | phobert-frozen+dense(h=256) | title+desc+req | segment | dev | 3812 | macroF1=0.6025 · acc=0.6511 · balAcc=0.6199 · top3=0.9318 | 15.0s | 12a847c1 |
| dl-cat-s2-cw | 09-15 14:23 | category | phobert-frozen+dense(h=256,cw) | title+desc+req | segment | dev | 3812 | macroF1=0.5710 · acc=0.5976 · balAcc=0.6931 · top3=0.9208 | 13.7s | 12a847c1+dirty |
| dl-sal-s2 | 09-16 06:25 | salary | phobert-frozen+dense(h=256) | title+desc+req | segment | dev | 2698 | MAE=4.15tr · MedAE=2.50tr · R2log=0.512 · ±20%=51.6% | 16.0s | 70ab8518 |
| probe-cat-s2 | 09-16 06:44 | category | logreg-probe(C=1.0) on phobert-frozen | title+desc+req | segment | dev | 3812 | macroF1=0.5898 · acc=0.6388 · balAcc=0.5969 · top3=0.9258 | 34.7s | probe |
| probe-sal-s2 | 09-16 06:45 | salary | ridge-probe(alpha=1.0) on phobert-frozen | title+desc+req | segment | dev | 2698 | MAE=4.42tr · MedAE=2.77tr · R2log=0.466 · ±20%=48.0% | 0.2s | probe |
| dl-cat-s2-ext37k-gold | 09-17 03:23 | category | phobert-frozen+dense(h=256) scored on VietJobs-37K | title+desc+req | segment+dedup+crosswalk(draft 2026-09-17, chưa duyệt) | ext37k-gold | 977 | strict: macroF1=0.3977 [0.353,0.443] · acc=0.5104 · top3=0.8204 (n=529) · lenient: hit=0.6080 · macroF1=0.4775 (n=977) | 0.7s | eval-only |
| dl-cat-s2-ext37k-test | 09-17 03:24 | category | phobert-frozen+dense(h=256) scored on VietJobs-37K | title+desc+req | segment+dedup+crosswalk(draft 2026-09-17, chưa duyệt) | ext37k-test | 3687 | strict: macroF1=0.4525 [0.421,0.479] · acc=0.5329 · top3=0.8315 (n=2053) · lenient: hit=0.6138 · macroF1=0.4935 (n=3687) | 66.7s | eval-only |
| dl-cat-focal-g1-s2 | 09-19 09:47 | category | phobert-frozen+dense(h=256,focal(g=1)) | title+desc+req | segment | dev | 3812 | macroF1=0.5938 · acc=0.6401 · balAcc=0.6174 · top3=0.9294 | 13.3s | 1cf72192+dirty |
| dl-cat-focal-s2 | 09-19 09:48 | category | phobert-frozen+dense(h=256,focal(g=2)) | title+desc+req | segment | dev | 3812 | macroF1=0.5962 · acc=0.6448 · balAcc=0.6089 · top3=0.9273 | 10.5s | 1cf72192+dirty |
| dl-cat-focal-g5-s2 | 09-19 09:48 | category | phobert-frozen+dense(h=256,focal(g=5)) | title+desc+req | segment | dev | 3812 | macroF1=0.5929 · acc=0.6348 · balAcc=0.6116 · top3=0.9250 | 10.8s | 1cf72192+dirty |
| dl-cat-ce-0920 | 09-20 03:18 | category | phobert-frozen+dense(h=256) | title+desc+req | segment | dev | 3812 | macroF1=0.6013 · F1=0.6411 · acc=0.6498 | 19.3s | 1cf72192+dirty |
| dl-cat-focal-0920 | 09-20 03:19 | category | phobert-frozen+dense(h=256,focal(g=2)) | title+desc+req | segment | dev | 3812 | macroF1=0.5962 · F1=0.6386 · acc=0.6448 | 12.6s | 1cf72192+dirty |
| dl-sal-0920 | 09-20 03:19 | salary | phobert-frozen+dense(h=256) | title+desc+req | segment | dev | 2698 | MAE=4.15tr · RMSE=8.29tr · R2log=0.512 · R2raw=0.342 · ±20%=51.6% | 16.6s | 1cf72192+dirty |
| probe-cat-0920 | 09-20 03:45 | category | logreg-probe(C=1.0) on phobert-frozen | title+desc+req | segment | dev | 3812 | macroF1=0.5898 · F1=0.6271 · acc=0.6388 | 34.6s | probe |
| probe-sal-0920 | 09-20 03:45 | salary | ridge-probe(alpha=1.0) on phobert-frozen | title+desc+req | segment | dev | 2698 | MAE=4.42tr · RMSE=8.36tr · R2log=0.466 · ±20%=48.0% | 0.2s | probe |
| dl-cat-ce-cpu-0920 | 09-20 04:22 | category | phobert-frozen+dense(h=256) | title+desc+req | segment | dev | 3812 | macroF1=0.6030 · F1=0.6413 · acc=0.6501 | 6.2s | 1cf72192+dirty |
| dl-cat-focal-cpu-0920 | 09-20 04:23 | category | phobert-frozen+dense(h=256,focal(g=2)) | title+desc+req | segment | dev | 3812 | macroF1=0.5972 · F1=0.6343 · acc=0.6427 | 6.9s | 1cf72192+dirty |
| dl-sal-cpu-0920 | 09-20 04:23 | salary | phobert-frozen+dense(h=256) | title+desc+req | segment | dev | 2698 | MAE=4.13tr · RMSE=8.13tr · R2log=0.515 · R2raw=0.368 · ±20%=52.6% | 6.1s | 1cf72192+dirty |
| dl-cat-focal-repo-0921 | 09-21 04:10 | category | phobert-frozen+dense(h=256,focal(g=2)) | title+desc+req | segment | dev | 3812 | macroF1=0.5972 · F1=0.6343 · acc=0.6427 | 7.8s | 1cf72192+dirty |
| dl-cat-rnn-ce | 09-22 04:02 | category | phobert-frozen+bigru-lstm-cnn(h=100,c=50,both) | title+desc+req | segment+token-level+time-capped | dev | 3812 | macroF1=0.6026 · F1=0.6527 · microF1=0.6624 · acc=0.6624 | 2342.5s | cd2aae64 |
| dl-cat-rnn-ce | 09-22 05:17 | category | phobert-frozen+bigru-lstm-cnn(h=100,c=50,both) | title+desc+req | segment+token-level | dev | 3812 | macroF1=0.6046 · F1=0.6567 · microF1=0.6655 · acc=0.6655 | 4230.5s | cd2aae64+dirty |
| dl-sal-rnn | 09-22 06:35 | salary | phobert-frozen+bigru-lstm-cnn(h=100,c=50,both) | title+desc+req | segment+token-level | dev | 2698 | MAE=3.92tr · RMSE=8.14tr · R2log=0.562 · R2raw=0.366 · ±20%=55.9% | 3788.7s | cd2aae64+dirty |
| dl-cat-rnn-focal | 09-22 06:46 | category | phobert-frozen+bigru-lstm-cnn(h=100,c=50,both,focal(g=2)) | title+desc+req | segment+token-level | dev | 3812 | macroF1=0.6033 · F1=0.6451 · microF1=0.6529 · acc=0.6529 | 4473.8s | cd2aae64+dirty |
