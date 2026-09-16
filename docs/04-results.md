# Experiment log (append-only)

Selection uses `val`. `test` is scored once, at the end.

> **This log restarts on 2026-09-08**, when the main track of the thesis moved to
> deep learning. The 101 machine-learning experiment rows (2026-08-26 → 09-07) sit
> intact in [archive/04-results-ml.md](archive/04-results-ml.md) — the bar to beat
> is `cat-FINAL-svm-C0.02-test`, **test macro-F1 0.6112**.

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
