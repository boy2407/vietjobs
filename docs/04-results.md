# Experiment log (append-only)

Selection uses `val`. `test` is scored once, at the end.

| RunID | UTC | Task | Model | Scope | Prep | Eval | n | Headline | Time | Commit |
|---|---|---|---|---|---|---|---|---|---|---|
| cat-P0-majority | 08-26 15:59 | category | majority | title | raw | val | 7159 | macroF1=0.0210 | acc=0.2014 | balAcc=0.0625 | F1@15=0.0225 | top3=0.2608 | 0.8s | uncommit+dirty |
| cat-P0-keyword | 08-26 15:59 | category | keyword | title | raw | val | 7159 | macroF1=0.4321 | acc=0.5678 | balAcc=0.4074 | F1@15=0.4621 | 0.2s | uncommit+dirty |
| cat-P0b-svm-title | 08-26 15:59 | category | svm | title | raw | val | 7159 | macroF1=0.5547 | acc=0.6061 | balAcc=0.6176 | F1@15=0.5871 | 3.8s | uncommit+dirty |
| cat-P1-svm-raw | 08-26 16:01 | category | svm | full | raw | val | 7159 | macroF1=0.5719 | acc=0.6124 | balAcc=0.5969 | F1@15=0.6022 | 77.0s | uncommit+dirty |
| cat-P2-svm-seg | 08-26 16:02 | category | svm | full | segment | val | 7159 | macroF1=0.5733 | acc=0.6114 | balAcc=0.5970 | F1@15=0.6020 | 67.4s | uncommit+dirty |
| cat-P3a-svm-prov | 08-26 16:04 | category | svm | full | province | val | 7159 | macroF1=0.5763 | acc=0.6131 | balAcc=0.6001 | F1@15=0.6065 | 77.2s | uncommit+dirty |
| cat-P3-svm-seg-prov | 08-26 16:06 | category | svm | full | segment+province | val | 7159 | macroF1=0.5743 | acc=0.6127 | balAcc=0.5982 | F1@15=0.6027 | 62.8s | uncommit+dirty |
| cat-P4-svm-charfold | 08-26 16:07 | category | svm | full | segment+charfold+province | val | 7159 | macroF1=0.5754 | acc=0.6146 | balAcc=0.5975 | F1@15=0.6046 | 70.0s | uncommit+dirty |
| cat-P5-svm-stop | 08-26 16:09 | category | svm | full | segment+charfold+province+stopwords | val | 7159 | macroF1=0.5756 | acc=0.6124 | balAcc=0.5983 | F1@15=0.6037 | 79.4s | uncommit+dirty |
| cat-A-logreg | 08-26 16:18 | category | logreg | full | province | val | 7159 | macroF1=0.5941 | acc=0.6304 | balAcc=0.6155 | F1@15=0.6260 | top3=0.9276 | 440.0s | uncommit+dirty |
| cat-A-knn30-title | 08-26 16:18 | category | knn | title | province | val | 7159 | macroF1=0.5266 | acc=0.5969 | balAcc=0.5103 | F1@15=0.5552 | top3=0.8307 | 0.6s | uncommit+dirty |
| cat-A-knn30-full | 08-26 16:19 | category | knn | full | province | val | 7159 | macroF1=0.4059 | acc=0.5086 | balAcc=0.3783 | F1@15=0.4268 | top3=0.8127 | 17.3s | uncommit+dirty |
| cat-A-knn15-title | 08-26 16:28 | category | knn(k=15) | title | province | val | 7159 | macroF1=0.5307 | acc=0.5955 | balAcc=0.5124 | F1@15=0.5601 | top3=0.8151 | 0.7s | uncommit+dirty |
| cat-A-knn50-title | 08-26 16:28 | category | knn(k=50) | title | province | val | 7159 | macroF1=0.5166 | acc=0.5958 | balAcc=0.5013 | F1@15=0.5447 | top3=0.8296 | 0.6s | uncommit+dirty |
| cat-T-svm-C0.1 | 08-26 16:29 | category | svm(C=0.1) | full | province | val | 7159 | macroF1=0.5983 | acc=0.6350 | balAcc=0.6455 | F1@15=0.6298 | 37.5s | uncommit+dirty |
| cat-T-svm-C1 | 08-26 16:31 | category | svm(C=1.0) | full | province | val | 7159 | macroF1=0.5618 | acc=0.6013 | balAcc=0.5787 | F1@15=0.5907 | 110.6s | uncommit+dirty |
| cat-T-svm-C4 | 08-26 16:37 | category | svm(C=4.0) | full | province | val | 7159 | macroF1=0.5171 | acc=0.5593 | balAcc=0.5286 | F1@15=0.5442 | 313.6s | uncommit+dirty |
| cat-T-svm-C0.02 | 08-26 16:38 | category | svm(C=0.02) | full | province | val | 7159 | macroF1=0.6050 | acc=0.6445 | balAcc=0.6713 | F1@15=0.6376 | 27.8s | uncommit+dirty |
| cat-T-svm-C0.05 | 08-26 16:39 | category | svm(C=0.05) | full | province | val | 7159 | macroF1=0.6030 | acc=0.6416 | balAcc=0.6600 | F1@15=0.6374 | 33.5s | uncommit+dirty |
| cat-T-svm-C0.2 | 08-26 17:07 | category | svm(C=0.2) | full | province | val | 7159 | macroF1=0.5873 | acc=0.6263 | balAcc=0.6231 | F1@15=0.6188 | 90.9s | uncommit+dirty |
| cat-T-logreg-C1 | 08-26 21:32 | category | logreg(C=1.0) | full | province | val | 7159 | macroF1=0.6038 | acc=0.6391 | balAcc=0.6419 | F1@15=0.6355 | top3=0.9318 | 684.7s | uncommit+dirty |
| cat-T-svm-C0.005 | 08-27 02:14 | category | svm(C=0.005) | full | province | val | 7159 | macroF1=0.5865 | acc=0.6307 | balAcc=0.6665 | F1@15=0.6180 | 44.5s | uncommit+dirty |
| cat-T-svm-C0.01 | 08-27 02:15 | category | svm(C=0.01) | full | province | val | 7159 | macroF1=0.5976 | acc=0.6381 | balAcc=0.6711 | F1@15=0.6308 | 34.4s | uncommit+dirty |
| cat-R-svm-C0.02-noprep | 08-27 02:16 | category | svm(C=0.02) | full | raw | val | 7159 | macroF1=0.6033 | acc=0.6438 | balAcc=0.6700 | F1@15=0.6366 | 31.8s | uncommit+dirty |
| cat-R-svm-C0.02-seg | 08-27 02:17 | category | svm(C=0.02) | full | segment+province | val | 7159 | macroF1=0.5998 | acc=0.6386 | balAcc=0.6655 | F1@15=0.6308 | 62.6s | uncommit+dirty |
| cat-R-svm-C0.02-all | 08-27 02:19 | category | svm(C=0.02) | full | segment+charfold+province | val | 7159 | macroF1=0.6024 | acc=0.6424 | balAcc=0.6657 | F1@15=0.6365 | 74.4s | uncommit+dirty |
| cat-FINAL-svm-C0.02-test | 08-27 02:28 | category | svm(C=0.02) | full | province | test | 7152 | macroF1=0.6112 | acc=0.6527 | balAcc=0.6816 | F1@15=0.6440 | 79.2s | uncommit+dirty |
| cat-TREE-rf100-struct | 08-31 02:12 | category | rf100 | structured | province | val | 7159 | macroF1=0.2045 | acc=0.2647 | balAcc=0.2288 | F1@15=0.2128 | top3=0.5413 | 4.6s | uncommit+dirty |
| cat-TREE-svm-struct | 08-31 02:13 | category | svm | structured | province | val | 7159 | macroF1=0.1603 | acc=0.2150 | balAcc=0.2281 | F1@15=0.1669 | 4.1s | uncommit+dirty |
| cat-TREE-rf-struct | 08-31 02:13 | category | rf | structured | province | val | 7159 | macroF1=0.2093 | acc=0.2742 | balAcc=0.2328 | F1@15=0.2192 | top3=0.5498 | 10.2s | uncommit+dirty |
| cat-TREE-lgbm-struct | 08-31 02:14 | category | lgbm | structured | province | val | 7159 | macroF1=0.2182 | acc=0.2738 | balAcc=0.2242 | F1@15=0.2314 | top3=0.5628 | 32.2s | uncommit+dirty |
| cat-TREE-xgb-struct | 08-31 02:14 | category | xgb | structured | province | val | 7159 | macroF1=0.2134 | acc=0.2668 | balAcc=0.2288 | F1@15=0.2255 | top3=0.5515 | 18.6s | uncommit+dirty |
| cat-TREE-lgbm100-full | 08-31 07:49 | category | lgbm100 | full | province | val | 7159 | macroF1=0.5555 | acc=0.6163 | balAcc=0.5476 | F1@15=0.5849 | top3=0.9163 | 744.3s | uncommit+dirty |
| cat-TREE-rf100-full | 08-31 07:51 | category | rf100 | full | province | val | 7159 | macroF1=0.5630 | acc=0.6082 | balAcc=0.5844 | F1@15=0.5935 | top3=0.9014 | 69.3s | uncommit+dirty |
| cat-TREE-xgb100-full | 08-31 08:45 | category | xgb100 | full | province | val | 7159 | macroF1=0.5861 | acc=0.6249 | balAcc=0.6105 | F1@15=0.6157 | top3=0.9239 | 3208.8s | uncommit+dirty |
| cat-CAL-knn30 | 08-31 10:10 | category | knn(n_neighbors=30) | full | province | val | 7159 | macroF1=0.4059±0.0081 · acc=0.5086 · balAcc=0.3783 · F1@15=0.4268 · top3=0.8127 | 27.0s | uncommit+dirty |
| cat-CAL-knn30-idle | 08-31 10:46 | category | knn(n_neighbors=30) | full | province | val | 7159 | macroF1=0.4059±0.0081 · acc=0.5086 · balAcc=0.3783 · F1@15=0.4268 · top3=0.8127 | 17.5s | uncommit+dirty |
