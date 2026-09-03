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
| cat-SW-svm-1 | 09-02 03:05 | category | svm(C=0.02) | full | province | val | 7159 | macroF1=0.6050±0.0071 · acc=0.6445 · balAcc=0.6713 · F1@15=0.6376 | 42.2s | 2d30be9b |
| cat-SW-knn-1 | 09-02 03:07 | category | knn(n_neighbors=30) | full | province | val | 7159 | macroF1=0.4059±0.0081 · acc=0.5086 · balAcc=0.3783 · F1@15=0.4268 · top3=0.8127 | 22.5s | 2d30be9b+dirty |
| cat-SW-rf-1 | 09-02 03:08 | category | rf(n_estimators=100) | full | province | val | 7159 | macroF1=0.5630±0.0079 · acc=0.6082 · balAcc=0.5844 · F1@15=0.5935 · top3=0.9014 | 55.3s | 2d30be9b+dirty |
| cat-SW-logreg-1 | 09-02 03:18 | category | logreg(C=1.0) | full | province | val | 7159 | macroF1=0.6038±0.0074 · acc=0.6391 · balAcc=0.6419 · F1@15=0.6355 · top3=0.9318 | 320.9s | 341685f7 |
| cat-SW-lgbm-1 | 09-02 03:29 | category | lgbm(n_estimators=100,learning_rate=0.3) | full | province | val | 7159 | macroF1=0.5555±0.0082 · acc=0.6163 · balAcc=0.5476 · F1@15=0.5849 · top3=0.9163 | 632.5s | 341685f7+dirty |
| cat-SW-xgb-1 | 09-02 04:08 | category | xgb(n_estimators=100,learning_rate=0.3,max_depth=6) | full | province | val | 7159 | macroF1=0.5861±0.0083 · acc=0.6249 · balAcc=0.6105 · F1@15=0.6157 · top3=0.9239 | 2351.6s | 341685f7+dirty |
| cat-SW-svm-2 | 09-02 04:09 | category | svm(C=0.05) | full | province | val | 7159 | macroF1=0.6030±0.0071 · acc=0.6416 · balAcc=0.6600 · F1@15=0.6374 | 38.9s | 341685f7+dirty |
| cat-SW-knn-2 | 09-02 04:11 | category | knn(n_neighbors=15) | full | province | val | 7159 | macroF1=0.4262±0.0085 · acc=0.5121 · balAcc=0.4014 · F1@15=0.4460 · top3=0.8104 | 20.8s | 341685f7+dirty |
| cat-SW-rf-2 | 09-02 04:15 | category | rf(n_estimators=300) | full | province | val | 7159 | macroF1=0.5629±0.0076 · acc=0.6080 · balAcc=0.5861 · F1@15=0.5932 · top3=0.9121 | 201.1s | 341685f7+dirty |
| cat-SW-lgbm-2 | 09-02 04:27 | category | lgbm(n_estimators=100,learning_rate=0.3,num_leaves=31) | full | province | val | 7159 | macroF1=0.5692±0.0083 · acc=0.6224 · balAcc=0.5683 · F1@15=0.5998 · top3=0.9169 | 666.8s | 341685f7+dirty |
| cat-SW-logreg-2 | 09-02 04:32 | category | logreg(C=0.5) | full | province | val | 7159 | macroF1=0.6072±0.0070 · acc=0.6437 · balAcc=0.6554 · F1@15=0.6416 · top3=0.9293 | 286.2s | 341685f7+dirty |
| cat-SW-xgb-2 | 09-02 04:52 | category | xgb(n_estimators=100,learning_rate=0.3,max_depth=4) | full | province | val | 7159 | macroF1=0.5895±0.0076 · acc=0.6263 · balAcc=0.6310 · F1@15=0.6221 · top3=0.9205 | 1200.3s | 341685f7+dirty |
| cat-SW-svm-3 | 09-02 04:53 | category | svm(C=0.01) | full | province | val | 7159 | macroF1=0.5976±0.0070 · acc=0.6381 · balAcc=0.6711 · F1@15=0.6308 | 26.2s | 341685f7+dirty |
| cat-SW-knn-3 | 09-02 04:54 | category | knn(n_neighbors=5) | full | province | val | 7159 | macroF1=0.4361±0.0082 · acc=0.4960 · balAcc=0.4281 · F1@15=0.4582 · top3=0.7739 | 16.7s | 341685f7+dirty |
| cat-SW-xgb-3 | 09-02 05:13 | category | xgb(n_estimators=100,learning_rate=0.3,max_depth=4,colsample_bytree=0.1) | full | province | val | 7159 | macroF1=0.5912±0.0076 · acc=0.6314 · balAcc=0.6323 · F1@15=0.6238 · top3=0.9184 | 1122.7s | 341685f7+dirty |
| cat-SW-lgbm-3 | 09-02 05:19 | category | lgbm(n_estimators=100,learning_rate=0.3,colsample_bytree=0.1) | full | province | val | 7159 | macroF1=0.5579±0.0085 · acc=0.6163 · balAcc=0.5458 · F1@15=0.5851 · top3=0.9158 | 310.5s | 341685f7+dirty |
| cat-SW-logreg-3 | 09-02 05:22 | category | logreg(C=0.25) | full | province | val | 7159 | macroF1=0.6005±0.0070 · acc=0.6391 · balAcc=0.6590 · F1@15=0.6344 · top3=0.9225 | 196.5s | 341685f7+dirty |
| cat-SW-rf-3 | 09-02 05:29 | category | rf(n_estimators=600) | full | province | val | 7159 | macroF1=0.5644±0.0078 · acc=0.6082 · balAcc=0.5880 · F1@15=0.5949 · top3=0.9166 | 316.3s | 341685f7+dirty |
| cat-SW-svm-4 | 09-02 05:30 | category | svm(C=0.1) | full | province | val | 7159 | macroF1=0.5983±0.0074 · acc=0.6350 · balAcc=0.6455 · F1@15=0.6298 | 46.1s | 341685f7+dirty |
| cat-SW-knn-4 | 09-02 05:31 | category | knn(n_neighbors=50) | full | province | val | 7159 | macroF1=0.3736±0.0082 · acc=0.4945 · balAcc=0.3452 · F1@15=0.3943 · top3=0.8058 | 20.2s | 341685f7+dirty |
| cat-SW-logreg-4 | 09-02 06:19 | category | logreg(C=0.1) | full | province | val | 7159 | macroF1=0.5867±0.0067 · acc=0.6258 · balAcc=0.6569 · F1@15=0.6206 · top3=0.9102 | 139.4s | 341685f7+dirty |
| cat-SW-lgbm-4 | 09-02 06:32 | category | lgbm(n_estimators=200,learning_rate=0.15,num_leaves=31) | full | province | val | 7159 | macroF1=0.5830±0.0081 · acc=0.6332 · balAcc=0.5821 · F1@15=0.6142 · top3=0.9274 | 772.2s | 341685f7+dirty |
| cat-SW-xgb-4 | 09-02 07:02 | category | xgb(n_estimators=100,learning_rate=0.3,max_depth=6,colsample_bytree=0.1) | full | province | val | 7159 | macroF1=0.5874±0.0079 · acc=0.6286 · balAcc=0.6068 · F1@15=0.6202 · top3=0.9204 | 1799.1s | 341685f7+dirty |
| cat-SW-knn-5 | 09-02 07:04 | category | knn(n_neighbors=10) | full | province | val | 7159 | macroF1=0.4360±0.0087 · acc=0.5093 · balAcc=0.4167 · F1@15=0.4571 · top3=0.8007 | 16.7s | 341685f7+dirty |
| cat-SW-svm-5 | 09-02 07:04 | category | svm(C=0.005) | full | province | val | 7159 | macroF1=0.5865±0.0068 · acc=0.6307 · balAcc=0.6665 · F1@15=0.6180 | 25.8s | 341685f7+dirty |
| cat-SW-logreg-5 | 09-02 07:06 | category | logreg(C=0.02) | full | province | val | 7159 | macroF1=0.5519±0.0067 · acc=0.5913 · balAcc=0.6370 · F1@15=0.5831 · top3=0.8631 | 96.6s | 341685f7+dirty |
| cat-SW-rf-5 | 09-02 07:09 | category | rf(n_estimators=300,max_features=1000) | full | province | val | 7159 | macroF1=0.5606±0.0080 · acc=0.6072 · balAcc=0.5810 · F1@15=0.5909 · top3=0.9149 | 164.7s | 341685f7+dirty |
| cat-SW-lgbm-5 | 09-02 08:30 | category | lgbm(n_estimators=100,learning_rate=0.3,num_leaves=95) | full | province | val | 7159 | macroF1=0.5618±0.0082 · acc=0.6209 · balAcc=0.5533 · F1@15=0.5917 · top3=0.9154 | 2119.5s | 341685f7+dirty |
| cat-SW-rf-6 | 09-02 08:31 | category | rf(n_estimators=300,max_features=log2) | full | province | val | 7159 | macroF1=0.5502±0.0073 · acc=0.5913 · balAcc=0.6369 · F1@15=0.5800 · top3=0.8793 | 21.6s | 341685f7+dirty |
| cat-SW-knn-6 | 09-02 08:32 | category | knn(n_neighbors=3) | full | province | val | 7159 | macroF1=0.4208±0.0079 · acc=0.4786 · balAcc=0.4221 · F1@15=0.4418 · top3=0.7322 | 17.0s | 341685f7+dirty |
| cat-SW-svm-6 | 09-02 08:33 | category | svm(C=0.2) | full | province | val | 7159 | macroF1=0.5873±0.0076 · acc=0.6263 · balAcc=0.6231 · F1@15=0.6188 | 46.5s | 341685f7+dirty |
| cat-SW-logreg-6 | 09-02 08:36 | category | logreg(C=0.05) | full | province | val | 7159 | macroF1=0.5768±0.0067 · acc=0.6170 · balAcc=0.6535 · F1@15=0.6102 · top3=0.8948 | 126.3s | 341685f7+dirty |
| cat-SW-xgb-6 | 09-02 09:01 | category | xgb(n_estimators=100,learning_rate=0.3,max_depth=5,colsample_bytree=0.2) | full | province | val | 7159 | macroF1=0.5861±0.0077 · acc=0.6276 · balAcc=0.6166 · F1@15=0.6191 · top3=0.9240 | 1487.7s | 341685f7+dirty |
| cat-SW-lgbm-6 | 09-02 09:16 | category | lgbm(n_estimators=100,learning_rate=0.3,colsample_bytree=0.5) | full | province | val | 7159 | macroF1=0.5708±0.0084 · acc=0.6233 · balAcc=0.5644 · F1@15=0.5992 · top3=0.9166 | 896.8s | 341685f7+dirty |
| cat-SW-nb-1 | 09-03 06:53 | category | nb(alpha=1.0) | full | province | val | 7159 | macroF1=0.5371±0.0077 · acc=0.6283 · balAcc=0.5246 · F1@15=0.5720 · top3=0.9174 | 26.1s | e99eb606 |
| cat-SW-sgd-1 | 09-03 07:04 | category | sgd(loss=hinge,alpha=0.0001) | full | province | val | 7159 | macroF1=0.5618±0.0070 · acc=0.5978 · balAcc=0.6318 · F1@15=0.5932 | 72.1s | e99eb606+dirty |
| cat-SW-svm_plain-1 | 09-03 07:05 | category | svm_plain(C=0.02) | full | province | val | 7159 | macroF1=0.5818±0.0082 · acc=0.6579 · balAcc=0.5608 · F1@15=0.6173 | 58.4s | e99eb606+dirty |
| cat-SW-extra-1 | 09-03 07:12 | category | extra(n_estimators=300) | full | province | val | 7159 | macroF1=0.5650±0.0079 · acc=0.6039 · balAcc=0.6071 · F1@15=0.5936 · top3=0.9133 | 334.9s | e99eb606+dirty |
| cat-SW-nb-2 | 09-03 07:13 | category | nb(alpha=0.1) | full | province | val | 7159 | macroF1=0.5535±0.0080 · acc=0.6340 · balAcc=0.5437 · F1@15=0.5869 · top3=0.9155 | 24.3s | e99eb606+dirty |
