# Experiment log (append-only)

Selection uses `val`. `test` is scored once, at the end.

| RunID | UTC | Task | Model | Scope | Prep | Eval | n | Headline | Time | Commit |
|---|---|---|---|---|---|---|---|---|---|---|
| dl-cat-cafe-rnn-ce-s42-t12 | 10-17 10:44 | category | cafebert-frozen+bigru-lstm-cnn(h=100,c=50,both) | title+desc+req | nosegment+token-level | dev | 3812 | macroF1=0.6077 · F1=0.6522 · microF1=0.6603 · acc=0.6603 | 2907.5s | uncommit |
| dl-sal-cafe-rnn-s42-t12 | 10-17 11:13 | salary | cafebert-frozen+bigru-lstm-cnn(h=100,c=50,both) | title+desc+req | nosegment+token-level | dev | 2698 | MAE=3.84tr · RMSE=7.76tr · R2log=0.606 · R2raw=0.424 · ±20%=55.3% | 1658.8s | uncommit |
