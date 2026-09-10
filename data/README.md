# Data

`data/` is in `.gitignore`. Nothing inside it is committed.

## Getting the raw data

The dataset is `dinhieufam/VietJobs` on the Hugging Face Hub — about 103 MB,
**48,092 rows**, 18 columns. Download it to `data/raw/VietJobs.csv`.

```bash
pip install -r requirements.txt
python -c "
from huggingface_hub import hf_hub_download
import shutil, pathlib
p = hf_hub_download('dinhieufam/VietJobs', 'VietJobs.csv',
                    repo_type='dataset', cache_dir='data/raw/.cache/huggingface')
pathlib.Path('data/raw').mkdir(parents=True, exist_ok=True)
shutil.copy(p, 'data/raw/VietJobs.csv')
print('ok')"
```

> `scripts/download_data.sh` was mentioned in an older README but **does not
> exist**. The snippet above is the download that is actually used; the directory
> `data/raw/.cache/huggingface/` is the trace it leaves.

Check you have the right file against the hash fixed in the manifest:

```bash
shasum -a 256 data/raw/VietJobs.csv
# 85862b06fda4e814fe0c1d8622f173d189c92758345f77df16d1232d0c49d477
```

## Building the splits

```bash
python -m vietjobs.dataset build                # ~26 minutes, mostly segmentation
python -m vietjobs.dataset build --no-segment   # fast, for debugging
```

## Layout

| Path | Produced by | Contents |
|---|---|---|
| `data/raw/VietJobs.csv` | downloaded from the Hub | The original, never touched |
| `data/processed/splits/` | `python -m vietjobs.dataset build` | `train/dev/test.parquet` — frozen, no group straddling two splits |
| `data/processed/manifest.json` | `python -m vietjobs.dataset build` | Row counts, the sha256, the split seed, per-split statistics |
| `data/interim/` | — | A place for temporary files, currently empty |

Sizes after the build: train 34,354 rows · dev 3,812 · test 9,541.

## The splits are a contract

The three parquet files are the contract for **every** experiment. **Do not rebuild
them with a different seed** once the first model has been trained — doing so makes
every row in `docs/04-results.md` incomparable with every other.

Details: [`docs/03-protocol.md`](../docs/03-protocol.md).
Why the cleaning works this way: [`docs/01-data-audit.md`](../docs/01-data-audit.md).
