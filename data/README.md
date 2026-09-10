# Data

Everything the experiments read lives here: the raw dump, the frozen splits, and
the manifest that ties them together.

**Nothing inside `data/` is committed** — only this README is. The subdirectories
are in `.gitignore` by design: `processed/splits/train.csv` alone is **229 MB**,
above GitHub's 100 MB per-file limit, so shipping them would mean Git LFS and a
717 MB repository.

That costs nothing, because you do not need the files — you rebuild them, and the
rebuild is **exact**, not approximate. Same postings in `train`, same in `dev`,
same in `test`, on any machine. The next section is how, and how to prove you got
it right.

## Reproducing the data on another machine

The splits are deterministic. Two people running these four steps on different
machines end up with byte-identical split membership: the same postings in
`train`, the same in `dev`, the same in `test`. Nothing is shared over the
network except the raw dump.

### 1. Get the raw dump and verify it

The dataset is `dinhieufam/VietJobs` on the Hugging Face Hub — about 103 MB,
**48,092 rows**, 18 columns.

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

**Verify before anything else.** A different raw file means a different split, and
nothing downstream will warn you:

```bash
shasum -a 256 data/raw/VietJobs.csv
# 85862b06fda4e814fe0c1d8622f173d189c92758345f77df16d1232d0c49d477
```

> `scripts/download_data.sh` was mentioned in an older README but **does not
> exist**. The snippet above is the download that is actually used; the directory
> `data/raw/.cache/huggingface/` is the trace it leaves.

### 2. Make sure the segmenter is installed

`underthesea` is the default word-segmentation backend, and it is **optional at
import time** — if it is missing, `dataset.build` still finishes, silently
leaving the `*_seg` columns unsegmented. That does **not** change the split, but
it does change what every model reads, so your numbers will not match
`docs/04-results.md`.

```bash
python -c "from underthesea import word_tokenize; print('segmenter ok')"
```

`$VIETJOBS_SEGMENTER` can switch the backend to `pyvi`. Leave it unset — the
published runs all used `underthesea`.

### 3. Build

```bash
python -m vietjobs.dataset build                # ~26 minutes, mostly segmentation
python -m vietjobs.dataset build --no-segment   # fast, for debugging only
```

### 4. Verify you got the same splits

Row counts are a weak check — they can match while the membership differs. This
compares the actual set of postings in each split:

```bash
python -c "
import hashlib
from vietjobs import dataset as D
for s in ['train', 'dev', 'test']:
    d = D.load_split(s)
    gids = ''.join(sorted(d.group_id.unique()))
    print('%-6s rows=%6d groups=%6d %s' % (s, len(d), d.group_id.nunique(),
          hashlib.sha256(gids.encode()).hexdigest()[:16]))"
```

Expected, exactly:

| Split | Rows | Groups | Fingerprint |
|---|---|---|---|
| `train` | 34,354 | 22,026 | `1ec048522c9e10f1` |
| `dev` | 3,812 | 3,728 | `fbff00bcbcd06ee4` |
| `test` | 9,541 | 9,145 | `8176be328df584b9` |

Then check `processed/manifest.json` agrees on the four fields that matter:

```json
"source_sha256": "85862b06fda4e814fe0c1d8622f173d189c92758345f77df16d1232d0c49d477",
"split_seed": 20260826,
"groups_straddling_splits": 0,
"segmenter_available": true
```

If the fingerprints match, you have the same data as every row in
`docs/04-results.md`. If they do not, stop — do not train, and do not log a
result. Re-check the sha256 first; that is the usual cause.

### Why it is reproducible

| Mechanism | Where | Effect |
|---|---|---|
| Repost groups keyed by `hashlib.sha1` of the accent-folded text | `vitext.group_key` | A content hash, not Python's randomized `hash()` — stable across machines, runs and Python versions |
| `np.random.default_rng(SPLIT_SEED)` | `dataset.group_stratified_split` | NumPy guarantees the PCG64 stream, so the same seed draws the same permutation everywhere |
| `group_id` computed **before** segmentation | `dataset.clean` | `--no-segment` produces the *same split* — segmentation only fills the `*_seg` mirrors |
| Group assignment driven by sorted group keys, not row order | `dataset.group_stratified_split` | Verified: re-deriving the split from the three stored files, fed in a different row order, reproduced all 47,707 assignments with 0 mismatches |

What **does** break reproducibility: a different raw file, a different
`SPLIT_SEED`, or different split fractions. All three are frozen in
[`config.py`](../src/vietjobs/config.py) and must stay frozen — see
[Rule 2](../AGENTS.md).

## How the split is drawn

Two stages, in this order (scheme v2, since 2026-09-09):

1. `train_pool : test` = 8 : 2
2. that pool again into `train : dev` = 9 : 1

which lands at **72 / 8 / 20** of the rows — 34,354 / 3,812 / 9,541 over 16
occupation classes. Stage 2 gets `seed + 1` so the two draws do not share a
permutation.

Two properties hold at both stages:

- **Group-disjoint.** Reposted ads are common in this dump, so rows are grouped
  by a hash of the folded posting text and the *group* — not the row — is
  assigned. `groups_straddling_splits` in the manifest is **0**; a row-level
  split would put the same posting in both `train` and `test` and inflate every
  score.
- **Stratified by `category`**, so the smallest of the 16 classes survives into
  every bucket.

The stages are kept separate rather than collapsed into one three-way draw
because stage 2 must be able to run again — a different `dev` slice, a k-fold
over the pool — **without a single row of `test` moving**.

`dev` is small on purpose (3,812 rows; the smallest class has 25 there). Read
`f1_macro_no_junk` next to any per-class number, and confirm on `test` at the end.

## Layout

| Path | Size | Produced by | Contents |
|---|---|---|---|
| `raw/VietJobs.csv` | 98 MB | downloaded from the Hub | The original, never touched. 48,092 rows, 18 columns |
| `processed/splits/` | 322 MB | `python -m vietjobs.dataset build` | `train.csv` · `dev.csv` · `test.csv` — **the splits every experiment reads** |
| `processed/manifest.json` | 2.6 KB | `python -m vietjobs.dataset build` | Row counts, the source sha256, the split seed, per-split statistics, and the `column_dtypes` map the CSV cannot carry |
| `processed/splits-parquet/` | 124 MB | a mirror of the above | The same v2 splits in parquet. Convenience copy — nothing in `src/` reads it |
| `processed/splits-scheme-v1/` | 124 MB | the retired split scheme | 33,396 / 7,159 / 7,152 rows, and `val` not `dev`. Kept only so the archived machine-learning numbers stay reproducible — **do not train on it** |
| `processed/manifest-scheme-v1.json` | 941 B | the retired split scheme | The manifest belonging to the directory above |
| `external/vietjobs37k/` | 50 MB | downloaded from Zenodo | A third-party occupation dataset, kept for cross-evaluation only — see [below](#the-external-dataset) |
| `interim/` | 0 B | — | A place for temporary files, currently empty |

> **Source:** `data/processed/manifest.json`; sizes from `du -sh`, 2026-09-10.

A split file is much larger than its share of the raw dump — 229 MB of `train`
against 98 MB of raw — because cleaning widens 18 columns into 50. **70 % of
`train.csv` is mirror columns**: each long text field is stored up to four times
(raw, salary-masked, segmented, masked-and-segmented). `description` alone
accounts for 46 % of the file. Both mirrors earn their place — `*_masked` is what
keeps the salary tasks honest (Rule 3), `*_seg` caches the 780.3 seconds of
segmentation.

## The external dataset

`external/vietjobs37k/` is **VietJobs-37K**, a separate, published dataset. It is
**not** an input to the pipeline — nothing in `src/` reads it — and it is never
trained on. It is there to cross-evaluate the `category` model on data this
project did not build.

| | |
|---|---|
| Source | https://zenodo.org/records/20472202 |
| Data DOI | `10.5281/zenodo.20472202` |
| Paper DOI | `10.1109/ACCESS.2026.3718998` (IEEE Access, vol. 14, pp. 119006–119026, 2026) |
| Original code | https://github.com/kaldlabs/VieJobBERT |
| Package | `vietjobs-37k-v1.0.0_r1_reviewer_package.zip`, 7.4 MB, LZMA-compressed |
| Package sha256 | `8c38f017dad78630738dc40298c3c44bf6ed942167416075897ec2b4e63956b8` |
| Licence | CC BY-NC 4.0, **author-created parts only**. The posting text belongs to the source platforms — read `LICENSE_SCOPE.md` in the package before redistributing anything |
| Contents | 37,274 postings, 60-label multi-label taxonomy, plus a 1,000-record human-reviewed gold anchor |

Download it — macOS `unzip` cannot read LZMA, so extract with Python:

```bash
mkdir -p data/external/vietjobs37k && cd data/external/vietjobs37k
curl -L -o vietjobs-37k.zip \
  "https://zenodo.org/api/records/20472202/files/vietjobs-37k-v1.0.0_r1_reviewer_package.zip/content"
python3 -c "import zipfile; zipfile.ZipFile('vietjobs-37k.zip').extractall('.')"
```

The package ships its own `SHA256SUMS`; verify with
`shasum -a 256 -c SHA256SUMS` from inside the extracted directory.

**The local copy is deliberately pruned**, 57 MB → 50 MB: 16 of the 59 declared
files are kept. Dropped are `validation/` (7.0 MB of the authors' own annotation
sheets and per-record predictions), `diagnostics/`, `scripts/`, `crosswalk/` (this
project uses its own 16 classes, not ISCO-08), and the release-note file. Every
one of them is re-downloadable from the link above, so `shasum -c` reports them
as missing and that is expected — what matters is that every *kept* file passes.
`PROVENANCE.md`, written inside the extracted directory, records the full
keep/drop reasoning.

> ⚠️ **Do not report a generalization number without de-duplicating first.**
> **13.6 %** of VietJobs-37K — **5,042 of 37,143** postings — near-duplicates a row
> in `raw/VietJobs.csv`. Both datasets scrape the same boards (topcv, careerviet,
> job3s). Treating it as a clean external set inflates the result.

## Reading a split

The splits are **plain CSV** so they open in anything, but CSV stores no types.
`manifest.json` carries the `column_dtypes` map, and `dataset.load_split` is the
**only supported way** to read a split back — it restores the integer and float
columns and keeps `NaN` on the postings that published no salary:

```python
from vietjobs import dataset as D
df = D.load_split("dev")      # (3812, 50), 16 categories
```

A bare `pd.read_csv` gives you every column as a string and silently turns the
missing salaries into empty text. Do not do it.

The two salary tasks may only read the `*_masked` columns —
`features.resolve_column` is the single place that decides this, and a mistake
here raises no error, it just produces a model that reads its own answer key.
See Rule 3 in [`AGENTS.md`](../AGENTS.md).

## The splits are a contract

The three CSV files are the contract for **every** experiment. **Do not rebuild
them with a different seed** once a model has been trained — doing so makes every
row in `docs/04-results.md` incomparable with every other. `test.csv` is stricter
still: it is read exactly once, at the end, behind `--confirm-test`, and never
filtered, de-duplicated, re-split or overwritten (Rule 5).

Details: [`docs/03-protocol.md`](../docs/03-protocol.md).
Why the cleaning works this way: [`docs/01-data-audit.md`](../docs/01-data-audit.md).
