"""Phân tích bộ dữ liệu gốc — nguồn số cho docs/05-phan-tich-du-lieu.md.

Đọc thẳng ``data/raw/VietJobs.csv`` (48.092 dòng, khử trùng lặp còn 47.707), tức
là mô tả **toàn bộ corpus như đã thu thập**, không phải một tập con. Đó là điều
một chương mô tả dữ liệu phải làm: người đọc cần biết bộ dữ liệu là gì trước khi
biết nó được chia thế nào.

Ranh giới với quy tắc 5 (``test`` bất khả xâm phạm) được giữ như sau:

    con số MÔ TẢ      -> đo trên tệp gốc (lệch lớp, phân bố lương, độ trải…)
    con số QUYẾT ĐỊNH -> đo trên train/dev (mốc trần, độ dài cắt token)

Không bước nào ở đây ghi vào ``splits/``, và không con số nào ở đây được dùng để
chọn mô hình hay chọn siêu tham số.

    python scripts/analyze_data.py              # đo + vẽ
    python scripts/analyze_data.py --no-plot    # chỉ đo
    python scripts/analyze_data.py --no-floors  # bỏ qua mốc trần (không đọc splits)
    PYTHONPATH=src .venv-dl/bin/python scripts/analyze_data.py --tokens

``--tokens`` đo độ dài chuỗi bằng đúng tokenizer PhoBERT trên đúng chuỗi mà mô
hình đọc, nên nó cần môi trường ``.venv-dl`` và chỉ đọc train+dev. Kết quả được
đệm lại; lần chạy chính sau đó vẽ hình 07 từ đệm ấy.

Ghi ra:
    artifacts/eda/summary.json          mọi con số, để tài liệu trích dẫn
    artifacts/eda/token_lengths.json    đệm của --tokens
    docs/figures/eda/*.png              bảy hình
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vietjobs import config as C  # noqa: E402
from vietjobs import dataset as D  # noqa: E402

OUT_JSON = C.ARTIFACT_DIR / "eda" / "summary.json"
TOKEN_JSON = C.ARTIFACT_DIR / "eda" / "token_lengths.json"
FIG_DIR = C.DOCS_DIR / "figures" / "eda"

# Ba trường văn bản tự do trong tệp gốc, theo đúng thứ tự dl/text.py ghép chúng.
TEXT_COLUMNS = ("job_title", "description", "requirements_text")
# Các độ dài cắt đáng cân nhắc cho PhoBERT. 256 là mặc định hiện tại của
# dl/encode.py; hai mốc kia có mặt để thấy cái giá của việc đổi.
MAX_LEN_GRID = (128, 256, 512)

INK, ACCENT, MUTED = "#0E6B5B", "#9E5C22", "#54625E"


# ---------------------------------------------------------------------------
# Đọc dữ liệu
# ---------------------------------------------------------------------------


def load_corpus():
    """Tệp gốc + nhãn suy ra, bằng đúng công thức mà ``dataset.clean`` dùng."""
    raw = D.load_raw()
    n0 = len(raw)
    raw = raw.drop_duplicates().reset_index(drop=True)

    df = pd.DataFrame(index=raw.index)
    df["category"] = raw["category"].str.strip()
    for col, values in D.derive_salary(raw).items():
        df[col] = values

    # Độ dài văn bản thô — ký tự và từ, trên chuỗi ghép đúng thứ tự của dl/text.py.
    # str.cat chứ không phải agg(axis=1): ghép theo dòng là vòng lặp Python và mất
    # hàng phút trên 47 nghìn dòng.
    parts = [raw[c].fillna("") for c in TEXT_COLUMNS]
    joined = parts[0].str.cat(parts[1:], sep=" . ")
    df["text_chars"] = joined.str.len()
    df["text_words"] = joined.str.split().map(len)
    for col in TEXT_COLUMNS:
        df[f"{col}_chars"] = raw[col].fillna("").str.len()

    provenance = {
        "source_file": str(C.RAW_CSV),
        "source_sha256": D._file_sha256(C.RAW_CSV),
        "rows_raw": int(n0),
        "rows_after_exact_dedup": int(len(raw)),
        "dropped_exact_duplicates": int(n0 - len(raw)),
        "n_columns_raw": int(raw.shape[1]),
    }
    return df, provenance, raw


# ---------------------------------------------------------------------------
# Đo
# ---------------------------------------------------------------------------


def q(s: pd.Series, p: float) -> float:
    return float(np.quantile(s, p))


def completeness(raw: pd.DataFrame) -> pd.DataFrame:
    """Mỗi cột gốc điền được bao nhiêu phần trăm dòng.

    Một trường trống 74 % không phải là đặc trưng, nó là một cái bẫy: mô hình học
    được "tin này thiếu trường ấy" chứ không học được nội dung của nó.
    """
    filled = raw.notna() & (raw.astype(str).apply(lambda s: s.str.strip()) != "")
    rows = [{"column": c, "filled": int(filled[c].sum()),
             "fill_rate": float(filled[c].mean())} for c in raw.columns]
    return pd.DataFrame(rows).sort_values("fill_rate").reset_index(drop=True)


def describe_salary(s: pd.Series) -> dict:
    """Phân bố một cột lương: vị trí, độ lệch, đuôi."""
    logged = np.log1p(s)
    iqr = q(s, 0.75) - q(s, 0.25)
    fence_hi = q(s, 0.75) + 1.5 * iqr
    fence_lo = q(s, 0.25) - 1.5 * iqr
    return {
        "n": int(s.size),
        "mean": float(s.mean()), "median": float(s.median()), "std": float(s.std()),
        "skew": float(s.skew()), "kurtosis": float(s.kurtosis()),
        "skew_log1p": float(logged.skew()), "std_log1p": float(logged.std()),
        "min": float(s.min()), "max": float(s.max()),
        "p01": q(s, 0.01), "p05": q(s, 0.05), "p25": q(s, 0.25), "p50": q(s, 0.50),
        "p75": q(s, 0.75), "p90": q(s, 0.90), "p95": q(s, 0.95), "p99": q(s, 0.99),
        "iqr": float(iqr), "fence_low": float(fence_lo), "fence_high": float(fence_hi),
        "n_above_fence": int((s > fence_hi).sum()),
        "n_below_fence": int((s < fence_lo).sum()),
        "n_implausible_low": int((s < C.SALARY_IMPLAUSIBLE_LOW).sum()),
        "n_implausible_high": int((s > C.SALARY_IMPLAUSIBLE_HIGH).sum()),
        "n_extreme_flagged": int((s >= C.SALARY_EXTREME_THRESHOLD).sum()),
    }


def per_category(df: pd.DataFrame) -> pd.DataFrame:
    """Một dòng mỗi ngành: cỡ lớp, tỷ lệ công bố lương, vị trí và độ trải lương."""
    rows = []
    for cat, g in df.groupby("category", observed=True):
        paid = g.loc[g["salary_disclosed"] == 1, "salary_mid"].dropna()
        rows.append({
            "category": cat,
            "n": int(len(g)),
            "share": len(g) / len(df),
            "n_disclosed": int(len(paid)),
            "disclosure_rate": float(len(paid) / len(g)) if len(g) else float("nan"),
            "median": float(paid.median()) if len(paid) else float("nan"),
            "p25": q(paid, 0.25) if len(paid) else float("nan"),
            "p75": q(paid, 0.75) if len(paid) else float("nan"),
            "p95": q(paid, 0.95) if len(paid) else float("nan"),
            "max": float(paid.max()) if len(paid) else float("nan"),
            "iqr": (q(paid, 0.75) - q(paid, 0.25)) if len(paid) else float("nan"),
            "median_text_words": float(g["text_words"].median()),
        })
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


def gini(counts: np.ndarray) -> float:
    """Bất bình đẳng của phân bố cỡ lớp. 0 = chia đều, 1 = một lớp nuốt hết."""
    x = np.sort(counts.astype(float))
    n = x.size
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def distribution_shape(cats: pd.DataFrame, n_rows: int) -> dict:
    """Hình dạng của phân bố cỡ lớp, và cỡ lớp có kéo theo cái gì khác không.

    Độ dốc Zipf là hồi quy ``log(n)`` theo ``log(hạng)``. Dốc gần −1 nghĩa là đuôi
    dài kiểu luật luỹ thừa — một dải liên tục từ lớp lớn xuống lớp nhỏ, chứ không
    phải vài lớp hiếm lẻ tẻ có thể gộp đi cho xong.
    """
    from scipy import stats

    n = cats["n"].to_numpy()                       # cats đã xếp giảm dần
    rank = np.arange(1, n.size + 1)
    slope, _ = np.polyfit(np.log(rank), np.log(n), 1)
    r_disc, p_disc = stats.pearsonr(np.log(n), cats["disclosure_rate"].to_numpy())
    rho_med, p_med = stats.spearmanr(n, cats["median"].to_numpy())
    return {
        "mean_class_size": float(n.mean()),
        "median_class_size": float(np.median(n)),
        "top4_share": float(n[:4].sum() / n_rows),
        "bottom4_share": float(n[-4:].sum() / n_rows),
        "n_classes_below_uniform": int((n < n.mean()).sum()),
        "zipf_slope": float(slope),
        "gini": gini(n),
        "size_vs_disclosure_pearson_logn": float(r_disc),
        "size_vs_disclosure_p": float(p_disc),
        "size_vs_median_spearman": float(rho_med),
        "size_vs_median_p": float(p_med),
    }


def variance_split(df: pd.DataFrame) -> dict:
    """Ngành nghề giải thích được bao nhiêu phần phương sai của lương?

    eta^2 = phương sai giữa các ngành / tổng phương sai, tính trên log1p(salary).
    Gần 0 nghĩa là biết ngành gần như không giúp đoán lương — điều đó quyết định
    kỳ vọng đặt vào nhánh hồi quy.
    """
    paid = df[(df["salary_disclosed"] == 1) & df["salary_mid"].notna()]
    y = np.log1p(paid["salary_mid"].to_numpy())
    grand = y.mean()
    ss_total = float(((y - grand) ** 2).sum())
    ss_between = 0.0
    for _, g in paid.groupby("category", observed=True):
        gy = np.log1p(g["salary_mid"].to_numpy())
        ss_between += len(gy) * (gy.mean() - grand) ** 2
    med = paid.groupby("category", observed=True)["salary_mid"].median()
    return {
        "eta_squared_log": ss_between / ss_total,
        "median_min": float(med.min()), "median_max": float(med.max()),
        "median_min_category": str(med.idxmin()), "median_max_category": str(med.idxmax()),
        "median_spread": float(med.max() - med.min()),
    }


def text_shape(df: pd.DataFrame) -> dict:
    """Độ dài đầu vào, đo thô bằng ký tự và từ trên tệp gốc.

    Con số quyết định (bao nhiêu tin bị cắt ở 256 token) nằm ở ``--tokens``; ở đây
    chỉ là hình dạng chung để biết văn bản dài cỡ nào trước khi chạm tokenizer.
    """
    w, ch = df["text_words"], df["text_chars"]
    return {
        "words": {"mean": float(w.mean()), "median": float(w.median()),
                  "p05": q(w, 0.05), "p25": q(w, 0.25), "p75": q(w, 0.75),
                  "p95": q(w, 0.95), "p99": q(w, 0.99), "max": int(w.max())},
        "chars": {"mean": float(ch.mean()), "median": float(ch.median()),
                  "p95": q(ch, 0.95), "max": int(ch.max())},
        "per_field_median_chars": {
            c: float(df[f"{c}_chars"].median()) for c in TEXT_COLUMNS
        },
        "n_empty_description": int((df["description_chars"] == 0).sum()),
        "n_empty_requirements": int((df["requirements_text_chars"] == 0).sum()),
    }


def floors():
    """Mốc không cần mô hình — con số QUYẾT ĐỊNH, nên chấm trên dev, học từ train.

    Trả về ``None`` nếu chưa có splits: mô tả corpus không được phụ thuộc vào việc
    đã chạy ``dataset build`` hay chưa.
    """
    try:
        train = D.load_split("train")
        dev = D.load_split("dev")
    except (FileNotFoundError, OSError, KeyError):
        return None

    tr = train[(train["salary_disclosed"] == 1) & train["salary_mid"].notna()]
    ev = dev[(dev["salary_disclosed"] == 1) & dev["salary_mid"].notna()]
    y = ev["salary_mid"].to_numpy()

    global_median = float(tr["salary_mid"].median())
    per_cat = tr.groupby("category", observed=True)["salary_mid"].median()
    oracle = ev["category"].map(per_cat).fillna(global_median).to_numpy()

    def score(pred) -> dict:
        pred = np.asarray(pred, dtype=float)
        return {
            "mae": float(np.abs(pred - y).mean()),
            "rmse": float(np.sqrt(((pred - y) ** 2).mean())),
            "mae_log": float(np.abs(np.log1p(pred) - np.log1p(y)).mean()),
            "r2": float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
        }

    majority = train["category"].value_counts().index[0]
    acc = float((dev["category"] == majority).mean())
    return {
        "scored_on": "dev", "fitted_on": "train",
        "n_dev": int(len(dev)), "n_dev_disclosed": int(len(ev)),
        "salary_global_median": {"value": global_median,
                                 **score(np.full(len(y), global_median))},
        "salary_category_median_oracle": {"note": "dùng nhãn ngành thật của dev",
                                          **score(oracle)},
        "category_majority": {
            "label": majority, "accuracy": acc,
            "macro_f1": float(2 * acc / (1 + acc) / dev["category"].nunique()),
        },
        "disclosed_majority": {
            "label": int(train["salary_disclosed"].mode().iloc[0]),
            "accuracy": float((dev["salary_disclosed"]
                               == train["salary_disclosed"].mode().iloc[0]).mean()),
        },
    }


# ---------------------------------------------------------------------------
# Độ dài token — chạy riêng trong .venv-dl
# ---------------------------------------------------------------------------


def measure_tokens(task: str = "category", splits=("train", "dev")) -> dict:
    """Đếm token PhoBERT trên đúng chuỗi mà mô hình đọc, để biết cắt ở 256 mất gì.

    Chỉ train+dev: đây là con số dùng để chọn ``--max-len``, nên quy tắc 5 cấm nó
    nhìn ``test``.
    """
    from transformers import AutoTokenizer

    from vietjobs.dl import text as T

    tok = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
    lengths = []
    for split in splits:
        texts = T.build_text(D.load_split(split), task=task)
        for i in range(0, len(texts), 512):
            enc = tok(texts[i:i + 512], add_special_tokens=True)["input_ids"]
            lengths.extend(len(x) for x in enc)
        print(f"  [{split}] {len(texts):,} tin", flush=True)
    arr = np.asarray(lengths)
    edges = list(range(0, 1050, 10))
    return {
        "task": task, "splits": list(splits), "tokenizer": "vinai/phobert-base-v2",
        "n": int(arr.size),
        "mean": float(arr.mean()), "median": float(np.median(arr)),
        "p75": float(np.quantile(arr, 0.75)), "p95": float(np.quantile(arr, 0.95)),
        "p99": float(np.quantile(arr, 0.99)), "max": int(arr.max()),
        "truncated_share": {str(m): float((arr > m).mean()) for m in MAX_LEN_GRID},
        "tokens_kept_share": {
            str(m): float(np.minimum(arr, m).sum() / arr.sum()) for m in MAX_LEN_GRID
        },
        # Histogram đệm lại để lần chạy chính vẽ được mà không cần torch.
        "hist_edges": edges,
        "hist_counts": [int(v) for v in np.histogram(np.clip(arr, 0, edges[-1] - 1),
                                                     bins=edges)[0]],
    }


# ---------------------------------------------------------------------------
# Vẽ
# ---------------------------------------------------------------------------


def short_label(cat: str, words: int = 2) -> str:
    """Nhãn ngắn để chú tại điểm — tên ngành đầy đủ dài tới 48 ký tự, chồng nhau."""
    parts = cat.split("_")
    return " ".join(parts[:words]) + ("…" if len(parts) > words else "")


def place_labels(ax, xs, ys, labels, renderer) -> None:
    """Chú nhãn cạnh điểm, thử bốn hướng và bỏ hướng nào đè lên nhãn đã đặt."""
    candidates = [(0, 9, "center", "bottom"), (0, -9, "center", "top"),
                  (8, 0, "left", "center"), (-8, 0, "right", "center")]
    placed = []
    for x, y, lab in zip(xs, ys, labels):
        for dx, dy, ha, va in candidates:
            t = ax.annotate(lab, (x, y), textcoords="offset points", xytext=(dx, dy),
                            ha=ha, va=va, fontsize=7, color=MUTED)
            bb = t.get_window_extent(renderer).expanded(1.02, 1.25)
            if not any(bb.overlaps(o) for o in placed):
                placed.append(bb)
                break
            t.remove()
        else:                      # bốn hướng đều kẹt: vẫn phải hiện, chấp nhận đè
            t = ax.annotate(lab, (x, y), textcoords="offset points", xytext=(0, 9),
                            ha="center", va="bottom", fontsize=7, color=MUTED)
            placed.append(t.get_window_extent(renderer))


def _setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.edgecolor": MUTED, "axes.labelcolor": "#141F1D",
        "text.color": "#141F1D", "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": "#DCE4E1", "grid.linewidth": 0.6,
        "axes.titlesize": 10,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    return matplotlib, plt


def fig_completeness(plt, comp: pd.DataFrame, n_rows: int, out: list) -> None:
    """01 — cột nào có dữ liệu, cột nào rỗng. Đọc trước khi chọn đặc trưng."""
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    rate = comp["fill_rate"] * 100
    colors = [ACCENT if r < 90 else INK for r in rate]
    ax.barh(comp["column"], rate, color=colors, alpha=0.88)
    for y, (r, f) in enumerate(zip(rate, comp["filled"])):
        ax.text(r + 1.2, y, f"{r:.1f} %  ({f:,})".replace(",", "."), va="center", fontsize=7.5)
    ax.set_xlim(0, 122)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("% trong {:,} dòng có giá trị".format(n_rows).replace(",", "."))
    ax.set_title("Hình 01 · Độ đầy của 18 trường trong tệp gốc\n"
                 "cam = dưới 90 %, không dùng được như một đặc trưng độc lập", loc="left")
    fig.tight_layout()
    p = FIG_DIR / "eda-01-do-day-truong.png"
    fig.savefig(p, dpi=200); plt.close(fig); out.append(str(p))


def fig_imbalance(plt, cats: pd.DataFrame, shape: dict, n_rows: int, out: list) -> None:
    """02a/02b — lệch lớp: cột ngang cho "ngành nào", Lorenz cho "lệch đến mức nào".

    Tách ra làm 2 hình riêng để dễ đưa vào báo cáo.
    """
    c = cats.sort_values("n")

    # 02a — Số lượng tin mỗi lớp (barh)
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    bars = ax.barh(c["category"], c["n"], color=INK, alpha=0.88)
    bars[list(c["category"]).index(C.JUNK_CATEGORY)].set_color(ACCENT)
    for y, (n, sh) in enumerate(zip(c["n"], c["share"])):
        ax.text(n + c["n"].max() * 0.012, y,
                f"{n:,}".replace(",", ".") + f"  ({sh*100:.1f} %)", va="center", fontsize=7.5)
    uniform = n_rows / len(c)
    ax.axvline(uniform, color=ACCENT, ls="--", lw=1.4,
               label="chia đều 16 lớp = {:,.0f}".format(uniform).replace(",", "."))
    ax.set_xlim(0, c["n"].max() * 1.24)
    ax.set_xlabel("Số tin tuyển dụng")
    ax.set_title("Lệch lớp {:.1f} : 1 trên {:,} tin".format(
                     cats["n"].max() / cats["n"].min(), n_rows).replace(",", ".") +
                 f"\ncam = lớp gom '{C.JUNK_CATEGORY}'", loc="left")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    p = FIG_DIR / "eda-02a-lech-lop.png"
    fig.savefig(p, dpi=200); plt.close(fig); out.append(str(p))

    # 02b — Đường Lorenz (sự không cân bằng)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    n = np.sort(cats["n"].to_numpy())[::-1]
    cum = np.concatenate([[0], np.cumsum(n) / n.sum()]) * 100
    xs = np.arange(len(n) + 1) / len(n) * 100
    ax.plot(xs, cum, color=INK, lw=2, marker="o", ms=3.5)
    ax.plot([0, 100], [0, 100], color=MUTED, ls=":", lw=1.2, label="nếu chia đều")
    ax.fill_between(xs, xs, cum, color=INK, alpha=0.10)
    ax.axvline(25, color=ACCENT, ls="--", lw=1.3)
    ax.annotate(f"4 lớp lớn nhất giữ {shape['top4_share']*100:.1f} %",
                (25, shape["top4_share"] * 100), textcoords="offset points",
                xytext=(9, -16), fontsize=8, color=ACCENT)
    ax.set_xlabel("% số lớp, xếp từ lớn xuống nhỏ")
    ax.set_ylabel("% số tin cộng dồn")
    ax.set_title(f"Đường Lorenz — Gini {shape['gini']:.3f}, "
                 f"dốc Zipf {shape['zipf_slope']:.2f}", loc="left")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    p = FIG_DIR / "eda-02b-lorenz.png"
    fig.savefig(p, dpi=200); plt.close(fig); out.append(str(p))


def fig_class_scatter(plt, mpl, cats: pd.DataFrame, shape: dict, df: pd.DataFrame,
                      out: list) -> None:
    """03a/03b — tán xạ cỡ lớp: 16 điểm nằm thành dải thế nào, và dải ấy kéo theo gì.

    Tách ra làm 2 hình riêng để dễ đưa vào báo cáo.
    """
    c = cats.sort_values("n")

    # 03a — Scatter cỡ lớp (số tin mỗi ngành)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.set_xscale("log")
    ax.set_xlim(220, 34000)
    ys = np.arange(len(c))
    ax.hlines(ys, ax.get_xlim()[0], c["n"], color=INK, alpha=0.22, lw=1)
    ax.scatter(c["n"], ys, s=78,
               color=[ACCENT if k == C.JUNK_CATEGORY else INK for k in c["category"]],
               zorder=3)
    for yi, (n_i, sh) in enumerate(zip(c["n"], c["share"])):
        ax.text(n_i * 1.09, yi, f"{n_i:,}".replace(",", ".") + f"  ({sh*100:.1f} %)",
                va="center", fontsize=8)
    uniform = len(df) / len(c)
    ax.axvline(uniform, color=ACCENT, ls="--", lw=1.4,
               label="chia đều = {:,.0f}".format(uniform).replace(",", "."))
    ax.set_yticks(ys); ax.set_yticklabels(c["category"])
    ax.set_ylim(-0.8, len(c) - 0.2)
    ax.set_xticks([300, 1000, 3000, 10000])
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_xlabel("Số tin — trục log")
    ax.set_title("Cỡ lớp trải thành một dải liên tục\n"
                 f"dốc Zipf {shape['zipf_slope']:.2f} — không có ngưỡng tự nhiên "
                 "để cắt lớp hiếm", loc="left")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    p = FIG_DIR / "eda-03a-co-lop.png"
    fig.savefig(p, dpi=200); plt.close(fig); out.append(str(p))

    # 03b — Scatter: tỷ lệ công bố lương vs cỡ lớp
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.set_xscale("log")
    ax.scatter(cats["n"], cats["disclosure_rate"] * 100, s=78,
               color=[ACCENT if k == C.JUNK_CATEGORY else INK for k in cats["category"]],
               zorder=3)
    ax.axhline(df["salary_disclosed"].mean() * 100, color=ACCENT, lw=1.4, ls="--")
    ax.set_xticks([300, 1000, 3000, 10000])
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_xlim(230, 15000)
    ax.set_xlabel("Số tin — trục log")
    ax.set_ylabel("% tin có công bố lương")
    ax.set_title("Lớp càng nhỏ càng ít công bố lương\n"
                 f"Pearson(log n, tỷ lệ) = {shape['size_vs_disclosure_pearson_logn']:.3f} "
                 f"(p = {shape['size_vs_disclosure_p']:.4f})", loc="left")
    fig.tight_layout()
    fig.canvas.draw()                      # cần renderer để đo bề rộng chữ
    place_labels(ax, cats["n"], cats["disclosure_rate"] * 100,
                 [short_label(k) for k in cats["category"]], fig.canvas.get_renderer())
    p = FIG_DIR / "eda-03b-cong-bo-luong.png"
    fig.savefig(p, dpi=200); plt.close(fig); out.append(str(p))


def fig_salary_by_category(plt, mpl, df: pd.DataFrame, var: dict, out: list) -> None:
    """04 — hình chính: lương trải thế nào bên trong từng ngành.

    Nằm ngang vì tên ngành dài tới 48 ký tự. Ba lớp thông tin chồng lên nhau: đám
    điểm thật (độ trải), hộp p25–p75 (khối chính), và vạch trung vị. Đường đứt là
    trung vị toàn corpus — mười sáu trung vị ngành đều bám sát nó, và đó chính là
    lập luận của cả hình.
    """
    paid = df[(df["salary_disclosed"] == 1) & df["salary_mid"].notna()]
    order = (paid.groupby("category", observed=True)["salary_mid"]
                 .median().sort_values().index.tolist())
    rng = np.random.default_rng(C.RANDOM_SEED)
    grand = float(paid["salary_mid"].median())

    fig, ax = plt.subplots(figsize=(12.8, 7.4))
    for i, cat in enumerate(order):
        v = paid.loc[paid["category"] == cat, "salary_mid"].to_numpy()
        y = i + rng.uniform(-0.30, 0.30, size=v.size)
        ax.scatter(v, y, s=4, alpha=0.13, color=INK, linewidths=0, rasterized=True)
        p25, p50, p75 = np.quantile(v, [0.25, 0.5, 0.75])
        ax.add_patch(plt.Rectangle((p25, i - 0.20), p75 - p25, 0.40, facecolor="white",
                                   edgecolor=MUTED, lw=1.1, alpha=0.92, zorder=3))
        ax.vlines(p50, i - 0.24, i + 0.24, color=ACCENT, lw=2.6, zorder=4)
        ax.text(640, i, "{:.1f}    n = {:,}".format(p50, len(v)).replace(",", "."),
                va="center", fontsize=8, color=MUTED)

    # Chú thẳng lên đường thay vì dùng legend: hộp legend ở góc nào cũng đè lên
    # một hàng ngành hoặc lên cột số bên phải.
    ax.axvline(grand, color=ACCENT, ls="--", lw=1.5, zorder=2)
    ax.text(grand * 1.06, len(order) - 0.55, f"trung vị toàn corpus = {grand:.1f}",
            color=ACCENT, fontsize=8.5, va="center", ha="left")
    ax.text(640, len(order) - 0.55, "trung vị    số tin", fontsize=8, color=MUTED,
            va="center", style="italic")
    ax.set_xscale("log")
    ax.set_xlim(0.8, 1600)
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100, 200, 500])
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_xlabel(f"Lương giữa khoảng ({C.SALARY_UNIT}) — trục log")
    ax.set_title(
        "Hình 04 · Phân bố lương theo ngành nghề — {:,} tin có công bố lương".format(
            len(paid)).replace(",", ".") +
        "\nhộp = p25–p75, vạch cam = trung vị · ngành chỉ giải thích "
        f"{var['eta_squared_log']*100:.1f} % phương sai log-lương (eta²)", loc="left")
    ax.grid(axis="y", visible=False)

    fig.tight_layout()
    p = FIG_DIR / "eda-04-luong-theo-nganh.png"
    fig.savefig(p, dpi=200); plt.close(fig); out.append(str(p))


def fig_salary_shape(plt, df: pd.DataFrame, sal: dict, out: list) -> None:
    """05a/05b/05c — vì sao nhãn hồi quy là log1p, và cái đuôi to đến đâu.

    Tách ra làm 3 hình riêng để dễ đưa vào báo cáo.
    """
    s = df.loc[(df["salary_disclosed"] == 1) & df["salary_mid"].notna(), "salary_mid"]

    # 05a — Thang thô
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.hist(s[s <= 100], bins=70, color=INK, alpha=0.88)
    ax.axvline(sal["median"], color=ACCENT, lw=1.8, label=f"trung vị {sal['median']:.1f}")
    ax.axvline(sal["mean"], color=ACCENT, ls="--", lw=1.8,
               label=f"trung bình {sal['mean']:.1f}")
    ax.set_xlabel(C.SALARY_UNIT)
    ax.set_ylabel("Số tin")
    ax.set_title(f"Thang thô — skew {sal['skew']:.2f}, kurtosis {sal['kurtosis']:.0f}\n"
                 "(cắt hiển thị ở 100)", loc="left")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = FIG_DIR / "eda-05a-thang-tho.png"
    fig.savefig(p, dpi=200); plt.close(fig); out.append(str(p))

    # 05b — Sau log1p
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.hist(np.log1p(s), bins=70, color=INK, alpha=0.88)
    ax.set_xlabel("log1p(salary_mid)")
    ax.set_ylabel("Số tin")
    ax.set_title(f"Sau log1p — skew {sal['skew_log1p']:.2f}\n"
                 "gần đối xứng: đây là nhãn nhánh hồi quy học", loc="left")
    fig.tight_layout()
    p = FIG_DIR / "eda-05b-sau-log1p.png"
    fig.savefig(p, dpi=200); plt.close(fig); out.append(str(p))

    # 05c — Đuôi trên rào IQR
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    tail = s[s > sal["fence_high"]]
    ax.hist(tail, bins=60, color=ACCENT, alpha=0.85)
    ax.set_yscale("log")
    ax.axvline(C.SALARY_EXTREME_THRESHOLD, color=INK, lw=1.4, ls="--",
               label=f"salary_extreme ≥ {C.SALARY_EXTREME_THRESHOLD:.0f}")
    ax.axvline(C.SALARY_IMPLAUSIBLE_HIGH, color=INK, lw=1.4, ls=":",
               label=f"vô lý > {C.SALARY_IMPLAUSIBLE_HIGH:.0f}")
    ax.set_xlabel(C.SALARY_UNIT)
    ax.set_ylabel("Số tin — trục log")
    ax.set_title("Đuôi trên hàng rào IQR ({:.2f}): {:,} tin".format(
                     sal["fence_high"], sal["n_above_fence"]).replace(",", ".") +
                 f"\ntrong đó {sal['n_implausible_high']} tin > "
                 f"{C.SALARY_IMPLAUSIBLE_HIGH:.0f} gần chắc sai đơn vị", loc="left")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = FIG_DIR / "eda-05c-duoi-iqr.png"
    fig.savefig(p, dpi=200); plt.close(fig); out.append(str(p))


def fig_disclosure(plt, df: pd.DataFrame, cats: pd.DataFrame, out: list) -> None:
    """06 — nhãn hồi quy chỉ có ở một phần dữ liệu, và phần ấy không ngẫu nhiên."""
    fig, ax = plt.subplots(figsize=(9.8, 5.4))
    c = cats.sort_values("disclosure_rate")
    ax.barh(c["category"], c["disclosure_rate"] * 100, color=INK, alpha=0.88)
    for y, (r, n) in enumerate(zip(c["disclosure_rate"] * 100, c["n_disclosed"])):
        ax.text(r + 0.7, y, f"{r:.1f} %  ({n:,})".replace(",", "."), va="center", fontsize=7.5)
    mean = df["salary_disclosed"].mean() * 100
    ax.axvline(mean, color=ACCENT, lw=2, label=f"toàn corpus {mean:.1f} %")
    ax.set_xlim(0, 95)
    ax.set_xlabel("% tin có công bố lương")
    ax.set_title("Hình 06 · Tỷ lệ công bố lương theo ngành — {:,} / {:,} tin".format(
                     int(df["salary_disclosed"].sum()), len(df)).replace(",", ".") +
                 "\nnhánh hồi quy chỉ có nhãn ở phần này, nên loss của nó phải được che",
                 loc="left")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    p = FIG_DIR / "eda-06-cong-bo-luong.png"
    fig.savefig(p, dpi=200); plt.close(fig); out.append(str(p))


def fig_token_length(plt, tok: dict, out: list) -> None:
    """07 — cái giá của max_len: bao nhiêu tin bị cắt, và cắt mất bao nhiêu chữ."""
    edges = np.asarray(tok["hist_edges"][:-1])
    counts = np.asarray(tok["hist_counts"])
    width = tok["hist_edges"][1] - tok["hist_edges"][0]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.5),
                             gridspec_kw={"width_ratios": [1.55, 1]})

    ax = axes[0]
    ax.bar(edges, counts, width=width, align="edge", color=INK, alpha=0.85)
    for m, ls in zip(MAX_LEN_GRID, ["-.", "-", ":"]):
        share = tok["truncated_share"][str(m)] * 100
        ax.axvline(m, color=ACCENT, ls=ls, lw=1.8,
                   label=f"max_len = {m} → cắt {share:.1f} % số tin")
    ax.set_xlim(0, tok["hist_edges"][-1])
    ax.set_xlabel("Số token PhoBERT của chuỗi đầu vào")
    ax.set_ylabel("Số tin")
    ax.set_title("Hình 07 · Độ dài đầu vào — {:,} tin (train + dev)".format(
                     tok["n"]).replace(",", ".") +
                 "\ntrung vị {:.0f} · p95 {:.0f} · tối đa {:,}".format(
                     tok["median"], tok["p95"], tok["max"]).replace(",", "."), loc="left")
    ax.legend(fontsize=8)

    ax = axes[1]
    ms = [str(m) for m in MAX_LEN_GRID]
    trunc = [tok["truncated_share"][m] * 100 for m in ms]
    kept = [tok["tokens_kept_share"][m] * 100 for m in ms]
    x = np.arange(len(ms))
    ax.bar(x - 0.19, trunc, 0.38, color=ACCENT, alpha=0.88, label="% tin bị cắt")
    ax.bar(x + 0.19, kept, 0.38, color=INK, alpha=0.88, label="% chữ giữ lại")
    for xi, (t, k) in enumerate(zip(trunc, kept)):
        ax.text(xi - 0.19, t + 1.8, f"{t:.1f}", ha="center", fontsize=8)
        ax.text(xi + 0.19, k + 1.8, f"{k:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([f"max_len={m}" for m in ms])
    ax.set_ylim(0, 116)
    ax.set_ylabel("%")
    ax.set_title("Cái giá của mỗi mức cắt\nmặc định hiện tại của dl/encode.py là 256",
                 loc="left")
    ax.legend(fontsize=8, loc="center right")

    fig.tight_layout()
    p = FIG_DIR / "eda-07-do-dai-token.png"
    fig.savefig(p, dpi=200); plt.close(fig); out.append(str(p))


def plots(df: pd.DataFrame, cats: pd.DataFrame, comp: pd.DataFrame, summary: dict) -> list:
    mpl, plt = _setup()
    out = []
    fig_completeness(plt, comp, len(df), out)
    fig_imbalance(plt, cats, summary["distribution"], len(df), out)
    fig_class_scatter(plt, mpl, cats, summary["distribution"], df, out)
    fig_salary_by_category(plt, mpl, df, summary["variance"], out)
    fig_salary_shape(plt, df, summary["salary"], out)
    fig_disclosure(plt, df, cats, out)
    if summary.get("tokens"):
        fig_token_length(plt, summary["tokens"], out)
    else:
        print(f"bỏ hình 07: chưa có {TOKEN_JSON} — chạy `--tokens` trong .venv-dl")
    return out


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--no-floors", action="store_true",
                    help="bỏ mốc trần — không đọc splits, chạy nhanh hơn nhiều")
    ap.add_argument("--tokens", action="store_true",
                    help="chỉ đo độ dài token (cần .venv-dl) rồi ghi đệm và thoát")
    args = ap.parse_args()

    if args.tokens:
        TOKEN_JSON.parent.mkdir(parents=True, exist_ok=True)
        tok = measure_tokens()
        TOKEN_JSON.write_text(json.dumps(tok, indent=2, ensure_ascii=False), encoding="utf-8")
        print("độ dài token: trung vị {:.0f} · p95 {:.0f} · tối đa {:,}".format(
            tok["median"], tok["p95"], tok["max"]))
        for m in MAX_LEN_GRID:
            print(f"  max_len={m:4d} → cắt {tok['truncated_share'][str(m)]*100:5.2f} % số tin, "
                  f"giữ {tok['tokens_kept_share'][str(m)]*100:5.2f} % số token")
        print(f"-> {TOKEN_JSON}")
        return

    df, prov, raw = load_corpus()
    comp = completeness(raw)
    cats = per_category(df)
    paid = df.loc[df["salary_disclosed"] == 1, "salary_mid"].dropna()
    counts = df["category"].value_counts()

    summary = {
        "scope": "tệp gốc, sau khử trùng lặp chính xác — mô tả, không dùng để chọn mô hình",
        "source": prov,
        "completeness": comp.to_dict("records"),
        "n_rows": int(len(df)),
        "n_classes": int(counts.size),
        "class_counts": {k: int(v) for k, v in counts.items()},
        "imbalance_ratio": float(counts.max() / counts.min()),
        "largest_class": counts.index[0],
        "largest_class_share": float(counts.iloc[0] / len(df)),
        "smallest_class": counts.index[-1],
        "junk_share": float(counts.get(C.JUNK_CATEGORY, 0) / len(df)),
        "disclosure_rate": float(df["salary_disclosed"].mean()),
        "salary": describe_salary(paid),
        "salary_is_range_share": float(
            df.loc[df["salary_disclosed"] == 1, "salary_is_range"].mean()),
        "distribution": distribution_shape(cats, len(df)),
        "variance": variance_split(df),
        "text": text_shape(df),
        "per_category": cats.to_dict("records"),
        "unit": C.SALARY_UNIT,
        "floors": None if args.no_floors else floors(),
    }
    if TOKEN_JSON.exists():
        summary["tokens"] = json.loads(TOKEN_JSON.read_text(encoding="utf-8"))
    if not args.no_plot:
        summary["figures"] = plots(df, cats, comp, summary)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    s, d, v = summary["salary"], summary["distribution"], summary["variance"]
    print(f"tệp gốc {prov['rows_raw']:,} dòng → {prov['rows_after_exact_dedup']:,} sau khử "
          f"trùng lặp ({prov['dropped_exact_duplicates']:,} bản sao)")
    print(f"{summary['n_classes']} lớp · lệch {summary['imbalance_ratio']:.1f}:1 · "
          f"Gini {d['gini']:.3f} · dốc Zipf {d['zipf_slope']:.2f}")
    print(f"trường rỗng nhất: {comp.iloc[0]['column']} chỉ đầy "
          f"{comp.iloc[0]['fill_rate']*100:.1f} %")
    print(f"công bố lương {summary['disclosure_rate']*100:.1f} % → {s['n']:,} nhãn hồi quy "
          f"({summary['salary_is_range_share']*100:.1f} % là một khoảng, không phải một số)")
    print(f"salary_mid  trung vị {s['median']:.1f} · trung bình {s['mean']:.1f} · "
          f"skew {s['skew']:.2f} → log1p {s['skew_log1p']:.2f}")
    print(f"biên  min {s['min']:.2f} · max {s['max']:.1f} · p99 {s['p99']:.1f} · "
          f"ngoài hàng rào IQR {s['n_above_fence']:,} trên / {s['n_below_fence']:,} dưới")
    print(f"vô lý  <{C.SALARY_IMPLAUSIBLE_LOW} triệu: {s['n_implausible_low']:,} · "
          f">{C.SALARY_IMPLAUSIBLE_HIGH} triệu: {s['n_implausible_high']:,}")
    print(f"cỡ lớp  bốn lớp lớn nhất giữ {d['top4_share']*100:.1f} % · "
          f"bốn lớp nhỏ nhất {d['bottom4_share']*100:.1f} %")
    print(f"cỡ lớp ↔ tỷ lệ công bố: Pearson(log n) {d['size_vs_disclosure_pearson_logn']:.3f} "
          f"(p {d['size_vs_disclosure_p']:.4f})")
    print(f"ngành giải thích {v['eta_squared_log']*100:.1f} % phương sai log-lương "
          f"(trung vị ngành trải {v['median_min']:.1f} → {v['median_max']:.1f})")
    t = summary["text"]["words"]
    print(f"độ dài văn bản  trung vị {t['median']:.0f} từ · p95 {t['p95']:.0f} · "
          f"tối đa {t['max']:,}")
    if summary["floors"]:
        b = summary["floors"]
        print(f"mốc trần (học train, chấm dev): trung vị cho mọi tin → MAE "
              f"{b['salary_global_median']['mae']:.2f} triệu · biết đúng ngành → "
              f"{b['salary_category_median_oracle']['mae']:.2f} triệu · đoán lớp đông nhất "
              f"→ macro-F1 {b['category_majority']['macro_f1']:.4f}")
    print(f"-> {OUT_JSON}")


if __name__ == "__main__":
    main()
