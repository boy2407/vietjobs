"""VietJobs-37K as an external evaluation set for the ``category`` task.

The external corpus (``data/external/vietjobs37k/``) differs from ours in three
ways that decide how it may be scored:

1. **Its text is one string** — ``[TITLE] .. [REQ] .. [DESC] ..`` — so it must be
   split back into the three fields ``dl/text.py`` concatenates, in *our* order
   (title, description, requirements), not the file's.
2. **Its labels are 60 and multi-label**; ours are 16 and single-label. A
   crosswalk (``crosswalk_60_to_16.json``) maps each of the 60 to one of ours or
   to ``null`` (= not scorable). A record with several labels can be scored
   *leniently* (prediction hits any mapped target) or *strictly* (single-label
   records only, ordinary metrics).
3. **It overlaps our corpus.** Same portals, overlapping crawl window. Every
   record whose ``group_key`` (the project's own near-duplicate notion,
   ``dataset.py``) appears in any of our splits is dropped before scoring —
   otherwise the "external" number is partly an in-sample number.

Pure pandas; no torch. The embedding + head part lives in
``scripts/eval_external.py`` because it needs ``.venv-dl``.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from . import config as C
from . import vitext as V

EXTERNAL_DIR = C.ROOT / "data" / "external" / "vietjobs37k"
CROSSWALK_PATH = EXTERNAL_DIR / "crosswalk_60_to_16.json"
SUBSETS = {
    "gold": EXTERNAL_DIR / "data" / "gold_anchor_1000.jsonl",
    "test": EXTERNAL_DIR / "data" / "test.jsonl",
    "dev": EXTERNAL_DIR / "data" / "dev.jsonl",
    "train": EXTERNAL_DIR / "data" / "train.jsonl",
}

_MARKERS = ("TITLE", "REQ", "DESC")
_MARKER_RE = re.compile(r"\[(TITLE|REQ|DESC)\]")
_FIELD_OF = {"TITLE": "job_title", "REQ": "requirements_text", "DESC": "description"}


# ---------------------------------------------------------------------------
# 1. Text → three fields
# ---------------------------------------------------------------------------


def parse_37k_text(text: object) -> Dict[str, str]:
    """``"[TITLE] a [REQ] b [DESC] c"`` → ``{job_title, description, requirements_text}``.

    Missing sections give ``""``; a literal ``nan`` placeholder (kept by the
    release for traceability) also gives ``""``. Text before the first marker
    is treated as the title, so a record without markers is still usable.
    """
    s = "" if text is None else str(text)
    out = {"job_title": "", "description": "", "requirements_text": ""}
    pieces = _MARKER_RE.split(s)
    # split() yields [pre, marker, body, marker, body, ...]
    pre = pieces[0].strip()
    if pre:
        out["job_title"] = pre
    for marker, body in zip(pieces[1::2], pieces[2::2]):
        body = body.strip()
        if body.lower() == "nan":
            body = ""
        out[_FIELD_OF[marker]] = body
    return out


def load_37k(path: Path) -> pd.DataFrame:
    """One jsonl split → frame with the three raw fields + label metadata."""
    rows: List[dict] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            fields = parse_37k_text(r.get("text", ""))
            rows.append({
                "id": r.get("id"),
                "job_title": fields["job_title"],
                "description": fields["description"],
                "requirements_text": fields["requirements_text"],
                "labels": list(r.get("labels", [])),
                "label_ids": [int(x) for x in r.get("label_ids", [])],
                "num_labels": int(r.get("num_labels", len(r.get("labels", [])))),
                "source": r.get("source", ""),
                "confidence": r.get("confidence"),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Crosswalk 60 → 16
# ---------------------------------------------------------------------------


def load_crosswalk(path: Path = CROSSWALK_PATH) -> dict:
    cw = json.loads(Path(path).read_text(encoding="utf-8"))
    if "map" not in cw:
        raise ValueError(f"{path}: no 'map' key")
    return cw


def crosswalk_targets(cw: dict) -> Dict[int, Optional[str]]:
    """``label_id → our class or None``."""
    return {int(k): v.get("target") for k, v in cw["map"].items()}


def apply_crosswalk(df: pd.DataFrame, cw: dict) -> Tuple[pd.DataFrame, dict]:
    """Add ``targets`` (list of our classes, deduplicated, order kept),
    ``n_targets``, ``strict_target`` (only when the record has exactly one
    external label and it maps somewhere). Records with ``n_targets == 0`` are
    unscorable; the caller decides whether to drop them. Returns the frame and a
    small report of how many records lost how much."""
    to_ours = crosswalk_targets(cw)
    known = set(to_ours)

    def _targets(ids: Iterable[int]) -> List[str]:
        seen: List[str] = []
        for i in ids:
            if i not in known:
                raise KeyError(f"label_id {i} missing from crosswalk")
            t = to_ours[i]
            if t is not None and t not in seen:
                seen.append(t)
        return seen

    out = df.copy()
    out["targets"] = out["label_ids"].map(_targets)
    out["n_targets"] = out["targets"].map(len)
    out["strict_target"] = pd.Series(
        [t[0] if (n == 1 and len(t) == 1) else None
         for n, t in zip(out["num_labels"], out["targets"])],
        index=out.index, dtype=object,
    )
    report = {
        "n": int(len(out)),
        "unscorable_all_null": int((out["n_targets"] == 0).sum()),
        "multi_target": int((out["n_targets"] > 1).sum()),
        "strict_eligible": int(out["strict_target"].notna().sum()),
        "crosswalk_status": cw.get("_status", "unknown"),
    }
    return out, report


# ---------------------------------------------------------------------------
# 3. Overlap with our own corpus
# ---------------------------------------------------------------------------


_DEID_RE = re.compile(r"\[(COMPANY|EMAIL|PHONE|URL)\]")
JACCARD_THRESHOLD = 0.5


def fold(s: object) -> str:
    """Case/accent/punctuation-folded string — the same folding ``group_key``
    hashes, exposed so titles can be compared without hashing."""
    blob = V.fold_accents(V.normalize_unicode(s).lower())
    return V._WS.sub(" ", re.sub(r"[^a-z0-9 ]+", "", blob)).strip()


def _words(s: object) -> set:
    return set(fold(_DEID_RE.sub(" ", str(s))).split())


class OurIndex:
    """What we need of our own corpus to spot an external repost.

    ``group_ids`` per split (exact near-duplicate hash) and, per folded title,
    the word sets of every description carrying that title (fuzzy path).
    """

    def __init__(self, split_names: Iterable[str] = ("train", "dev", "test"),
                 out_dir: Path = C.SPLIT_DIR):
        self.group_ids: Dict[str, set] = {}
        self.by_title: Dict[str, List[Tuple[str, set]]] = {}
        for name in split_names:
            self.add(name, pd.read_csv(Path(out_dir) / f"{name}.csv",
                                       usecols=["group_id", "job_title", "description"]))

    def add(self, name: str, df: pd.DataFrame) -> "OurIndex":
        self.group_ids[name] = set(df["group_id"].astype(str))
        for t, d in zip(df["job_title"].fillna(""), df["description"].fillna("")):
            self.by_title.setdefault(fold(t), []).append((name, _words(d)))
        return self


def overlap_mask(df: pd.DataFrame, index: OurIndex,
                 threshold: float = JACCARD_THRESHOLD) -> Tuple[pd.Series, dict]:
    """True where an external record is a repost of one of ours.

    Two paths, either one is enough:

    * **exact** — the record's ``group_key`` (the project's own near-duplicate
      hash, ``dataset.py``) exists in one of our splits;
    * **fuzzy** — same folded title *and* word-set Jaccard of the descriptions
      ≥ ``threshold``, after stripping the de-identification tokens
      (``[COMPANY]``…) that the release inserted. Needed because those tokens
      and the release's reformatting defeat the exact hash almost always.

    The threshold is deliberately low (0.5, vs 0.85 in the release's own
    audit): dropping a genuine external record costs a little sample; keeping
    an in-sample record inflates the whole number. The report also counts the
    stricter 0.85 hits so both readings are on record.
    """
    keys = pd.Series([V.group_key(t, d, r) for t, d, r in
                      zip(df["job_title"], df["description"], df["requirements_text"])],
                     index=df.index)
    exact = pd.Series(False, index=df.index)
    per_split = {name: pd.Series(False, index=df.index) for name in index.group_ids}
    for name, ids in index.group_ids.items():
        hit = keys.isin(ids)
        per_split[name] |= hit
        exact |= hit

    fuzzy = pd.Series(False, index=df.index)
    strict85 = 0
    title_hits = 0
    for i, (t, d) in enumerate(zip(df["job_title"], df["description"])):
        cands = index.by_title.get(fold(t))
        if not cands:
            continue
        title_hits += 1
        w = _words(d)
        best, best_split = 0.0, None
        for name, ow in cands:
            union = len(w | ow)
            j = len(w & ow) / union if union else 0.0
            if j > best:
                best, best_split = j, name
        if best >= threshold:
            fuzzy.iloc[i] = True
            per_split[best_split].iloc[i] = True
        if best >= 0.85:
            strict85 += 1

    mask = exact | fuzzy
    report = {
        "n": int(len(df)),
        "overlap_total": int(mask.sum()),
        "overlap_pct": round(100.0 * float(mask.mean()), 2) if len(df) else 0.0,
        "exact_hash": int(exact.sum()),
        "fuzzy_title_and_jaccard": int(fuzzy.sum()),
        "jaccard_threshold": threshold,
        "fuzzy_at_0.85": int(strict85),
        "same_title_any_jaccard": int(title_hits),
        **{f"overlap_{k}": int(v.sum()) for k, v in per_split.items()},
    }
    return mask, report


# ---------------------------------------------------------------------------
# 4. Scoring conventions
# ---------------------------------------------------------------------------


def lenient_truth(pred: Iterable[str], targets: Iterable[List[str]]) -> List[str]:
    """Resolve multi-target rows to one gold label for standard metrics.

    Convention: if the prediction is one of the row's targets, the gold *is*
    the prediction (credit given); otherwise the gold is the first target. This
    makes accuracy equal to "hit any target" and lets macro-F1 be computed with
    the usual functions. It is generous by construction and must be reported
    next to the strict number, never alone.
    """
    out = []
    for p, ts in zip(pred, targets):
        if not ts:
            raise ValueError("lenient_truth on a row with no targets — drop those first")
        out.append(p if p in ts else ts[0])
    return out


# ---------------------------------------------------------------------------
# 5. Salary read off the text (silver)
# ---------------------------------------------------------------------------
# The release has no salary field, but ~12 % of postings state pay in the text.
# These patterns *label*, they do not mask, so precision beats recall: a figure
# counts only when it follows a pay word closely, in a plausible monthly range.
# Real shapes seen in train: "Lương: 10-15 triệu", "Lương 20.000.000",
# "Lương cứng: 8,000,000", "lương: 8 15 triệu", "upto 35tr", "Từ 9.000.000 đ".

_PAY_WORD = re.compile(r"\b(?:lương|luong|thu nhập|thu nhap|salary|income)\b", re.IGNORECASE)
# Between the pay word and the figure: short, same clause, no other benefit.
_GAP_MAX = 40
# Once a figure is read, a ladder may continue without repeating the pay word:
# "Lương:- Junior: Từ 10.000.000 đến 25.000.000 … - Mid-Level: Từ 12.000.000 …".
# Naming only the first rung would put the ad's top pay outside its own range.
_CHAIN_MAX = 60
_GAP_BREAK = re.compile(
    r"[\n;|]|\bnăm\b|\bnam\b|annual|hỗ trợ|thưởng|phụ cấp|trợ cấp|hoa hồng|bảo hiểm|nghỉ|ăn trưa|tháng 13|kpi"
    r"|ngân sách|ngan sach|trị giá|tri gia|hạn mức|han muc|doanh thu",
    re.IGNORECASE,
)
# "M" is triệu in ad shorthand: "Thu nhập 30-40M", "Lương Upto 15M" (T9.2b).
_UNIT = r"(?P<unit>triệu|trieu|tr|củ|vnd|vnđ|đồng|đ|usd|\$|k|m)?"
_SALARY_FIGURE = re.compile(
    rf"(?P<lo>{V._NUM})(?:\s*(?:-|–|—|~|đến|tới|to|\s)\s*(?P<hi>{V._NUM}))?\s*{_UNIT}(?![\w])",
    re.IGNORECASE,
)
_DONG_WRITTEN = re.compile(r"\d{1,3}(?:[.,]\d{3}){2,}")
# De-identification stripped the "/", so "100 triệu/năm" reaches us as "100
# triệu năm": the period word alone must refuse it. Only the toned spellings
# count there — bare "nam" is the gender in "Thu nhập 8tr Nam nữ tốt nghiệp".
_NOT_MONTHLY = re.compile(
    r"^\s*(?:(?:/|một|mỗi|per)\s*(?:giờ|gio|h\b|ngày|ngay|ca\b|năm|nam\b|tuần|hour|day|year)"
    r"|(?:giờ|ngày|năm|tuần|hour|day|year)\b)",
    re.IGNORECASE,
)


def _to_million(num: str, unit: str) -> Optional[float]:
    """One figure → triệu VND. ``None`` when the unit makes it unreadable."""
    unit = (unit or "").lower()
    if unit in ("usd", "$"):
        return None
    if re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", num):
        value = float(re.sub(r"[.,]", "", num))
    else:
        value = float(num.replace(",", "."))
    if value >= 100_000:              # written out in dong, unit or not
        return value / 1e6
    if unit in ("triệu", "trieu", "tr", "củ", "m"):
        return value
    if unit == "k":
        return value / 1e3
    return None                       # "Lương 25.000", "Lương: 12" — no safe reading


def _read_figure(m: "re.Match", tail: str) -> Tuple[Optional[List[float]], str]:
    """One ``_SALARY_FIGURE`` match → its bounds in triệu VND, or why it is refused.

    The refusal code is what the coverage audit (:func:`audit_salary`) groups by,
    so a widening decision is made against counted evidence, not a guess.
    """
    if _NOT_MONTHLY.match(tail):
        return None, "not_monthly"
    # "9 15 triệu" is a range only when a unit closes it; else it is noise —
    # unless both sides are written out in dong ("Lương cơ bản 10.000.000
    # 13.000.000"), where the shape itself is unambiguous (T9.2c).
    if m.group("hi") and not re.search(r"[-–—~]|đến|tới|to", m.group(0)) and not m.group("unit"):
        if not all(_DONG_WRITTEN.fullmatch(m.group(g)) for g in ("lo", "hi")):
            return None, "loose_range"
    figures = [_to_million(m.group(g), m.group("unit")) for g in ("lo", "hi") if m.group(g)]
    if not figures or any(f is None for f in figures):
        return None, "unit"
    if any(not C.SALARY_IMPLAUSIBLE_LOW <= f <= C.SALARY_IMPLAUSIBLE_HIGH for f in figures):
        return None, "out_of_range"
    return figures, ""


# A section header stands in for the pay word: "Quyền lợi:- 3.000.000 -
# 16.000.000 VNĐ" states pay without ever saying "lương" (T9.2a).
_HEADER_WORD = re.compile(
    r"(?:quyền lợi|quyen loi|phúc lợi|phuc loi|chế độ|che do|chính sách|chinh sach|benefits?)",
    re.IGNORECASE,
)
_ANCHOR = re.compile(f"{_PAY_WORD.pattern}|{_HEADER_WORD.pattern}", re.IGNORECASE)
# With no anchor at all, only an explicit range closed by a unit is pay:
# "Kế Toán Trưởng (45 - 60 Triệu)" — a lone figure there is an allowance (T9.2d).
_BARE_RANGE = re.compile(
    rf"(?P<lo>{V._NUM})\s*(?:-|–|—|~|đến|tới)\s*(?P<hi>{V._NUM})\s*(?P<unit>triệu|trieu|tr|trđ)(?![\w])",
    re.IGNORECASE,
)


def _scan(s: str) -> Tuple[List[float], List[str], List[str]]:
    """Walk every pay-word anchor once: figures read, snippets, refusal codes.

    One walk serves both callers — :func:`extract_salary` takes the figures,
    :func:`audit_salary` takes the codes — so the audit can never describe a
    rule the extractor does not apply.
    """
    values: List[float] = []
    snippets: List[str] = []
    reasons: List[str] = []
    read_to = 0
    for anchor in _ANCHOR.finditer(s):
        if anchor.end() < read_to:        # already swallowed by a ladder
            continue
        cursor, limit = anchor.end(), _GAP_MAX
        while True:
            window = s[cursor: cursor + limit + 40]
            m = _SALARY_FIGURE.search(window)
            if not m:
                break
            if m.start() > limit:
                reasons.append("gap")
                break
            if _GAP_BREAK.search(window[: m.start()]):
                reasons.append("break_word")
                break
            figures, why = _read_figure(m, window[m.end(): m.end() + 12])
            if figures is None:
                reasons.append(why)
                break
            values.extend(figures)
            cursor, limit = cursor + m.end(), _CHAIN_MAX
        if cursor > anchor.end():
            snippets.append(s[anchor.start(): cursor].strip())
            read_to = cursor
    return values, snippets, reasons


def extract_salary(text: object) -> Optional[Tuple[float, float, str]]:
    """Monthly pay stated in a posting → ``(lo, hi, snippet)`` in triệu VND.

    Every readable figure after a pay word is collected, including the rungs of
    a level ladder that follow it; several levels (Junior/Senior,
    probation/official) collapse to their overall min and max. ``None`` when
    nothing readable is stated ("thỏa thuận", USD, hourly or annual pay, figures
    outside ``[SALARY_IMPLAUSIBLE_LOW, SALARY_IMPLAUSIBLE_HIGH]``).
    """
    s = V.normalize_unicode(text)
    if not s:
        return None
    values, snippets, _ = _scan(s)
    if not values:
        return _bare_range(s)
    return min(values), max(values), " | ".join(dict.fromkeys(snippets))


def _bare_range(s: str) -> Optional[Tuple[float, float, str]]:
    """No pay word, no header — only "45 - 60 triệu" in the clause counts."""
    for m in _BARE_RANGE.finditer(s):
        if _GAP_BREAK.search(s[max(0, m.start() - 25): m.start()]):
            continue
        figures, _ = _read_figure(m, s[m.end(): m.end() + 12])
        if figures:
            return min(figures), max(figures), m.group(0).strip()
    return None


# ---------------------------------------------------------------------------
# 6. Coverage audit (T9.1): why a posting has no salary
# ---------------------------------------------------------------------------
# Every posting lands in exactly one bucket, so "13.8 % have a salary" can be
# read as "of the rest, N state a figure we refuse and M state none at all".

_HEADER_GAP = 60
_NEGOTIABLE = re.compile(
    r"(?:thỏa thuận|thoả thuận|thoa thuan|cạnh tranh|canh tranh|theo năng lực|theo nang luc|hấp dẫn|hap dan)",
    re.IGNORECASE,
)
# Any money-looking figure, however it is refused — the denominator of the audit.
_ANY_MONEY = re.compile(
    rf"{V._NUM}\s*(?:triệu|trieu|tr|củ|vnd|vnđ|đồng|đ|usd|\$|k)\b|\d{{1,3}}[.\s]\d{{3}}[.\s]\d{{3}}",
    re.IGNORECASE,
)

BUCKETS = (
    "read",
    "payword_break_word", "payword_gap", "payword_unit",
    "payword_out_of_range", "payword_not_monthly", "payword_loose_range",
    "payword_far", "header_figure", "figure_only", "negotiable", "no_figure",
)

# T9.3 — the twelve buckets collapse to the three reasons a reader of the file
# needs: the ad names a price we would not read, it says pay is negotiable, or
# it is silent. An empty salary is then never silent about *why* it is empty.
NOTE_OF_BUCKET = {
    "read": "",
    "negotiable": "negotiable",
    "no_figure": "no_figure",
}


def salary_note(text: object) -> str:
    """Why a posting has no salary — ``""`` when it has one (T9.3)."""
    return NOTE_OF_BUCKET.get(audit_salary(text), "figure_unread")


RULES = ("payword", "header", "m_unit", "dong_range", "bare_range")


def salary_rule(text: object) -> str:
    """Which widening rule read this posting's pay — ``""`` when none did.

    Attribution is read back off the snippet, not recorded during the scan: the
    review (T9.5) needs a per-rule precision, and a rule that cannot be pointed
    at cannot be rolled back on its own.
    """
    found = extract_salary(text)
    if not found:
        return ""
    snippet = found[2]
    if not _ANCHOR.search(snippet):
        return "bare_range"
    if _HEADER_WORD.search(snippet) and not _PAY_WORD.search(snippet):
        return "header"
    if re.search(r"\d\s*m\b", snippet, re.IGNORECASE):
        return "m_unit"
    if re.search(rf"{_DONG_WRITTEN.pattern}\s+{_DONG_WRITTEN.pattern}", snippet):
        return "dong_range"
    return "payword"


def template_ids(description: pd.Series, requirements: pd.Series) -> pd.Series:
    """One id per identical posting body — the reposted-ad group (T9.4).

    An ad reposted per branch keeps its body and changes only the title, so the
    body is the template. A split that ignores this puts the same ad in `train`
    and in `dev`. The body is description *and* requirements because the largest
    template in this corpus (711 rows, one bank) has an empty description and
    repeats itself in the requirements. A body blank on both sides groups with
    nothing.
    """
    body = (description.fillna("").str.strip() + "\n"
            + requirements.fillna("").str.strip())
    blank = body.str.strip().eq("")
    key = body.where(~blank, "__row" + description.index.astype(str))
    return pd.Series(pd.factorize(key)[0], index=description.index)


def audit_salary(text: object) -> str:
    """Which coverage bucket a posting falls in — one of :data:`BUCKETS`.

    ``read`` is what :func:`extract_salary` returns a figure for; every
    ``payword_*`` bucket names the rule that refused a figure the posting does
    state next to a pay word; ``header_figure`` and ``figure_only`` are figures
    with no pay word at all; the last two state no figure.
    """
    s = V.normalize_unicode(text)
    if not s:
        return "no_figure"
    values, _, reasons = _scan(s)
    if values or _bare_range(s):
        return "read"
    money = _ANY_MONEY.search(s)
    if reasons:
        return "payword_" + reasons[0]
    if _PAY_WORD.search(s) and money:
        return "payword_far"
    if money:
        for header in _HEADER_WORD.finditer(s):
            near = _ANY_MONEY.search(s, header.end(), header.end() + _HEADER_GAP)
            if near and not _GAP_BREAK.search(s[header.end(): near.start()]):
                return "header_figure"
        return "figure_only"
    if _NEGOTIABLE.search(s):
        return "negotiable"
    return "no_figure"
