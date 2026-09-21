"""Vietnamese text processing for job postings.

Every function here is pure and deterministic so it can run inside a sklearn
pipeline or ahead of the split without changing behaviour. The order the steps
are meant to compose in is documented in ``docs/02-vietnamese-nlp.md`` and
implemented by :func:`preprocess`.

Sections below are numbered to match the plan:

    2.1 normalize_unicode      Unicode NFC, line endings, bullets, whitespace
    2.2 normalize_tone         unify the two Vietnamese tone-placement styles
    2.3 expand_abbreviations   NV -> nhân viên, BHXH -> bảo hiểm xã hội, ...
    2.4 segment                word segmentation (nhân_viên kinh_doanh)
    2.5 remove_stopwords       optional, ablation-gated
    2.6 fold_accents           accent-folded mirror, used by group_key
    2.7 normalize_province     hà đông -> hà nội
    2.8 mask_salary            replace pay figures with <SALARY>
    2.9 experience_to_months / parse_list_field / split_locations
"""
from __future__ import annotations

import ast
import functools
import hashlib
import json
import os
import re
import unicodedata

from . import config as C

# ---------------------------------------------------------------------------
# 2.1  Unicode and whitespace
# ---------------------------------------------------------------------------

_WS = re.compile(r"\s+")
_BULLET = re.compile(r"^[\s\-\*•·●▪◦\+]+", re.MULTILINE)
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

_NULLISH = frozenset({"", "nan", "none", "null", "unk", "n/a", "na", "-"})


def normalize_unicode(text: object) -> str:
    """NFC-normalise, drop bullets and control chars, collapse whitespace.

    NFC first: without it ``"ế"`` can exist as one code point or as ``e`` plus
    two combining marks, which a tokenizer would see as two different tokens.
    """
    if text is None:
        return ""
    s = str(text)
    if s.strip().lower() in _NULLISH:
        return ""
    s = unicodedata.normalize("NFC", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = _CONTROL.sub(" ", s)
    s = _BULLET.sub(" ", s)
    return _WS.sub(" ", s).strip()


# ---------------------------------------------------------------------------
# 2.2  Tone placement
# ---------------------------------------------------------------------------
# Vietnamese has two conventions for where the tone mark sits in the diphthongs
# "oa", "oe" and "uy": on the first vowel ("hòa", "thúy") or on the second
# ("hoà", "thuý"). Both are correct spellings of the same word, and NFC does
# NOT unify them — so "hoà" and "hòa" tokenize as two different words.
#
# Canonical form chosen here: tone on the SECOND vowel ("hoà", "thuý").
# That direction is safe for "qu-": "quý" already has the mark on the second
# vowel and is therefore left untouched, whereas normalising the other way
# would corrupt it into "qúy".

_TONE_PAIRS = {
    "òa": "oà", "óa": "oá", "ỏa": "oả", "õa": "oã", "ọa": "oạ",
    "òe": "oè", "óe": "oé", "ỏe": "oẻ", "õe": "oẽ", "ọe": "oẹ",
    "ùy": "uỳ", "úy": "uý", "ủy": "uỷ", "ũy": "uỹ", "ụy": "uỵ",
}
# Uppercase and title-case variants, so "HÒA" and "Hòa" normalise too.
_TONE_MAP: dict[str, str] = {}
for _k, _v in _TONE_PAIRS.items():
    _TONE_MAP[_k] = _v
    _TONE_MAP[_k.upper()] = _v.upper()
    _TONE_MAP[_k.capitalize()] = _v.capitalize()

_TONE_RE = re.compile("|".join(sorted(map(re.escape, _TONE_MAP), key=len, reverse=True)))


def normalize_tone(text: object) -> str:
    """Unify tone placement in oa / oe / uy diphthongs."""
    s = text if isinstance(text, str) else normalize_unicode(text)
    if not s:
        return ""
    return _TONE_RE.sub(lambda m: _TONE_MAP[m.group(0)], s)


# ---------------------------------------------------------------------------
# 2.3  Abbreviations
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _abbreviations() -> tuple[dict[str, str], list[dict]]:
    """Load the abbreviation table.

    Returns ``(simple, contextual)``. ``simple`` maps an unambiguous token to
    its expansion. ``contextual`` holds the ambiguous ones — each entry needs a
    regex that must match nearby text before the expansion fires.
    """
    if not C.ABBREVIATIONS.exists():
        return {}, []
    data = json.loads(C.ABBREVIATIONS.read_text(encoding="utf-8"))
    simple = {k.lower(): v for k, v in data.get("simple", {}).items()}
    return simple, data.get("contextual", [])


_TOKEN_RE = re.compile(r"[0-9A-Za-zÀ-ỹĐđ][0-9A-Za-zÀ-ỹĐđ.]*")


def expand_abbreviations(text: object) -> str:
    """Expand recruitment-domain abbreviations.

    Unambiguous entries are replaced on a whole-token basis. Ambiguous ones
    (``TP`` = "trưởng phòng" or "thành phố"; ``CP`` = "cổ phần" or "chính phủ")
    only fire when their context pattern matches, and are otherwise left alone —
    a wrong expansion is worse than no expansion.
    """
    s = text if isinstance(text, str) else normalize_unicode(text)
    if not s:
        return ""
    simple, contextual = _abbreviations()

    for rule in contextual:
        s = re.sub(rule["pattern"], rule["replacement"], s, flags=re.IGNORECASE)

    if not simple:
        return s

    def _sub(m: re.Match) -> str:
        tok = m.group(0)
        hit = simple.get(tok.lower().rstrip("."))
        return hit if hit else tok

    return _TOKEN_RE.sub(_sub, s)


# ---------------------------------------------------------------------------
# 2.4  Word segmentation
# ---------------------------------------------------------------------------
# Vietnamese is an isolating language: whitespace separates SYLLABLES, not
# words. "nhân viên kinh doanh" is 2 words spread over 4 syllables. Without
# segmentation, a model sees one "viên" shared by nhân viên / chuyên viên /
# nghiên cứu viên / công viên — four unrelated meanings.

_SEGMENTER_STATE: dict[str, object] = {"loaded": False, "fn": None, "name": None}

#: Environment variable choosing the backend: "underthesea" (default) or "pyvi".
#: It is read from the environment rather than set by a function call because
#: ``segment_many`` fans out over loky worker processes, which re-import this
#: module from scratch — a module-level global set in the parent would not
#: survive the trip, and the workers would silently segment with the default.
SEGMENTER_ENV = "VIETJOBS_SEGMENTER"

_SEGMENTERS = ("underthesea", "pyvi")


def _load_segmenter() -> tuple[str | None, object]:
    """Import the backend named by $VIETJOBS_SEGMENTER. Returns ``(name, fn)``."""
    want = (os.environ.get(SEGMENTER_ENV) or "underthesea").strip().lower()
    if want not in _SEGMENTERS:
        raise SystemExit(f"{SEGMENTER_ENV}={want!r}: choose one of {_SEGMENTERS}")
    try:
        if want == "pyvi":
            from pyvi import ViTokenizer

            return want, ViTokenizer.tokenize
        from underthesea import word_tokenize

        return want, lambda s: word_tokenize(s, format="text")
    except ImportError:
        return None, None


def segmenter_available() -> bool:
    """True if the configured segmenter imported successfully (checked once)."""
    if not _SEGMENTER_STATE["loaded"]:
        _SEGMENTER_STATE["loaded"] = True
        name, fn = _load_segmenter()
        _SEGMENTER_STATE["name"], _SEGMENTER_STATE["fn"] = name, fn
    return _SEGMENTER_STATE["fn"] is not None


def use_segmenter(name: str) -> None:
    """Switch backend at runtime: sets $VIETJOBS_SEGMENTER and drops the cache.

    Setting the variable is the point — ``segment_many`` fans out over worker
    processes that read it on import, so a plain module-level assignment would
    change the parent and nothing else.
    """
    os.environ[SEGMENTER_ENV] = name
    _SEGMENTER_STATE.update(loaded=False, fn=None, name=None)
    # joblib keeps its loky pool alive between calls, and a live worker already
    # imported this module with the OLD value — it inherits the environment it
    # was spawned with, not the current one. Without this shutdown the switch
    # silently does nothing in the workers: two backends, byte-identical output.
    try:
        from joblib.externals.loky import get_reusable_executor

        get_reusable_executor().shutdown(wait=True)
    except Exception:
        pass


def segmenter_name() -> str | None:
    """Which backend is in use, for the manifest and the results log."""
    segmenter_available()
    return _SEGMENTER_STATE["name"]  # type: ignore[return-value]


#: Incremented whenever :func:`segment` falls back to unsegmented text.
segment_failures = 0


def segment(text: object) -> str:
    """``"Nhân viên kinh doanh"`` -> ``"Nhân_viên kinh_doanh"``.

    Falls back to the input unchanged when the backend is unavailable or throws,
    so the pipeline still runs end to end; :data:`segment_failures` counts how
    often that happened and ``dataset.py`` records it in the manifest.
    """
    global segment_failures
    s = text if isinstance(text, str) else normalize_unicode(text)
    if not s:
        return ""
    if not segmenter_available():
        segment_failures += 1
        return s
    try:
        return _SEGMENTER_STATE["fn"](s)  # type: ignore[operator]
    except Exception:
        segment_failures += 1
        return s


def _segment_chunk(chunk: list[str]) -> tuple[list[str], int]:
    """Segment a list of strings in one worker process. Returns (out, failures)."""
    before = segment_failures
    out = [segment(s) for s in chunk]
    return out, segment_failures - before


def segment_many(texts, n_jobs: int = -1, chunk_size: int = 500) -> list[str]:
    """Segment a whole column, in parallel when joblib is available.

    Segmentation dominates dataset build time (~68 min single-threaded for this
    corpus). It is embarrassingly parallel — each document is independent — so
    it is worth spreading over cores. Falls back to a serial map if joblib is
    missing or only one core is usable.
    """
    global segment_failures
    texts = list(texts)
    if not texts or not segmenter_available():
        return [segment(t) for t in texts]

    try:
        import os

        from joblib import Parallel, delayed
    except ImportError:
        return [segment(t) for t in texts]

    workers = (os.cpu_count() or 1) if n_jobs in (-1, None) else n_jobs
    if workers <= 1 or len(texts) < chunk_size * 2:
        return [segment(t) for t in texts]

    chunks = [texts[i : i + chunk_size] for i in range(0, len(texts), chunk_size)]
    results = Parallel(n_jobs=workers, backend="loky")(
        delayed(_segment_chunk)(c) for c in chunks
    )
    out: list[str] = []
    for piece, failures in results:
        out.extend(piece)
        segment_failures += failures  # workers have their own copy of the counter
    return out


# ---------------------------------------------------------------------------
# 2.5  Stopwords
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def stopwords() -> frozenset[str]:
    """Vietnamese stopwords, as underscore-joined words to match segmented text."""
    if not C.STOPWORDS.exists():
        return frozenset()
    words = set()
    for line in C.STOPWORDS.read_text(encoding="utf-8").splitlines():
        line = line.strip().lower()
        if line and not line.startswith("#"):
            words.add(line)
            words.add(line.replace(" ", "_"))
    return frozenset(words)


def remove_stopwords(text: object) -> str:
    """Drop stopwords. Ablation-gated — see docs/02-vietnamese-nlp.md.

    Domain-bearing negations and thresholds ("không", "trên", "tối thiểu") are
    deliberately kept in ``stopwords_vi.txt``'s exclusion list, because
    "không yêu cầu kinh nghiệm" means the opposite of "yêu cầu kinh nghiệm".
    """
    s = text if isinstance(text, str) else normalize_unicode(text)
    if not s:
        return ""
    sw = stopwords()
    if not sw:
        return s
    return " ".join(t for t in s.split() if t.lower() not in sw)


# ---------------------------------------------------------------------------
# 2.6  Accent folding
# ---------------------------------------------------------------------------


def fold_accents(text: object) -> str:
    """Strip diacritics — used by ``group_key`` ONLY.

    5.1% of titles in this dump are written without diacritics ("Nhan Vien
    Kinh Doanh"); folding makes a repost match its accented twin when hashing.

    This must NEVER be applied to the main word channel: in Vietnamese the
    diacritics are the word. má / mà / mả / mã / mạ are five different words.
    """
    s = text if isinstance(text, str) else normalize_unicode(text)
    if not s:
        return ""
    s = s.replace("Đ", "D").replace("đ", "d")
    decomposed = unicodedata.normalize("NFD", s)
    return "".join(c for c in decomposed if unicodedata.category(c) != "Mn")


# ---------------------------------------------------------------------------
# 2.7  Place names
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _province_map() -> dict[str, str]:
    if not C.PROVINCES.exists():
        return {}
    data = json.loads(C.PROVINCES.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for province, aliases in data.items():
        if province.startswith("_"):  # "_readme" and friends are documentation
            continue
        out[_place_key(province)] = province
        for alias in aliases:
            out[_place_key(alias)] = province
    return out


def _place_key(name: str) -> str:
    """Fold a place name to a lookup key: no accents, no punctuation, lowercase."""
    return _WS.sub(" ", re.sub(r"[^a-z0-9 ]+", " ", fold_accents(str(name)).lower())).strip()


# "Tp. Hồ Chí Minh" and "Tỉnh Bắc Ninh" carry an administrative prefix that the
# alias table does not repeat. "quận"/"huyện" are NOT stripped — "quận 1" is a
# lookup key in its own right and would collapse to a bare "1".
_ADMIN_PREFIX = re.compile(r"^(?:tp|thanh pho|tinh|thi xa|tx)\s+")
_ADMIN_PREFIX_VI = re.compile(r"^(?:tp\.?|thành phố|tỉnh|thị xã|tx\.?)\s+", re.IGNORECASE)


def normalize_province(value: object) -> str:
    """Map a district or alias to its province.

    Fixes the fragmentation found in the audit: "hà đông" (a ward of Hanoi)
    was a separate location value from "hà nội".
    """
    s = normalize_unicode(value).lower()
    if not s:
        return "unknown"
    parts = split_locations(s)
    head = parts[0] if parts else s
    table = _province_map()

    key = _place_key(head)
    if key in table:
        return table[key]
    stripped = _ADMIN_PREFIX.sub("", key)
    if stripped in table:
        return table[stripped]
    # Unknown place: keep its real (accented) name, minus any admin prefix, so
    # "TP. Xyz" and "Xyz" still land on one one-hot column.
    return _ADMIN_PREFIX_VI.sub("", head).strip() or head


# ---------------------------------------------------------------------------
# 2.8  Salary masking
# ---------------------------------------------------------------------------
# ~0.2% of descriptions and ~4.9% of benefit lists restate the offered pay.
# Left alone, the salary model reads the answer off its own input. Masking
# keeps the *fact* that pay was mentioned (informative) and drops the *value*.

_SALARY_MASK = " <SALARY> "

# Two shapes: thousands-grouped ("8.000.000", "12,500,000") and plain with an
# optional decimal ("1200", "15", "15,5"). The grouped form must come first, or
# the plain alternative eats only its first group and the unit never matches.
_NUM = r"(?:\d{1,3}(?:[.,]\d{3})+|\d+(?:[.,]\d{1,2})?)"
_RANGE = rf"{_NUM}(?:\s*(?:-|–|—|~|đến|tới|to)\s*{_NUM})?"

_PAT_MILLIONS = re.compile(rf"\b{_RANGE}\s*(?:triệu|trieu|tr|củ)\b", re.IGNORECASE)
_PAT_DONG = re.compile(rf"\b{_RANGE}\s*(?:vnd|vnđ|đồng|dong|đ)\b", re.IGNORECASE)
_PAT_USD = re.compile(rf"(?:usd|\$)\s*{_RANGE}\b|\b{_RANGE}\s*(?:usd|\$)", re.IGNORECASE)

# Words that mark the surrounding text as being about pay.
_PAY_CONTEXT = re.compile(
    r"lương|luong|thu nhập|thu nhap|thù lao|thưởng|phụ cấp|hoa hồng|trợ cấp|"
    r"salary|income|package|gross|net|/\s*tháng|/\s*thang|per month",
    re.IGNORECASE,
)
# Nouns the number is COUNTING rather than paying: "40 triệu người dùng" is
# 40 million users, "500 triệu đồng doanh thu" is revenue. An optional currency
# word may sit in between, hence the "đồng"/"đ" prefix group.
_COUNTING_NOUN = re.compile(
    r"^\s*(?:(?:đồng|đ|vnd|vnđ)\s+)?"
    r"(?:người|nguoi|khách|khach|user|view|lượt|luot|đơn hàng|don hang|"
    r"sản phẩm|san pham|doanh thu|doanh số|doanh so|hợp đồng|hop dong|"
    r"giao dịch|giao dich|thành viên|thanh vien|lượt xem|follower|subscriber)",
    re.IGNORECASE,
)

_CONTEXT_WINDOW = 40


def _is_pay(text: str, start: int, end: int) -> bool:
    """Decide whether a number-plus-unit match is pay or just a quantity.

    Recall matters more than precision here — an unmasked salary figure leaks
    the target, while an over-masked quantity costs one token. So a match counts
    as pay unless it is clearly counting something else, and even then an
    explicit pay word nearby wins.

    Without this check "40 triệu người dùng" (40 million users) and "500 triệu
    đồng doanh thu" (500 million in revenue) would be masked as if they were pay.
    """
    if not _COUNTING_NOUN.match(text[end : end + 48]):
        return True
    window = text[max(0, start - _CONTEXT_WINDOW) : end + _CONTEXT_WINDOW]
    return bool(_PAY_CONTEXT.search(window))


def mask_salary(text: object) -> str:
    """Replace pay figures with ``<SALARY>``, leaving other quantities alone."""
    s = text if isinstance(text, str) else normalize_unicode(text)
    if not s:
        return ""
    for pattern in (_PAT_DONG, _PAT_MILLIONS, _PAT_USD):
        out, cursor, pieces = [], 0, False
        for m in pattern.finditer(s):
            if not _is_pay(s, m.start(), m.end()):
                continue
            out.append(s[cursor : m.start()])
            out.append(_SALARY_MASK)
            cursor = m.end()
            pieces = True
        if pieces:
            out.append(s[cursor:])
            s = "".join(out)
    return _WS.sub(" ", s).strip()


# ---------------------------------------------------------------------------
# 2.9  Structured field parsers
# ---------------------------------------------------------------------------

_EXP_UNITS = {"năm": 12, "nam": 12, "tháng": 1, "thang": 1}
_EXP_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(năm|nam|tháng|thang)", re.IGNORECASE)


def experience_to_months(value: object) -> float:
    """``"2 năm"`` -> 24.0, ``"6 tháng"`` -> 6.0, ``"Không yêu cầu"`` -> 0.0."""
    s = normalize_unicode(value).lower()
    if not s or "không" in s or "khong" in s:
        return 0.0
    m = _EXP_RE.search(s)
    if not m:
        return 0.0
    return float(m.group(1).replace(",", ".")) * _EXP_UNITS.get(m.group(2).lower(), 1)


def parse_list_field(value: object) -> list[str]:
    """Turn ``"['a', 'b']"`` into ``["a", "b"]``; tolerate plain and broken text."""
    if value is None:
        return []
    s = str(value).strip()
    if not s or s.lower() in _NULLISH or s == "[]":
        return []
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = ast.literal_eval(s)
        except (ValueError, SyntaxError):
            parsed = s.strip("[]").split(",")
        if isinstance(parsed, (list, tuple, set)):
            return [x for x in (normalize_unicode(p) for p in parsed) if x]
        one = normalize_unicode(parsed)
        return [one] if one else []
    return [x for x in (normalize_unicode(p) for p in s.split(",")) if x]


def join_list_field(value: object, sep: str = " ; ") -> str:
    """Flatten a list-ish field to one lower-cased string for vectorising."""
    return sep.join(parse_list_field(value)).lower()


def split_locations(value: object) -> list[str]:
    """``"hà nội, hồ chí minh"`` -> ``["hà nội", "hồ chí minh"]`` (order kept)."""
    s = normalize_unicode(value).lower()
    if not s:
        return []
    return [p.strip() for p in re.split(r"[,/;|]", s) if p.strip()]


def split_languages(value: object) -> list[str]:
    """Language list, with the dataset's ``UNK`` sentinel treated as absent."""
    s = normalize_unicode(value).lower()
    if not s or s == "unk":
        return []
    return [p.strip() for p in re.split(r"[,/;|]", s) if p.strip() and p.strip() != "unk"]


# ---------------------------------------------------------------------------
# Grouping key (used by dataset.py to keep reposts inside one split)
# ---------------------------------------------------------------------------


def group_key(*parts: object) -> str:
    """Stable 16-hex-char id for near-duplicate postings.

    Case- and accent-folded so a repost with cosmetic edits lands in the same
    group, and therefore the same split.
    """
    blob = " || ".join(normalize_unicode(p).lower() for p in parts)
    folded = re.sub(r"[^a-z0-9 ]+", "", fold_accents(blob))
    folded = _WS.sub(" ", folded).strip()
    return hashlib.sha1(folded.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Composed pipeline — the canonical order (plan section 2.10)
# ---------------------------------------------------------------------------


def preprocess(
    text: object,
    *,
    tone: bool = True,
    abbrev: bool = True,
    mask: bool = False,
    do_segment: bool = False,
    drop_stopwords: bool = False,
) -> str:
    """Run the Vietnamese pipeline in its canonical order.

    ``mask=True`` is mandatory for the salary and disclosed tasks. Each flag
    corresponds to one ablation row in ``docs/02-vietnamese-nlp.md``.
    """
    s = normalize_unicode(text)
    if not s:
        return ""
    if tone:
        s = normalize_tone(s)
    if abbrev:
        s = expand_abbreviations(s)
    if mask:
        s = mask_salary(s)
    if do_segment:
        s = segment(s)
    if drop_stopwords:
        s = remove_stopwords(s)
    return s
