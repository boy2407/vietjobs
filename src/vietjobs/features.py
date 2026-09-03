"""Feature extraction — one sparse ColumnTransformer per (task, preprocessing).

Two things this module is responsible for:

1. **Choosing which text mirror a task may read.** The classification task reads
   the posting verbatim. The salary tasks read the ``*_masked`` mirrors, where
   pay figures have been replaced by ``<SALARY>``. Getting this wrong does not
   raise — it silently produces a salary model that reads its own answer and
   reports excellent, meaningless numbers. ``tests/test_no_leak.py`` pins it.

2. **Making the Vietnamese preprocessing switchable**, so every step in
   ``vitext.py`` can be turned on and off and measured. "Does segmentation
   help?" is an experiment, not an assumption — see ``PrepConfig``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler, OneHotEncoder

from . import config as C
from . import vitext as V

# ---------------------------------------------------------------------------
# Which column a task is allowed to read
# ---------------------------------------------------------------------------
# Four mirrors exist for each free-text field, built once in dataset.py:
#
#   description              raw
#   description_seg          raw, word-segmented
#   description_masked       pay figures removed
#   description_masked_seg   pay figures removed, word-segmented
#
# The salary tasks are restricted to the masked pair. `resolve_column` is the
# only place that decision is made.

_MASKABLE = {
    "job_title": "job_title_masked",
    "description": "description_masked",
    "requirements_text": "requirements_masked",
    "benefits_text": "benefits_masked",
}

# Not every column has a segmented mirror — short list-ish fields like
# soft_skills are left alone because segmenting them buys nothing.
_SEG_NAME = {
    "job_title": "job_title_seg",
    "description": "description_seg",
    "requirements_text": "requirements_seg",
    "job_title_masked": "job_title_masked_seg",
    "description_masked": "description_masked_seg",
    "requirements_masked": "requirements_masked_seg",
    "benefits_masked": "benefits_masked_seg",
    "technical_skills_text": "technical_skills_seg",
    "qualifications_text": "qualifications_seg",
}


#: Columns a salary/disclosed model must never see. Used by the leak test.
#
# The segmented names come from ``_SEG_NAME``, not from ``f"{c}_seg"``. Building
# them by string concatenation produced ``requirements_text_seg`` while the real
# column is ``requirements_seg`` — a shield guarding a name that does not exist,
# which is exactly the silent failure this set was written to prevent.
# ``tests/test_no_leak.py`` now asserts every one of these names is real.
UNMASKED_COLUMNS = (
    frozenset(_MASKABLE)
    | {_SEG_NAME[c] for c in _MASKABLE if c in _SEG_NAME}
    | set(C.TARGET_LEAK_COLUMNS)
    | {"salary_mid", "salary_mid_log", "salary_is_range",
       "salary_disclosed", "salary_extreme"}
)


def resolve_column(name: str, *, task: str, segmented: bool) -> str:
    """Map a logical field to the actual dataframe column for this task.

    ``("description", task="salary", segmented=True)`` -> ``description_masked_seg``
    """
    col = _MASKABLE[name] if (task in C.MASKED_TASKS and name in _MASKABLE) else name
    if segmented:
        seg = _SEG_NAME.get(col)
        if seg is not None:
            return seg
    return col



# ---------------------------------------------------------------------------
# Preprocessing configuration — one object per ablation row
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PrepConfig:
    """Which Vietnamese preprocessing steps are switched on.

    Every field here is a row in the ablation table. The defaults are all False
    on purpose: the baseline is "do nothing special", and each step has to earn
    its place by moving the score.
    """

    segment: bool = False       # vitext 2.4 — word segmentation
    charfold: bool = False      # vitext 2.6 — accent-folded char n-gram channel
    province: bool = False      # vitext 2.7 — district -> province
    stopwords: bool = False     # vitext 2.5 — drop stopwords

    def tag(self) -> str:
        """Short label for run ids and the results log."""
        on = [k for k in ("segment", "charfold", "province", "stopwords") if getattr(self, k)]
        return "+".join(on) if on else "raw"


PREP_RAW = PrepConfig()
PREP_FULL = PrepConfig(segment=True, charfold=True, province=True)


# ---------------------------------------------------------------------------
# Vectorisers
# ---------------------------------------------------------------------------


def _word_tfidf(prep: PrepConfig, **kw) -> TfidfVectorizer:
    params = dict(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.6,
        sublinear_tf=True,
        # Vietnamese diacritics are meaning-bearing: má/mà/mả/mã/mạ are five
        # different words. Never strip them from the word channel.
        strip_accents=None,
        stop_words=sorted(V.stopwords()) if prep.stopwords else None,
    )
    params.update(kw)
    return TfidfVectorizer(**params)


def _char_tfidf(**kw) -> TfidfVectorizer:
    """Char n-grams over accent-FOLDED text.

    5.1% of titles are written without diacritics. This channel is what lets
    "Nhan Vien Kinh Doanh" match "Nhân Viên Kinh Doanh". It supplements the word
    channel; it never replaces it.
    """
    params = dict(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=3,
        sublinear_tf=True,
        max_features=120_000,
        preprocessor=V.fold_accents,
    )
    params.update(kw)
    return TfidfVectorizer(**params)


NUMERIC_COLUMNS = [
    "experience_months",
    "n_technical_skills",
    "n_soft_skills",
    "n_benefits",
    "n_qualifications",
    "n_locations",
    "n_languages",
    "n_acronyms",
    "requires_english",
    "is_major_city",
    "has_working_hours",
    "desc_len",
    "req_len",
    "title_len",
]


def categorical_columns(prep: PrepConfig) -> list[str]:
    """Location column depends on whether province normalisation is on."""
    return [
        "province" if prep.province else "location_raw",
        "contract_type",
        "experience_raw",
    ]


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def build_features(
    task: str,
    scope: str = "full",
    prep: PrepConfig = PREP_RAW,
    svd: int | None = None,
):
    """Assemble the feature matrix.

    task
        ``"category"`` reads raw text; ``"salary"``/``"disclosed"`` read masked.
    scope
        ``"structured"`` — numeric + one-hot only, no text (p ~ 60).
                           The well-posed setting for plain LinearRegression.
        ``"title"``      — title only. The cheap baseline.
        ``"full"``       — title + body + skills + structured fields.
    prep
        Which Vietnamese preprocessing steps are on.
    svd
        If set, project the sparse matrix to this many dense dimensions.
        Used for the LinearRegression L-b setting and the KNN dimensionality ablation.
    """
    col = lambda name: resolve_column(name, task=task, segmented=prep.segment)  # noqa: E731

    blocks: list[tuple] = []

    if scope != "structured":
        blocks.append(("title_word", _word_tfidf(prep, max_features=60_000), col("job_title")))
        if prep.charfold:
            # Char channel reads the UNSEGMENTED mirror: underscores from the
            # segmenter would pollute character n-grams.
            raw_title = resolve_column("job_title", task=task, segmented=False)
            blocks.append(("title_char", _char_tfidf(max_features=60_000), raw_title))

    if scope == "full":
        blocks += [
            ("desc_word", _word_tfidf(prep, max_features=150_000), col("description")),
            ("req_word", _word_tfidf(prep, max_features=100_000), col("requirements_text")),
            ("tech_word", _word_tfidf(prep, max_features=40_000), col("technical_skills_text")),
            ("qual_word", _word_tfidf(prep, max_features=20_000), col("qualifications_text")),
            ("soft_word", _word_tfidf(prep, max_features=20_000), "soft_skills_text"),
            ("benefit_word", _word_tfidf(prep, max_features=30_000), col("benefits_text")),
            ("lang_word", _word_tfidf(prep, min_df=1, ngram_range=(1, 1)), "languages_text"),
        ]

    if scope in {"full", "structured"}:
        blocks += [
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", min_frequency=20),
                categorical_columns(prep),
            ),
            (
                "numeric",
                Pipeline(
                    [
                        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                        ("scale", MaxAbsScaler()),
                    ]
                ),
                NUMERIC_COLUMNS,
            ),
        ]

    if not blocks:
        raise ValueError(f"scope={scope!r} produced no feature blocks")

    ct = ColumnTransformer(
        blocks,
        remainder="drop",
        sparse_threshold=1.0,
        verbose_feature_names_out=True,
    )

    if svd:
        return Pipeline([("sparse", ct), ("svd", TruncatedSVD(n_components=svd, random_state=C.RANDOM_SEED))])
    return ct


def source_columns(transformer) -> list[str]:
    """Every dataframe column the transformer reads. Used by the leak test."""
    ct = transformer.named_steps["sparse"] if hasattr(transformer, "named_steps") else transformer
    out: list[str] = []
    for _name, _est, cols in ct.transformers:
        out.extend([cols] if isinstance(cols, str) else list(cols))
    return out
