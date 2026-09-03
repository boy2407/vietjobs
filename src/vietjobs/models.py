"""Model registry — the five required algorithms, plus the baselines they must beat.

Classification : KNN, SVM (LinearSVC), LogisticRegression   <- the required three
                 + for the algorithm-comparison axis:
                   RandomForest, ExtraTrees, LightGBM, XGBoost   (trees)
                   ComplementNB, SGDClassifier, NearestCentroid  (classical text)
Regression     : LinearRegression, LightGBM

Baselines are not optional decoration. A text model that cannot beat
"predict the median salary of this job category" has not learned anything from
the text, and reporting it without that comparison hides the fact.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    Ridge,
    SGDClassifier,
)
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight

from . import config as C


# ---------------------------------------------------------------------------
# Baselines that read the dataframe directly, bypassing the feature matrix
# ---------------------------------------------------------------------------


class GroupMedianRegressor(BaseEstimator, RegressorMixin):
    """Predict the training median of the group a row belongs to.

    The honest floor for salary: "look up what this kind of job usually pays".
    """

    def __init__(self, group_column: str = "category"):
        self.group_column = group_column

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        g = np.asarray(X[self.group_column])
        self.global_median_ = float(np.median(y))
        self.medians_ = {k: float(np.median(y[g == k])) for k in np.unique(g)}
        return self

    def predict(self, X):
        return np.array(
            [self.medians_.get(k, self.global_median_) for k in X[self.group_column]],
            dtype=float,
        )


class TitleKeywordClassifier(BaseEstimator, ClassifierMixin):
    """Assign the category whose training titles share the most words.

    A deliberately naive "no machine learning" reference point: it shows how far
    plain keyword overlap on the job title alone already gets you, which is the
    number a TF-IDF model has to justify itself against.
    """

    def fit(self, X, y):
        from collections import Counter, defaultdict

        y = np.asarray(y)
        self.classes_ = np.unique(y)
        counts: dict[str, Counter] = defaultdict(Counter)
        for title, label in zip(X["job_title"], y):
            counts[label].update(set(title.lower().split()))
        # Score a word for a class by how concentrated it is in that class.
        totals: Counter = Counter()
        for c in counts:
            totals.update(counts[c])
        self.weights_ = {
            c: {w: n / totals[w] for w, n in counts[c].items() if totals[w] >= 3}
            for c in counts
        }
        self.fallback_ = Counter(y).most_common(1)[0][0]
        return self

    def predict(self, X):
        out = []
        for title in X["job_title"]:
            words = set(title.lower().split())
            best, best_score = self.fallback_, 0.0
            for c, w in self.weights_.items():
                s = sum(w.get(t, 0.0) for t in words)
                if s > best_score:
                    best, best_score = c, s
            out.append(best)
        return np.array(out)


DATAFRAME_MODELS = {
    "category_median": lambda seed: GroupMedianRegressor("category"),
    "province_median": lambda seed: GroupMedianRegressor("province"),
    "experience_median": lambda seed: GroupMedianRegressor("experience_raw"),
    "keyword": lambda seed: TitleKeywordClassifier(),
}


# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------


def _knn(seed: int, k: int = 30) -> KNeighborsClassifier:
    """Cosine KNN on the sparse TF-IDF matrix.

    ``algorithm="brute"`` is not a tuning choice — KD-tree and Ball-tree cannot
    index a high-dimensional sparse matrix at all. Measured on this corpus,
    running KNN through TruncatedSVD(300) made it *worse* (macro-F1 0.520 vs
    0.546) and no faster, because 300 components keep only 68.8% of variance.

    KNN has no class_weight, so with 27:1 imbalance it leans hard toward the
    large classes. distance weighting softens that but does not fix it; the
    limitation belongs in the report rather than being hidden.
    """
    return KNeighborsClassifier(
        n_neighbors=k, metric="cosine", algorithm="brute", weights="distance", n_jobs=-1
    )


def _random_forest(seed: int, n_estimators: int = 300) -> RandomForestClassifier:
    """Random forest straight on the sparse TF-IDF matrix.

    ``max_features="sqrt"`` means each split considers only sqrt(236_596) ~= 486
    of the columns. On a matrix that is 99.79% zero, most of those 486 draws are
    all-zero columns that cannot split anything — that is the structural reason
    a forest is expected to struggle here, and the reason this run exists is to
    measure the size of that effect rather than assert it.

    ``class_weight="balanced_subsample"`` rather than "balanced": the weights are
    recomputed per bootstrap sample, which is the correct analogue of the
    ``class_weight="balanced"`` the linear models get at 27:1 imbalance.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )


def _lgbm_classifier(seed: int, n_estimators: int = 400, learning_rate: float = 0.1):
    """LightGBM multiclass on the sparse matrix.

    n_estimators is 400, not the 2000 the salary regressor uses: 16 classes means
    one tree per class per round, so 400 rounds is already 6400 trees. Raising it
    is a later sweep, not a default.

    ``colsample_bytree=0.3`` because with p/n = 7.1 most columns are noise; giving
    every tree the full column set mostly gives it more ways to memorise.
    """
    import lightgbm as lgb

    return lgb.LGBMClassifier(
        objective="multiclass",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=63,
        min_child_samples=20,
        colsample_bytree=0.3,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )


class BalancedXGBClassifier(BaseEstimator, ClassifierMixin):
    """XGBoost multiclass, with the two things it does not do on its own.

    1. XGBoost >= 2.0 refuses string labels — it wants 0..n-1. The 16 job
       categories are strings, so encode on fit and decode on predict.
    2. XGBoost has no ``class_weight``. Without it the 27:1 imbalance would make
       the comparison against SVM/LogReg/RandomForest unfair, so feed the same
       balanced weights through ``sample_weight``.

    Every tuneable knob is declared on ``__init__`` — sklearn's ``get_params``
    reads the signature, so anything hidden inside ``fit`` is unreachable from
    ``set_params`` and therefore unreachable from a sweep. ``colsample_bytree``
    in particular is this model's strongest cost lever; leaving it buried would
    have made six "configurations" collapse into six near-identical runs.
    Defaults reproduce the original hard-coded values, so ``xgb`` and ``xgb100``
    keep the meaning their existing log rows were written under.
    """

    def __init__(self, seed: int = 0, n_estimators: int = 400,
                 learning_rate: float = 0.1, max_depth: int = 8,
                 colsample_bytree: float = 0.3, subsample: float = 1.0,
                 min_child_weight: float = 1.0, reg_lambda: float = 1.0,
                 max_bin: int = 256):
        self.seed = seed
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.colsample_bytree = colsample_bytree
        self.subsample = subsample
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.max_bin = max_bin

    def fit(self, X, y):
        import xgboost as xgb

        self.encoder_ = LabelEncoder().fit(y)
        self.classes_ = self.encoder_.classes_
        self.model_ = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=len(self.classes_),
            tree_method="hist",
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            colsample_bytree=self.colsample_bytree,
            subsample=self.subsample,
            min_child_weight=self.min_child_weight,
            reg_lambda=self.reg_lambda,
            max_bin=self.max_bin,
            random_state=self.seed,
            n_jobs=-1,
        )
        self.model_.fit(
            X, self.encoder_.transform(y),
            sample_weight=compute_sample_weight("balanced", y),
        )
        return self

    def predict(self, X):
        return self.encoder_.inverse_transform(self.model_.predict(X))

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


CLASSIFIERS = {
    # --- baselines ---
    "majority": lambda seed: DummyClassifier(strategy="most_frequent"),
    "stratified": lambda seed: DummyClassifier(strategy="stratified", random_state=seed),
    # --- the three required algorithms ---
    "knn": lambda seed: _knn(seed, k=30),
    "knn5": lambda seed: _knn(seed, k=5),
    "knn15": lambda seed: _knn(seed, k=15),
    "knn50": lambda seed: _knn(seed, k=50),
    "svm": lambda seed: LinearSVC(C=0.5, class_weight="balanced", random_state=seed, max_iter=4000),
    "svm_plain": lambda seed: LinearSVC(C=0.5, random_state=seed, max_iter=4000),
    "logreg": lambda seed: LogisticRegression(
        C=4.0, class_weight="balanced", max_iter=2000, n_jobs=-1, random_state=seed
    ),
    # --- classical text baselines the first sweep left out ---
    # Naive Bayes is THE textbook baseline for TF-IDF text classification, and
    # its absence was the most conspicuous gap in the comparison. ComplementNB
    # rather than MultinomialNB: the complement formulation was designed for
    # imbalanced corpora, and this one is 27:1.
    #
    # Caveat that belongs in the report, not hidden here: NB assumes features
    # are counts, while these are L2-normalised TF-IDF weights. It runs, it is
    # what everyone does, and the assumption is still violated. That is a real
    # weakness of the model on this data, not a detail.
    #
    # ComplementNB has no class_weight — same limitation as knn and centroid.
    "nb": lambda seed: ComplementNB(alpha=1.0),
    # Same linear family as svm/logreg but fitted one sample at a time. Cheap
    # enough to answer: is LinearSVC paying for an exact solution it does not
    # need? Unlike the other two it does support class_weight.
    "sgd": lambda seed: SGDClassifier(
        loss="hinge", alpha=1e-4, class_weight="balanced",
        max_iter=2000, tol=1e-4, random_state=seed, n_jobs=-1,
    ),
    # Randomised split thresholds instead of searched ones. On a matrix that is
    # 99.79% zero, most candidate splits are uninformative anyway, so the
    # question is whether searching them was ever worth the cost.
    "extra": lambda seed: ExtraTreesClassifier(
        n_estimators=300, max_features="sqrt", min_samples_leaf=2,
        class_weight="balanced_subsample", random_state=seed, n_jobs=-1,
    ),
    # Rocchio: one centroid per class, predict by nearest centroid. The control
    # for knn — also distance-based, but comparing against 16 averaged vectors
    # instead of 33.396 individual ones, so it does not fail the same way.
    #
    # scikit-learn 1.9 restricts metric to {euclidean, manhattan}; cosine is not
    # available. On L2-normalised rows euclidean distance is a monotone function
    # of cosine distance for the PAIRWISE case, but a centroid is not itself
    # normalised, so the equivalence does not carry. This handicaps the model on
    # text, and the report must say so rather than presenting the score bare.
    "centroid": lambda seed: NearestCentroid(metric="euclidean"),
    # --- tree ensembles, for the algorithm-comparison axis ---
    # These are NOT in the required set. They are here to turn "linear beats
    # trees on TF-IDF" from a convention into a measured row.
    "rf": lambda seed: _random_forest(seed),
    "rf100": lambda seed: _random_forest(seed, n_estimators=100),
    "lgbm": lambda seed: _lgbm_classifier(seed),
    # Reduced-budget variants. Measured on scope=full: fit time is
    # ``119s + 23.8s * n_rounds`` on top of an 82s feature build, so the 400-round
    # default costs ~2.7h — against 27.8s for LinearSVC on the same matrix.
    #
    # lgbm100 cuts rounds 4x but raises the learning rate 3x to compensate:
    # ``learning_rate * n_estimators`` is the quantity that governs how much a
    # booster actually learns, and 0.3 * 100 = 30 stays comparable to the
    # 0.1 * 400 = 40 the xgb run uses. Cutting rounds while holding lr at 0.1
    # would leave the model underfitted, and its low score would then be
    # misread as "boosting is weak on TF-IDF" rather than "we stopped early".
    "lgbm100": lambda seed: _lgbm_classifier(seed, n_estimators=100, learning_rate=0.3),
    "xgb": lambda seed: BalancedXGBClassifier(seed=seed),
    # xgb at 400 rounds / max_depth=8 was killed on scope=full after 4h07m without
    # finishing. max_depth=8 means up to 256 leaves per tree — four times the 63
    # leaves LightGBM builds — which is why the LightGBM cost curve underestimated
    # it by more than half. xgb100 matches lgbm100 on both levers that drive cost
    # and learning: lr * rounds = 30, and a leaf budget in the same range.
    "xgb100": lambda seed: BalancedXGBClassifier(
        seed=seed, n_estimators=100, learning_rate=0.3, max_depth=6
    ),
    # Probability-calibrated SVM, for when the system needs confidences.
    "svm_calibrated": lambda seed: CalibratedClassifierCV(
        LinearSVC(C=0.5, class_weight="balanced", random_state=seed, max_iter=4000),
        method="sigmoid", cv=3,
    ),
}


def svm_with_C(value: float, seed: int) -> LinearSVC:
    return LinearSVC(C=value, class_weight="balanced", random_state=seed, max_iter=4000)


def logreg_with_C(value: float, seed: int) -> LogisticRegression:
    return LogisticRegression(
        C=value, class_weight="balanced", max_iter=2000, n_jobs=-1, random_state=seed
    )


# ---------------------------------------------------------------------------
# Regressors
# ---------------------------------------------------------------------------


def _lightgbm(seed: int, objective: str = "regression_l1", alpha: float | None = None):
    """LightGBM straight on the sparse matrix.

    Deliberately NOT wrapped in TruncatedSVD: SVD destroys the sparsity that
    makes TF-IDF informative, and LightGBM consumes scipy.sparse natively.
    """
    import lightgbm as lgb

    params = dict(
        objective=objective,
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.6,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    if alpha is not None:
        params.update(objective="quantile", alpha=alpha)
    return lgb.LGBMRegressor(**params)


REGRESSORS = {
    # --- baselines ---
    "median": lambda seed: DummyRegressor(strategy="median"),
    "mean": lambda seed: DummyRegressor(strategy="mean"),
    # --- the two required algorithms ---
    # Plain OLS. On full sparse text this collapses (measured: R2 val 0.081,
    # MAE 6.00 triệu — worse than the category-median baseline at 5.54). Kept
    # so the report can show that, and so L-a/L-b can show it working properly
    # once the feature space is well-posed.
    "linreg": lambda seed: LinearRegression(),
    # Same model family, with L2. This is the one that actually ships.
    "ridge": lambda seed: Ridge(alpha=1.0, solver="sparse_cg", random_state=seed),
    "lgbm": lambda seed: _lightgbm(seed),
    "lgbm_q10": lambda seed: _lightgbm(seed, alpha=0.1),
    "lgbm_q90": lambda seed: _lightgbm(seed, alpha=0.9),
}


# ---------------------------------------------------------------------------


def build_estimator(task: str, name: str, seed: int):
    """Return ``(estimator, reads_dataframe)``.

    ``reads_dataframe=True`` means the estimator is fitted on the raw frame and
    must not be put behind the feature matrix.
    """
    if name in DATAFRAME_MODELS:
        return DATAFRAME_MODELS[name](seed), True
    registry = REGRESSORS if task == C.TASK_SALARY else CLASSIFIERS
    if name not in registry:
        raise SystemExit(
            f"unknown model {name!r} for task {task!r}.\n"
            f"  choices: {', '.join(sorted(registry) + sorted(DATAFRAME_MODELS))}"
        )
    return registry[name](seed), False
