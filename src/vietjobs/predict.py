"""Inference: a raw job posting in, a category (and later a salary) out.

    from vietjobs.predict import JobPostingModel
    m = JobPostingModel.load(category_run="cat-svm-full-segment+province-...")
    m.predict({"job_title": "Nhân Viên Kinh Doanh Bất Động Sản",
               "description": "Tìm kiếm khách hàng, tư vấn căn hộ..."})

The caller passes RAW fields, exactly as they appear on a job board. Everything
in between — Vietnamese normalisation, abbreviation expansion, segmentation,
the derived numeric columns — happens here, using the same code path that built
the training data. That identity is the whole point of this module: a
preprocessing mismatch between training and serving is silent and produces
quietly degraded predictions.

CLI:
    python -m vietjobs.predict --title "..." --description "..."
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config as C
from . import vitext as V

# The raw fields a caller may supply. Anything omitted defaults to empty, which
# the pipeline handles — a title-only posting is a legitimate input.
RAW_FIELDS = (
    "job_title",
    "description",
    "requirements_text",
    "qualifications",
    "technical_skills",
    "soft_skills",
    "benefits",
    "location",
    "country",
    "languages_required",
    "experience_required",
    "contract_type",
    "working_hours",
)


def _latest_run(prefix: str) -> str:
    """Most recently modified artifact directory starting with ``prefix``."""
    if not C.ARTIFACT_DIR.exists():
        raise FileNotFoundError(f"{C.ARTIFACT_DIR} does not exist — train a model first")
    runs = [d for d in C.ARTIFACT_DIR.iterdir() if d.is_dir() and d.name.startswith(prefix)]
    if not runs:
        raise FileNotFoundError(
            f"no run under artifacts/ starting with {prefix!r}. "
            f"Train one: python -m vietjobs.train --task category --model svm"
        )
    return max(runs, key=lambda d: d.stat().st_mtime).name


def build_frame(posting: dict | list[dict]) -> pd.DataFrame:
    """Raw posting dict(s) -> the exact column layout the models were fitted on.

    Mirrors ``dataset.clean`` for a handful of rows. Kept deliberately close to
    it; ``tests/test_predict.py`` asserts the two agree column-for-column.
    """
    rows = [posting] if isinstance(posting, dict) else list(posting)
    raw = pd.DataFrame(
        [{f: str(r.get(f, "") or "") for f in RAW_FIELDS} for r in rows]
    )

    out = pd.DataFrame(index=raw.index)

    base = lambda c: raw[c].map(lambda s: V.preprocess(s, mask=False))     # noqa: E731
    masked = lambda c: raw[c].map(lambda s: V.preprocess(s, mask=True))    # noqa: E731

    out["job_title"] = base("job_title")
    out["description"] = base("description")
    out["requirements_text"] = base("requirements_text")
    out["job_title_masked"] = masked("job_title")
    out["description_masked"] = masked("description")
    out["requirements_masked"] = masked("requirements_text")

    for src in C.LIST_COLUMNS:
        out[f"{src}_text"] = raw[src].map(V.join_list_field)
        out[f"n_{src}"] = raw[src].map(lambda v: len(V.parse_list_field(v)))
    out["benefits_masked"] = out["benefits_text"].map(V.mask_salary)

    locs = raw["location"].map(V.split_locations)
    out["location_raw"] = locs.map(lambda xs: xs[0] if xs else "unknown")
    out["province"] = raw["location"].map(V.normalize_province)
    out["n_locations"] = locs.map(len)
    out["is_major_city"] = out["province"].isin(C.MAJOR_CITIES).astype(int)
    out["country"] = raw["country"].map(V.normalize_unicode)

    langs = raw["languages_required"].map(V.split_languages)
    out["languages_text"] = langs.map(" ".join)
    out["n_languages"] = langs.map(len)
    out["requires_english"] = langs.map(lambda xs: int(any("anh" in x for x in xs)))

    out["experience_months"] = raw["experience_required"].map(V.experience_to_months)
    out["experience_raw"] = raw["experience_required"].map(V.normalize_unicode)
    out["contract_type"] = raw["contract_type"].map(V.normalize_unicode).replace("", "unknown")
    out["working_hours"] = raw["working_hours"].map(V.normalize_unicode)
    out["has_working_hours"] = (out["working_hours"] != "").astype(int)

    out["desc_len"] = out["description"].str.len()
    out["req_len"] = out["requirements_text"].str.len()
    out["title_len"] = out["job_title"].str.len()
    out["n_acronyms"] = raw["job_title"].map(
        lambda s: sum(1 for t in V.normalize_unicode(s).split() if len(t) > 1 and t.isupper())
    )

    # Segmented mirrors. Serving one posting at a time, so no parallelism.
    for src, dst in [
        ("job_title", "job_title_seg"),
        ("description", "description_seg"),
        ("requirements_text", "requirements_seg"),
        ("job_title_masked", "job_title_masked_seg"),
        ("description_masked", "description_masked_seg"),
        ("requirements_masked", "requirements_masked_seg"),
        ("benefits_masked", "benefits_masked_seg"),
        ("technical_skills_text", "technical_skills_seg"),
        ("qualifications_text", "qualifications_seg"),
    ]:
        out[dst] = out[src].map(V.segment)

    return out


class JobPostingModel:
    """Loads the trained models and answers with category (+ salary when given)."""

    def __init__(self, category=None, disclosed=None, salary=None, meta=None):
        self.category = category
        self.disclosed = disclosed
        self.salary = salary
        self.meta = meta or {}

    # -- loading -------------------------------------------------------------

    @classmethod
    def load(
        cls,
        category_run: str | None = None,
        disclosed_run: str | None = None,
        salary_run: str | None = None,
    ) -> "JobPostingModel":
        """Load by run id. ``None`` for category picks the newest ``cat-*`` run."""

        def _load(run: str | None, prefix: str, required: bool):
            if run is None and required:
                run = _latest_run(prefix)
            if run is None:
                return None, None
            path: Path = C.ARTIFACT_DIR / run / "model.joblib"
            if not path.exists():
                raise FileNotFoundError(path)
            metrics = json.loads((C.ARTIFACT_DIR / run / "metrics.json").read_text("utf-8"))
            return joblib.load(path), {"run_id": run, "metrics": metrics["metrics"]}

        cat, cat_meta = _load(category_run, "cat-", required=True)
        dis, dis_meta = _load(disclosed_run, "dis-", required=False)
        sal, sal_meta = _load(salary_run, "sal-", required=False)
        return cls(cat, dis, sal, {"category": cat_meta, "disclosed": dis_meta, "salary": sal_meta})

    # -- prediction ----------------------------------------------------------

    def predict(self, posting: dict, top_k: int = 3) -> dict:
        """One posting in, one result dict out."""
        return self.predict_batch([posting], top_k=top_k)[0]

    def predict_batch(self, postings: list[dict], top_k: int = 3) -> list[dict]:
        frame = build_frame(postings)
        n = len(frame)
        results: list[dict] = [{"warnings": []} for _ in range(n)]

        # --- category -------------------------------------------------------
        labels = np.asarray(self.category.classes_)
        if hasattr(self.category, "predict_proba"):
            proba = self.category.predict_proba(frame)
        elif hasattr(self.category, "decision_function"):
            # LinearSVC gives margins, not probabilities. Softmax turns them
            # into a ranking that is comparable within one posting; it is NOT a
            # calibrated probability, and is labelled "score" for that reason.
            margins = np.atleast_2d(self.category.decision_function(frame))
            e = np.exp(margins - margins.max(axis=1, keepdims=True))
            proba = e / e.sum(axis=1, keepdims=True)
        else:
            proba = None

        preds = self.category.predict(frame)
        calibrated = hasattr(self.category, "predict_proba")

        for i in range(n):
            r = results[i]
            r["category"] = str(preds[i])
            if proba is not None:
                order = np.argsort(proba[i])[::-1][:top_k]
                key = "confidence" if calibrated else "score"
                r[f"category_{key}"] = round(float(proba[i][order[0]]), 4)
                r["category_topk"] = [
                    {"category": str(labels[j]), key: round(float(proba[i][j]), 4)}
                    for j in order
                ]
                if proba[i][order[0]] < 0.35:
                    r["warnings"].append(
                        "mô tả không đủ đặc trưng — hãy xem thêm các lựa chọn trong category_topk"
                    )
            if not frame["description"].iloc[i] and len(frame["job_title"].iloc[i]) < 8:
                r["warnings"].append("đầu vào quá ngắn, dự đoán kém tin cậy")

        # --- salary (only if those models were loaded) ----------------------
        if self.salary is not None:
            pred_log = self.salary.predict(frame)
            est = np.clip(np.expm1(pred_log), 0.5, None)
            for i in range(n):
                results[i]["salary_estimate_trieu"] = round(float(est[i]), 1)
        if self.disclosed is not None:
            if hasattr(self.disclosed, "predict_proba"):
                p = self.disclosed.predict_proba(frame)[:, 1]
                for i in range(n):
                    results[i]["salary_disclosed_prob"] = round(float(p[i]), 3)
                    if p[i] < 0.5:
                        results[i]["warnings"].append(
                            "tin đăng kiểu 'Thoả thuận' — ước lượng lương độ tin cậy thấp"
                        )
        return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict job category from a posting.")
    ap.add_argument("--title", required=True)
    ap.add_argument("--description", default="")
    ap.add_argument("--requirements", default="")
    ap.add_argument("--location", default="")
    ap.add_argument("--experience", default="")
    ap.add_argument("--category-run", default=None)
    ap.add_argument("--salary-run", default=None)
    ap.add_argument("--disclosed-run", default=None)
    args = ap.parse_args()

    model = JobPostingModel.load(args.category_run, args.disclosed_run, args.salary_run)
    out = model.predict(
        {
            "job_title": args.title,
            "description": args.description,
            "requirements_text": args.requirements,
            "location": args.location,
            "experience_required": args.experience,
        }
    )
    print(f"\nmodel: {model.meta['category']['run_id']}")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
