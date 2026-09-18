"""VietJobs-37K as an external set: parsing, crosswalk, overlap, conventions."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from vietjobs import external as X
from vietjobs import vitext as V


# --- parse -----------------------------------------------------------------


def test_parse_full_record_reorders_fields():
    r = X.parse_37k_text("[TITLE] Kế toán [REQ] 2 năm [DESC] Lập chứng từ")
    assert r == {"job_title": "Kế toán", "description": "Lập chứng từ",
                 "requirements_text": "2 năm"}


def test_parse_nan_placeholder_becomes_empty():
    r = X.parse_37k_text("[TITLE] Kế toán [REQ] nan [DESC] Mô tả")
    assert r["requirements_text"] == ""
    assert r["description"] == "Mô tả"


def test_parse_missing_desc_and_no_markers():
    assert X.parse_37k_text("[TITLE] A [REQ] B")["description"] == ""
    assert X.parse_37k_text("Chỉ có tiêu đề")["job_title"] == "Chỉ có tiêu đề"
    assert X.parse_37k_text(None) == {"job_title": "", "description": "",
                                      "requirements_text": ""}


def test_load_37k_roundtrip(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text(json.dumps({"id": 7, "text": "[TITLE] A [REQ] B [DESC] C",
                             "labels": ["L"], "label_ids": [3], "num_labels": 1,
                             "source": "topcv", "confidence": 0.9},
                            ensure_ascii=False) + "\n", encoding="utf-8")
    df = X.load_37k(p)
    assert list(df.columns) == ["id", "job_title", "description", "requirements_text",
                                "labels", "label_ids", "num_labels", "source", "confidence"]
    assert df.loc[0, "label_ids"] == [3]


# --- crosswalk -------------------------------------------------------------

CW = {"_status": "draft", "map": {
    "0": {"target": "kinh_doanh"},
    "1": {"target": "kinh_doanh"},
    "2": {"target": "tai_chinh"},
    "3": {"target": None},
}}


def _frame(label_ids, num_labels=None):
    return pd.DataFrame({
        "label_ids": label_ids,
        "num_labels": num_labels or [len(x) for x in label_ids],
    })


def test_crosswalk_null_targets_are_dropped_and_counted():
    df, rep = X.apply_crosswalk(_frame([[3], [0, 3], [0, 1]]), CW)
    assert df["targets"].tolist() == [[], ["kinh_doanh"], ["kinh_doanh"]]
    assert rep["unscorable_all_null"] == 1
    assert rep["multi_target"] == 0
    assert rep["crosswalk_status"] == "draft"


def test_strict_requires_single_external_label():
    df, rep = X.apply_crosswalk(_frame([[0], [0, 1], [0, 2]]), CW)
    # [0,1] both map to kinh_doanh → one target, but two external labels → not strict
    assert df["strict_target"].tolist() == ["kinh_doanh", None, None]
    assert df["n_targets"].tolist() == [1, 1, 2]
    assert rep["strict_eligible"] == 1
    assert rep["multi_target"] == 1


def test_crosswalk_unknown_label_raises():
    with pytest.raises(KeyError):
        X.apply_crosswalk(_frame([[99]]), CW)


def test_shipped_crosswalk_covers_all_60_and_only_our_classes():
    if not X.CROSSWALK_PATH.exists():
        pytest.skip("external data not on this machine")
    cw = X.load_crosswalk()
    ids = set(X.crosswalk_targets(cw))
    assert ids == set(range(60))
    ours = {
        "kinh_doanh_bán_hàng_chăm_sóc_khách_hàng", "sản_xuất_lao_động_phổ_thông_cơ_khí",
        "marketing_truyền_thông_quảng_cáo_nội_dung", "tài_chính_kế_toán_ngân_hàng_bảo_hiểm",
        "du_lịch_nhà_hàng_khách_sạn_dịch_vụ", "thiết_kế_nghệ_thuật_giải_trí_truyền_hình_báo_chí",
        "nhân_sự_hành_chính_pháp_chế_tư_vấn", "xây_dựng_kiến_trúc_bất_động_sản",
        "logistics_vận_tải_chuỗi_cung_ứng", "công_nghệ_thông_tin_kỹ_thuật_số",
        "kỹ_thuật_điện_điện_tử_viễn_thông", "y_tế_dược_chăm_sóc_sức_khỏe_công_nghệ_sinh_học",
        "giáo_dục_đào_tạo_nghiên_cứu", "ngôn_ngữ_dịch_thuật", "nhóm_nghề_khác",
        "nông_nghiệp_năng_lượng_môi_trường",
    }
    targets = {t for t in X.crosswalk_targets(cw).values() if t is not None}
    assert targets <= ours, targets - ours


# --- overlap ---------------------------------------------------------------


def _index():
    train = pd.DataFrame({
        "job_title": ["Nhân viên kế toán", "Kỹ sư xây dựng"],
        "description": ["Lập chứng từ.", "Giám sát công trình dân dụng và hạ tầng kỹ thuật ở Hà Nội"],
        "requirements_text": ["2 năm", "5 năm"],
    })
    train["group_id"] = [V.group_key(t, d, r) for t, d, r in
                         zip(train["job_title"], train["description"], train["requirements_text"])]
    dev = pd.DataFrame({"job_title": [], "description": [], "requirements_text": [], "group_id": []})
    return X.OurIndex(split_names=()).add("train", train).add("dev", dev)


def test_overlap_exact_hash_catches_cosmetic_repost():
    ext = pd.DataFrame({
        "job_title": ["NHÂN VIÊN KẾ TOÁN", "Lập trình viên"],
        "description": ["Lập chứng từ", "Viết code"],
        "requirements_text": ["2 năm", "3 năm"],
    })
    mask, rep = X.overlap_mask(ext, _index())
    assert mask.tolist() == [True, False]
    assert rep["exact_hash"] == 1 and rep["overlap_train"] == 1 and rep["overlap_dev"] == 0


def test_overlap_fuzzy_survives_deidentification_and_reformatting():
    # same title, description with [COMPANY] inserted and the requirements moved
    ext = pd.DataFrame({
        "job_title": ["Kỹ sư xây dựng"],
        "description": ["[COMPANY] Giám sát công trình dân dụng và hạ tầng kỹ thuật ở Hà Nội"],
        "requirements_text": [""],
    })
    mask, rep = X.overlap_mask(ext, _index())
    assert mask.tolist() == [True]
    assert rep["exact_hash"] == 0 and rep["fuzzy_title_and_jaccard"] == 1


def test_overlap_same_title_different_description_is_not_a_repost():
    ext = pd.DataFrame({
        "job_title": ["Kỹ sư xây dựng"],
        "description": ["Thiết kế kết cấu thép cho nhà xưởng"],
        "requirements_text": [""],
    })
    mask, rep = X.overlap_mask(ext, _index())
    assert mask.tolist() == [False]
    assert rep["same_title_any_jaccard"] == 1 and rep["overlap_total"] == 0


# --- conventions -----------------------------------------------------------


def test_lenient_truth_credits_any_target():
    y = X.lenient_truth(["a", "b", "c"], [["a", "z"], ["z", "b"], ["x", "y"]])
    assert y == ["a", "b", "x"]


def test_lenient_truth_refuses_empty_targets():
    with pytest.raises(ValueError):
        X.lenient_truth(["a"], [[]])


# --- salary read off the text ------------------------------------------------


@pytest.mark.parametrize("text, expected", [
    ("Lương: Từ 9.000.000 đến 25.000.000", (9.0, 25.0)),
    ("Thu nhập: 15 triệu", (15.0, 15.0)),
    ("Lương cứng: 8,000,000", (8.0, 8.0)),
    ("Thu nhập 7 - 7,5 triệu", (7.0, 7.5)),
    ("lương: 8 15 triệu", (8.0, 15.0)),
    # The chain stops at the bonus: 50tr/năm is not monthly pay.
    ("Lương cơ bản 8.000.000đ, thưởng 50.000.000/năm", (8.0, 8.0)),
    ("Junior: Lương 9 - 12 triệu. Senior: Lương 20 triệu", (9.0, 20.0)),
    # A ladder states the pay word once; the later rungs must still be read.
    ("Lương:- Junior: Từ 10.000.000 đến 25.000.000, theo năng suất- "
     "Mid-Level: Từ 12.000.000 đến 30.000.000- Professional: Từ 15.000.000 đến 35.000.000",
     (10.0, 35.0)),
])
def test_extract_salary_reads_monthly_pay(text, expected):
    assert X.extract_salary(text)[:2] == expected


@pytest.mark.parametrize("text", [
    "Lương: thỏa thuận",
    "tính lương, chấm công",
    "40 triệu người dùng",
    "Lương 50.000đ/giờ",
    "Lương 1000 USD",
    "Lương 25.000",
    "Thu nhập năm từ 200 triệu",
    "lương trong năm, hỗ trợ nghỉ dưỡng lên đến 9 triệu",
    None,
])
def test_extract_salary_refuses_unreadable(text):
    assert X.extract_salary(text) is None


# --- T9.2: one accept + one refuse per widening rule ------------------------


@pytest.mark.parametrize("text, expected", [
    # (a) a section header anchors the figure — the ad never says "lương".
    ("3. Quyền lợi:- 3.000.000 - 16.000.000 VNĐ, theo năng suất lao động", (3.0, 16.0)),
    # (b) "M" is triệu in ad shorthand.
    ("Thu nhập 30-40M bao gồm lương cứng", (30.0, 40.0)),
    # (c) two figures written out in dong are a range even with no dash.
    ("Lương cơ bản 10.000.000 13.000.000", (10.0, 13.0)),
    # (d) no pay word at all: only an explicit range closed by a unit.
    ("Kế Toán Trưởng (45 - 60 Triệu) Đi Làm Ngay", (45.0, 60.0)),
    # The period word refuses only when it is the toned spelling: "Nam" here is
    # the gender, so the pay stays readable.
    ("NHÂN VIÊN THU MUA - THU NHẬP 10tr Nam Nữ: trên 25 tuổi", (10.0, 10.0)),
])
def test_extract_salary_round_two_reads(text, expected):
    assert X.extract_salary(text)[:2] == expected


@pytest.mark.parametrize("text", [
    # (a) the header is there, but the figure it introduces is a meal allowance.
    "Quyền lợi- Phụ cấp ăn trưa: 40.000 vnđ",
    # (b) an annual benefit package is not monthly pay ("/" lost to de-id).
    "Gói Benefit 18M năm",
    # (c) small bare figures stay noise: "lương tháng 13 14" is not 13–14 triệu.
    "Lương tháng 13 14 Đầy đủ chế độ BHXH",
    # (d) a project budget is money, not pay.
    "Tìm agency, ngân sách trong khoảng 100 - 200 triệu tháng",
    # Annual pay, stated as a range.
    "Thu nhập: 130-160tr năm Phụ cấp: Ăn ca",
])
def test_extract_salary_round_two_refuses(text):
    assert X.extract_salary(text) is None


@pytest.mark.parametrize("text, note", [
    ("Lương: 10 - 15 triệu", ""),
    ("Lương tháng 13, thưởng lễ tết", "figure_unread"),
    ("Lương: thỏa thuận", "negotiable"),
    ("Tuyển nhân viên kinh doanh", "no_figure"),
])
def test_salary_note_explains_every_empty_row(text, note):
    """T9.3 — a row without a salary always says why."""
    assert X.salary_note(text) == note
    assert (X.extract_salary(text) is None) == bool(note)


def test_template_ids_group_reposted_ads():
    """T9.4 — same body, different branch title → one group.

    The two first rows are the shape of the corpus's largest template: an empty
    description, the ad repeated in the requirements.
    """
    desc = pd.Series(["", "", "Một mô tả khác hẳn", "", ""])
    req = pd.Series([
        "Quyền lợi- 3.000.000 - 16.000.000 VNĐ",   # Hoài Nhơn branch
        "Quyền lợi- 3.000.000 - 16.000.000 VNĐ",   # Bình Định branch
        "Yêu cầu khác",
        "",                                         # blank groups with nothing
        "",
    ])
    ids = X.template_ids(desc, req)
    assert ids[0] == ids[1]
    assert ids[2] not in (ids[0], ids[3], ids[4])
    assert ids[3] != ids[4]


def test_audit_buckets_are_exhaustive():
    """Every posting lands in exactly one declared bucket (T9.1)."""
    samples = [
        "Lương: 10 - 15 triệu",                      # read
        "Lương tháng 13, thưởng lễ tết",             # a figure, refused
        "Quyền lợi- Phụ cấp ăn trưa: 40.000 vnđ",    # a figure, no pay word
        "Lương: thỏa thuận",                         # negotiable
        "Tuyển nhân viên kinh doanh",                # no figure
        "",
    ]
    buckets = [X.audit_salary(s) for s in samples]
    assert all(b in X.BUCKETS for b in buckets)
    assert buckets[0] == "read"
    assert buckets[3] == "negotiable"
    assert buckets[4] == buckets[5] == "no_figure"
