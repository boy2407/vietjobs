"""Unit tests for the Vietnamese text pipeline.

These run before anything touches a model. Every step in vitext.py that the
ablation table depends on is pinned here, so a regression in preprocessing
shows up as a red test rather than as a mysteriously worse macro-F1.
"""
from __future__ import annotations

import pytest

from vietjobs import vitext as V


# --- 2.1 Unicode ------------------------------------------------------------


def test_normalize_unicode_collapses_whitespace_and_bullets():
    raw = "- Vận hành hằng ngày\r\n  * Kiểm tra   thiết bị\n\n• Báo cáo"
    assert V.normalize_unicode(raw) == "Vận hành hằng ngày Kiểm tra thiết bị Báo cáo"


def test_normalize_unicode_treats_sentinels_as_empty():
    for junk in ["", "  ", "nan", "NaN", "None", "UNK", "N/A", "-"]:
        assert V.normalize_unicode(junk) == ""


def test_normalize_unicode_unifies_composed_and_decomposed():
    composed = "ế"                    # ế as one code point
    decomposed = "ế"           # e + circumflex + acute
    assert V.normalize_unicode(composed) == V.normalize_unicode(decomposed)


# --- 2.2 Tone placement -----------------------------------------------------


@pytest.mark.parametrize(
    "old,new",
    [
        ("hòa", "hoà"),
        ("thúy", "thuý"),
        ("hòe", "hoè"),
        ("tòa nhà", "toà nhà"),
        ("thủy sản", "thuỷ sản"),
        ("khóa học", "khoá học"),
    ],
)
def test_normalize_tone_unifies_both_styles(old, new):
    assert V.normalize_tone(old) == V.normalize_tone(new) == new


def test_normalize_tone_preserves_qu_words():
    """'quý' must survive: the u in 'qu' is a glide, the tone belongs on the y.

    Normalising toward the first vowel would corrupt this into 'qúy'.
    """
    for word in ["quý", "quỳnh", "quỹ", "quyết", "quy trình"]:
        assert V.normalize_tone(word) == word


def test_normalize_tone_handles_case():
    assert V.normalize_tone("HÒA") == "HOÀ"
    assert V.normalize_tone("Hòa") == "Hoà"


# --- 2.3 Abbreviations ------------------------------------------------------


def test_expand_unambiguous_abbreviations():
    assert "nhân viên" in V.expand_abbreviations("NV kinh doanh")
    assert "bảo hiểm xã hội" in V.expand_abbreviations("Đóng BHXH đầy đủ")
    assert "khu công nghiệp" in V.expand_abbreviations("Làm việc tại KCN Tân Bình")


def test_tp_is_city_when_followed_by_a_place():
    assert "thành phố" in V.expand_abbreviations("Làm việc tại TP.HCM")
    assert "thành phố" in V.expand_abbreviations("Văn phòng TP Hà Nội")


def test_tp_is_department_head_when_followed_by_a_department():
    out = V.expand_abbreviations("Tuyển TP Kinh doanh")
    assert "trưởng phòng" in out
    assert "thành phố" not in out


def test_ambiguous_token_is_left_alone_when_no_rule_matches():
    """A wrong expansion is worse than none."""
    out = V.expand_abbreviations("Chỉ tiêu TP được giao")
    assert "trưởng phòng" not in out
    assert "thành phố" not in out


def test_cv_disambiguation():
    assert "hồ sơ" in V.expand_abbreviations("Gửi CV xin việc qua email")
    assert "chuyên viên" in V.expand_abbreviations("Tuyển CV Kinh doanh")


def test_cp_is_joint_stock_only_after_company():
    assert "cổ phần" in V.expand_abbreviations("Công ty CP Xây dựng")


# --- 2.4 Word segmentation --------------------------------------------------

requires_segmenter = pytest.mark.skipif(
    not V.segmenter_available(), reason="underthesea not installed"
)


@requires_segmenter
def test_segment_joins_multi_syllable_words():
    """The whole point: 'nhân viên' is ONE word spread over two syllables."""
    assert V.segment("Nhân viên kinh doanh") == "Nhân_viên kinh_doanh"
    assert V.segment("bất động sản") == "bất_động_sản"


@requires_segmenter
def test_segment_disambiguates_the_shared_syllable():
    """'viên' means different things in each; segmentation keeps them apart."""
    a = V.segment("nhân viên").split()
    b = V.segment("chuyên viên").split()
    assert a == ["nhân_viên"] and b == ["chuyên_viên"]


@requires_segmenter
def test_segment_keeps_english_tokens_readable():
    """Known limitation: adjacent English tokens may be joined.

    'Fresher SEO' becomes 'Fresher_SEO'. Documented rather than fixed — the
    accent-folded char n-gram channel (2.6) still matches the pieces, and no
    reliable rule separates English from undiacritised Vietnamese, which is
    also pure ASCII.
    """
    out = V.segment("Nhân viên Digital Marketing")
    assert "Nhân_viên" in out
    assert "Digital" in out


def test_segment_is_a_noop_on_empty_input():
    assert V.segment("") == ""
    assert V.segment(None) == ""


@pytest.fixture()
def restore_segmenter():
    """Put the process back on the default backend after a switch."""
    yield
    V.use_segmenter("underthesea")


def test_use_segmenter_switches_backend(restore_segmenter):
    """The switch must change what actually segments, not just a label.

    docs/02-vietnamese-nlp.md §5 compares two backends; if the switch silently
    kept the old one, that table would be one backend measured twice.
    """
    pytest.importorskip("pyvi")
    V.use_segmenter("pyvi")
    assert V.segmenter_name() == "pyvi"
    assert V.segment("Nhân viên kinh doanh") == "Nhân_viên kinh_doanh"


def test_unknown_segmenter_is_fatal(restore_segmenter):
    """A typo in $VIETJOBS_SEGMENTER must stop the run, not fall back quietly."""
    with pytest.raises(SystemExit):
        V.use_segmenter("underthesa")
        V.segmenter_available()


# --- 2.6 Accent folding -----------------------------------------------------


def test_fold_accents():
    assert V.fold_accents("Nhân Viên Kinh Doanh") == "Nhan Vien Kinh Doanh"
    assert V.fold_accents("Đà Nẵng") == "Da Nang"
    assert V.fold_accents("đường") == "duong"


def test_fold_accents_makes_undiacritised_titles_match():
    assert V.fold_accents("Nhân Viên Kinh Doanh").lower() == "nhan vien kinh doanh"


# --- 2.7 Provinces ----------------------------------------------------------


def test_district_folds_into_its_province():
    assert V.normalize_province("hà đông") == "hà nội"
    assert V.normalize_province("cầu giấy") == "hà nội"
    assert V.normalize_province("bình thạnh") == "hồ chí minh"
    assert V.normalize_province("thủ đức") == "hồ chí minh"


def test_city_aliases_fold_together():
    for alias in ["TP.HCM", "tphcm", "Sài Gòn", "hcm", "Tp. Hồ Chí Minh"]:
        assert V.normalize_province(alias) == "hồ chí minh"


def test_province_keeps_its_own_name():
    assert V.normalize_province("bắc ninh") == "bắc ninh"
    assert V.normalize_province("hà nội") == "hà nội"


def test_province_takes_the_first_of_a_list():
    assert V.normalize_province("hà đông, bắc ninh") == "hà nội"


def test_unknown_place_passes_through():
    assert V.normalize_province("") == "unknown"


# --- 2.8 Salary masking -----------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "thu nhập 15 - 22 triệu",
        "lương 15 triệu",
        "Mức lương: 8.000.000đ",
        "từ 10-15 triệu/tháng",
        "Lương 1200 USD",
        "thưởng 5tr mỗi quý",
    ],
)
def test_pay_figures_are_masked(text):
    assert "<SALARY>" in V.mask_salary(text)


@pytest.mark.parametrize(
    "text",
    [
        "nền tảng có 40 triệu người dùng",
        "500 triệu đồng doanh thu mỗi tháng",
        "tiếp cận hàng triệu khách hàng",
        "kênh đạt 2 triệu lượt xem",
    ],
)
def test_quantities_are_not_masked(text):
    """Counting nouns must not be mistaken for pay — this was a real bug."""
    assert "<SALARY>" not in V.mask_salary(text)


def test_masking_keeps_the_fact_that_pay_was_mentioned():
    out = V.mask_salary("Lương cứng 15 triệu cộng hoa hồng")
    assert "Lương" in out and "<SALARY>" in out and "15" not in out


def test_masking_removes_every_digit_of_the_figure():
    out = V.mask_salary("Thu nhập từ 12.000.000 đến 18.000.000 VNĐ")
    assert "12" not in out and "18" not in out


# --- 2.9 Structured fields --------------------------------------------------


@pytest.mark.parametrize(
    "raw,months",
    [
        ("2 năm", 24.0),
        ("6 tháng", 6.0),
        ("1 năm", 12.0),
        ("Không yêu cầu", 0.0),
        ("", 0.0),
        ("Chưa có kinh nghiệm", 0.0),
    ],
)
def test_experience_to_months(raw, months):
    assert V.experience_to_months(raw) == months


def test_parse_list_field_handles_python_repr():
    assert V.parse_list_field("['Cao đẳng', 'Đại học']") == ["Cao đẳng", "Đại học"]


def test_parse_list_field_tolerates_broken_input():
    assert V.parse_list_field("[Cao đẳng, Đại học") != []
    assert V.parse_list_field("[]") == []
    assert V.parse_list_field(None) == []
    assert V.parse_list_field("nan") == []


def test_parse_list_field_accepts_plain_csv():
    assert V.parse_list_field("Python, SQL") == ["Python", "SQL"]


# --- Grouping ---------------------------------------------------------------


def test_group_key_is_stable_and_fold_insensitive():
    a = V.group_key("Nhân Viên Kinh Doanh", "Mô tả", "Yêu cầu")
    b = V.group_key("nhan vien kinh doanh", "mô tả", "yêu cầu")
    assert a == b
    assert len(a) == 16


def test_group_key_separates_different_postings():
    a = V.group_key("Nhân Viên Kinh Doanh", "Mô tả A", "")
    b = V.group_key("Nhân Viên Kế Toán", "Mô tả B", "")
    assert a != b


# --- Composition ------------------------------------------------------------


def test_preprocess_masks_only_when_asked():
    text = "Lương 15 triệu, làm tại TP.HCM"
    assert "<SALARY>" not in V.preprocess(text, mask=False)
    assert "<SALARY>" in V.preprocess(text, mask=True)


def test_preprocess_is_deterministic():
    text = "NV Kinh doanh tại TP.HCM, lương 15 triệu"
    assert V.preprocess(text, mask=True) == V.preprocess(text, mask=True)


def test_preprocess_empty_input():
    assert V.preprocess(None) == ""
    assert V.preprocess("nan") == ""
