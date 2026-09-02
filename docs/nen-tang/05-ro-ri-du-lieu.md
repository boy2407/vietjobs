[← Ma trận thưa](04-ma-tran-thua-va-so-chieu.md) · [Nền tảng](00-index.md) · [Đo lường và baseline →](06-do-luong-va-baseline.md)

# 5. Rò rỉ dữ liệu

Rò rỉ (*data leakage*) là khi mô hình nhìn thấy — trực tiếp hay gián tiếp —
thông tin lẽ ra nó không được có lúc dự đoán thật.

Điều làm rò rỉ nguy hiểm hơn mọi loại bug khác:

> **Nó không báo lỗi. Nó làm điểm số ĐẸP LÊN.**

Một bug thường làm chương trình chết, hoặc làm điểm tệ đi — bạn nhìn ra ngay.
Rò rỉ làm mọi thứ trông tuyệt vời. Bạn chỉ phát hiện ra khi đưa mô hình ra đời
thật và nó hỏng, mà lúc đó thì báo cáo đã in xong rồi.

---

## Ba loại rò rỉ trong đúng dự án này

### Loại 1 — nhãn nằm ngay trong đầu vào

**Bài toán:** dự đoán mức lương từ nội dung tin.
**Vấn đề:** rất nhiều tin **viết thẳng con số lương trong phần mô tả và phúc lợi**.

Đo được:

| Ô văn bản | Số dòng nhắc lại con số lương |
|---|---|
| phúc lợi | **5.081 (10,65 %)** |
| mô tả | 250 (0,52 %) |
| yêu cầu | 103 (0,22 %) |
| tiêu đề | 121 (0,25 %) |

Không xử lý thì mô hình chỉ cần học một quy tắc: *"tìm con số đứng cạnh chữ lương,
in nó ra"*. R² đẹp. Nhưng nó không **dự đoán** gì cả — nó **sao chép**. Với tin
thật không ghi lương, nó vô dụng.

**Cách chặn — che, không xoá:**

```
"Lương 15 - 22 triệu/tháng"   →   "Lương <SALARY>/tháng"
```

Con số biến mất, chữ "Lương" **ở lại**. Đó là cố ý: *"tin này có nhắc tới lương"*
là tín hiệu hợp lệ; *"lương bằng bao nhiêu"* mới là đáp án phải giấu.

**Phần khó nhất là phân biệt tiền với số đếm:**

| Câu | Che? | Vì sao |
|---|---|---|
| "thu nhập 15 - 22 triệu" | ✅ | tiền |
| "40 triệu người dùng" | ❌ | đang đếm người |
| "500 triệu đồng doanh thu" | ❌ | đang đếm doanh thu |
| "thưởng 5tr mỗi quý" | ✅ | tiền |

Comment trong `tests/test_vitext.py:210` ghi thẳng: *"this was a real bug"*.
Che quá tay cũng là một dạng phá dữ liệu.

### Loại 2 — cùng một tin nằm ở cả train lẫn test

Nhà tuyển dụng đăng lại một tin nhiều lần, sửa vài chữ. Đo được:
**12.808 dòng (26,8 %)** là tin đăng lại; 47.707 dòng chỉ có **34.899 nhóm** thật sự.

Chia tập ngẫu nhiên **theo dòng** thì gần như chắc chắn một tin có bản ở train và
bản gần giống ở test. Mô hình chỉ cần **thuộc lòng** là ăn điểm — và điểm đó không
nói gì về khả năng xử lý tin nó chưa từng thấy.

**Cách chặn:** băm nội dung → `group_id`, và bắt **cả nhóm** đi cùng một tập.

```python
straddling = int((df.groupby("group_id")["split"].nunique() > 1).sum())
assert straddling == 0, f"{straddling} groups straddle splits — split is leaking"
```

`assert` này không được phép tắt. Kết quả hiện tại: **0 nhóm lọt**.

> Lưu ý: tin đăng lại **không bị xoá** — chúng vẫn là dữ liệu thật. Chỉ có cách
> **chia tập** thay đổi. Xem [01-data-audit.md §2](../01-data-audit.md#2-khử-trùng-lặp--hai-tầng-hai-mục-đích-khác-nhau).

### Loại 3 — chạm vào tập test nhiều lần

Loại này tinh vi nhất vì nó không nằm trong code, nó nằm trong **quy trình làm việc**.

Mỗi lần bạn: chấm test → thấy điểm thấp → chỉnh mô hình → chấm test lại, là một
lần thông tin từ test chảy vào quyết định thiết kế. Làm mười lần thì test không
còn là ước lượng không thiên lệch nữa — nó đã trở thành một tập val thứ hai, và
bạn không còn tập nào để biết mô hình thật sự tốt đến đâu.

**Cách chặn — bằng luật, không bằng code:**

| Tập | Dùng để | Chạm bao nhiêu lần |
|---|---|---|
| `train` | Huấn luyện | Không giới hạn |
| `val` | Chọn mô hình, siêu tham số, bước tiền xử lý | Không giới hạn |
| `test` | Báo cáo con số cuối cùng | **Đúng một lần**, ở cuối |

`train.py` từ chối `--eval test` nếu không có cờ `--confirm-test`. Cờ đó tồn tại
để việc chạm vào test là **hành động có chủ ý**, không phải mặc định.

Trong 27 thí nghiệm của dự án, đúng **một** dòng chấm trên test:
`cat-FINAL-svm-C0.02-test`.

---

## Cơ chế phòng thủ: một cửa duy nhất

Loại 1 được chặn bằng một quy tắc kiến trúc, không phải bằng sự cẩn thận:

**Mỗi cột văn bản tồn tại ở hai bản — bản thô và bản đã che — và chỉ có MỘT hàm
quyết định bài toán nào đọc bản nào.**

```python
def resolve_column(name, *, task, segmented):
    col = _MASKABLE[name] if (task in C.MASKED_TASKS and name in _MASKABLE) else name
    ...
```

Vì sao gom vào một hàm thay vì rải `if` khắp nơi: **một cửa thì viết test được.**
`tests/test_no_leak.py` chỉ cần một dòng để canh toàn bộ:

```python
assert set(F.source_columns(ct)) & F.UNMASKED_COLUMNS == set()
```

---

## Bài học đắt nhất: test chỉ bảo vệ được thứ bạn nghĩ tới

Trong lúc viết tài liệu này, đọc kỹ `features.py` thì phát hiện:

**`soft_skills_text` và `qualifications_text` đi vào mô hình lương ở dạng chưa che.**
Hai cột đó không có bản `_masked`, và **không nằm trong `UNMASKED_COLUMNS`** — nên
`tests/test_no_leak.py` chạy xanh mà không hề kiểm tra chúng.

Chưa biết hai cột đó có nhắc lại con số lương hay không, vì
`scripts/measure_vitext.py` không đo chúng. Việc phải làm ghi ở
[09-lo-trinh.md — Ưu tiên 0b](../09-lo-trinh.md#ưu-tiên-0b--đóng-lỗ-hổng-che-lương-chưa-được-test-phủ).

Đây là ví dụ sống cho điều đáng nhớ nhất trong cả note này:

> Một test chống rò rỉ **không** chứng minh rằng không có rò rỉ.
> Nó chỉ chứng minh rằng những đường rò rỉ **người viết test đã nghĩ ra** đang bị chặn.
>
> Danh sách kiểu `UNMASKED_COLUMNS` là danh sách **cho phép sai sót**: quên thêm
> một cột vào đó thì test im lặng bỏ qua cột ấy. Một thiết kế an toàn hơn sẽ đảo
> ngược mặc định — liệt kê những cột **được phép** đọc, và từ chối mọi cột khác.

---

## Danh sách kiểm tra khi bạn nghi có rò rỉ

1. **Điểm cao bất thường?** Nghi ngờ trước, ăn mừng sau. Rò rỉ luôn giống một
   mô hình xuất sắc.
2. **Đặc trưng nào mạnh nhất?** Nếu đặc trưng số một là thứ mà lúc dự đoán thật
   bạn *chưa thể có*, đó là rò rỉ.
3. **Đặc trưng này có tồn tại lúc dự đoán không?** Câu hỏi vàng cho mọi cột.
4. **Có bản sao gần giống giữa train và test không?** Băm nội dung rồi đếm.
5. **Đã chạm test mấy lần rồi?** Nếu phải nghĩ để nhớ, thì đã quá nhiều.
6. **Cột nào KHÔNG nằm trong danh sách kiểm tra của test?** — bài học ở trên.

---

## Quay lại thực tế dự án

- [03-protocol.md](../03-protocol.md) — hợp đồng train/val/test đầy đủ
- [01-data-audit.md](../01-data-audit.md#3-bốn-bản-sao-của-mỗi-cột-văn-bản) — bốn bản sao của mỗi cột
- [05-dac-trung-tfidf.md](../05-dac-trung-tfidf.md#1-resolve_column--cửa-duy-nhất) — `resolve_column` và lỗ hổng chưa vá
- [02-vietnamese-nlp.md](../02-vietnamese-nlp.md) — bước 4 (che lương) và bước 9 (khoá gộp nhóm)
