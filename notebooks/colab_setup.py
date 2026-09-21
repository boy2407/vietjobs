"""Khôi phục dữ liệu + cache từ Google Drive cho một phiên Colab, rồi chạy cổng test.

    python notebooks/colab_setup.py --family raw        # T8.3, T8.5 (bài category)
    python notebooks/colab_setup.py --family masked     # T8.4 (bài salary)
    python notebooks/colab_setup.py --family both

Ba notebook T8_3 / T8_4 / T8_5 dùng chung script này thay vì chép ba lần cùng một
đoạn — chép tay ba lần thì sớm muộn lệch nhau. Mỗi bước in đúng một dòng xác nhận.

Đường dẫn trên Drive (xem memory `colab-drive-paths`):

    MyDrive/vietjobs_splits/     train.csv dev.csv test.csv manifest.json   (337 MB)
    MyDrive/vietjobs_cache/      *-len256.npy *-mask.npy *-tok.f16.json + dev-*-tok.f16.npy
    Othercomputers/My Mac/vietjobs/artifacts/embeddings/   bản đầy đủ 30 GB, chỉ đọc

Hai tệp train-*-tok.f16.npy (12,9 GB mỗi họ) cố ý KHÔNG khôi phục từ Drive: encode
lại trên T4 mất ~8–10 phút, copy qua FUSE mount thường lâu hơn. Cờ
--copy-train-token-from-mac đổi lựa chọn đó nếu muốn.

Mã thoát: 0 xong · 1 cổng test đỏ · 2 thiếu dữ liệu trên Drive · 3 không ở gốc repo.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

EXPECTED = {"train": 34354, "dev": 3812, "test": 9541}
FAMILY_TASK = {"raw": "category", "masked": "salary"}
HIDDEN = 768
MAX_LEN = 256


def step(msg: str) -> None:
    print(f"[setup] {msg}", flush=True)


def mount_drive(root: Path) -> Path:
    mydrive = root / "MyDrive"
    if mydrive.exists():
        step(f"Drive đã mount tại {root}")
        return mydrive
    try:
        from google.colab import drive  # type: ignore
    except ImportError:
        sys.exit(f"[setup] không phải Colab và {mydrive} không tồn tại — script này chỉ chạy trên Colab")
    drive.mount(str(root))
    step(f"đã mount Drive tại {root}")
    return mydrive


def restore_splits(mydrive: Path) -> None:
    src = mydrive / "vietjobs_splits"
    dst = Path("data/processed")
    need = ["train.csv", "dev.csv", "test.csv", "manifest.json"]
    missing = [f for f in need if not (src / f).exists()]
    if missing:
        print(f"[setup] THIẾU trên Drive: {src} — {missing}")
        print("[setup] Chạy nhánh dựng lại từ nguồn: tải HuggingFace → sha256 → "
              "`python -m vietjobs.dataset build` (~20 phút), rồi lưu lên Drive.")
        sys.exit(2)
    (dst / "splits").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for f in need[:3]:
        shutil.copy(src / f, dst / "splits" / f)
    shutil.copy(src / "manifest.json", dst / "manifest.json")
    m = json.loads((dst / "manifest.json").read_text())
    rows = {s: m["splits"][s]["rows"] for s in EXPECTED}
    assert rows == EXPECTED, f"manifest lệch máy chính: {rows} != {EXPECTED}"
    assert m["groups_straddling_splits"] == 0, "có nhóm bị chia lệch giữa các tập"
    step(f"splits khôi phục {rows}, 0 nhóm lệch — {time.time() - t0:.0f}s")


def restore_small_cache(mydrive: Path, families: list[str]) -> None:
    src = mydrive / "vietjobs_cache"
    dst = Path("artifacts/embeddings")
    dst.mkdir(parents=True, exist_ok=True)
    t0, n, gb = time.time(), 0, 0.0
    for fam in families:
        for split in ("train", "dev"):
            stem = f"{split}-{fam}-len{MAX_LEN}"
            names = [f"{stem}.npy", f"{stem}-mask.npy", f"{stem}-tok.f16.json"]
            if split == "dev":
                names.append(f"{stem}-tok.f16.npy")        # 1,43 GB — đáng copy
            for name in names:
                if (dst / name).exists():
                    continue
                if not (src / name).exists():
                    step(f"  không có {name} trên Drive — encode sẽ tự tạo")
                    continue
                shutil.copy(src / name, dst / name)
                n += 1
                gb += (dst / name).stat().st_size / 1e9
    step(f"cache nhỏ: {n} tệp, {gb:.2f} GB từ {src} — {time.time() - t0:.0f}s")


def train_token_cache(mydrive: Path, fam: str, copy_from_mac: bool) -> None:
    dst = Path("artifacts/embeddings")
    name = f"train-{fam}-len{MAX_LEN}-tok.f16.npy"
    if (dst / name).exists():
        step(f"{name} đã có, bỏ qua")
        return
    t0 = time.time()
    if copy_from_mac:
        src = (mydrive.parent / "Othercomputers" / "My Mac" / "vietjobs"
               / "artifacts" / "embeddings" / name)
        if not src.exists():
            sys.exit(f"[setup] --copy-train-token-from-mac nhưng không thấy {src}")
        shutil.copy(src, dst / name)
        step(f"{name} copy từ Mac backup (12,9 GB) — {time.time() - t0:.0f}s")
        return
    # `token_cache_for` chỉ bỏ qua khi CẢ tok lẫn mask đã có; tok thiếu → encode lại
    # train cho họ này (mask bị ghi đè bằng đúng nội dung). dev đã có → bỏ qua.
    cmd = [sys.executable, "-m", "vietjobs.dl.encode", "--task", FAMILY_TASK[fam],
           "--splits", "train", "--pooling", "none", "--device", "cuda"]
    step(f"encode train token họ {fam} trên GPU: {' '.join(cmd[2:])}")
    subprocess.run(cmd, check=True, env={**os.environ, "PYTHONPATH": "src"})
    step(f"{name} — {time.time() - t0:.0f}s")


def run_tests() -> None:
    cmd = [sys.executable, "-m", "pytest", "-q",
           "tests/test_dl_encode.py", "tests/test_dl_heads.py"]
    r = subprocess.run(cmd, env={**os.environ, "PYTHONPATH": "src"})
    if r.returncode:
        print("[setup] CỔNG TEST ĐỎ — cache hoặc split sai, ĐỪNG huấn luyện. Dừng.")
        sys.exit(1)
    step("cổng test xanh — cache khớp split, sẵn sàng huấn luyện")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", default="raw", choices=["raw", "masked", "both"],
                    help="raw = bài category (T8.3/T8.5) · masked = bài salary (T8.4)")
    ap.add_argument("--drive-root", default="/content/drive")
    ap.add_argument("--copy-train-token-from-mac", action="store_true",
                    help="copy 12,9 GB/họ từ Othercomputers/My Mac thay vì encode lại (thường chậm hơn)")
    ap.add_argument("--skip-tests", action="store_true")
    args = ap.parse_args()

    if not Path("src/vietjobs").exists():
        sys.exit("[setup] hãy chạy từ gốc repo (thư mục có src/vietjobs)")
    families = ["raw", "masked"] if args.family == "both" else [args.family]

    mydrive = mount_drive(Path(args.drive_root))
    restore_splits(mydrive)
    restore_small_cache(mydrive, families)
    for fam in families:
        train_token_cache(mydrive, fam, args.copy_train_token_from_mac)
    if not args.skip_tests:
        run_tests()
    step(f"xong — họ {families}; đĩa còn: "
         + subprocess.run(["df", "-h", "/content"], capture_output=True, text=True)
         .stdout.strip().splitlines()[-1])


if __name__ == "__main__":
    main()
