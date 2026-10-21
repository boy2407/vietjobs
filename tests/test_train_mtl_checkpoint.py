"""Ghi checkpoint nguyên tử của train_mtl: máy tắt giữa lúc ghi không được làm hỏng bản cũ."""
import pytest

torch = pytest.importorskip("torch")

from vietjobs.dl.train_mtl import save_atomic  # noqa: E402


def test_save_atomic_writes_loadable_file_and_leaves_no_tmp(tmp_path):
    p = tmp_path / "best.pt"
    save_atomic({"w": torch.arange(3)}, p)
    assert torch.load(p)["w"].tolist() == [0, 1, 2]
    assert not (tmp_path / "best.pt.tmp").exists()


def test_save_atomic_failure_keeps_previous_file(tmp_path, monkeypatch):
    p = tmp_path / "best.pt"
    save_atomic({"w": torch.zeros(2)}, p)

    def boom(obj, path):
        open(path, "wb").write(b"partial")  # ghi dở rồi chết
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", boom)
    with pytest.raises(OSError):
        save_atomic({"w": torch.ones(2)}, p)
    monkeypatch.undo()
    assert torch.load(p)["w"].tolist() == [0.0, 0.0]  # bản cũ còn nguyên
