from pathlib import Path

from src.io_utils import relativize_paths


def test_project_paths_become_relative(tmp_path):
    root = tmp_path / "proj"
    obj = {"a": root / "datasets" / "x.json", "b": [root / "weights" / "w.pth"], "c": 3}
    out = relativize_paths(obj, root=root, home=tmp_path / "home")
    assert out == {"a": "datasets/x.json", "b": ["weights/w.pth"], "c": 3}


def test_home_paths_hide_username(tmp_path):
    home = tmp_path / "Users" / "someone"
    out = relativize_paths({"p": home / "Downloads" / "v.mp4"}, root=tmp_path / "proj", home=home)
    assert out == {"p": "~/Downloads/v.mp4"}
    assert "someone" not in str(out)


def test_other_paths_and_strings_kept(tmp_path):
    out = relativize_paths({"p": Path("/opt/data/a.jpg"), "s": "text"}, root=tmp_path / "proj", home=tmp_path / "h")
    assert out == {"p": "/opt/data/a.jpg", "s": "text"}


def test_paths_inside_strings_are_hidden(tmp_path):
    """건너뜀 사유 같은 문장 속 경로도 가린다 (예: '가중치 파일 없음: C:\\Users\\이름\\...')."""
    root, home = tmp_path / "home" / "proj", tmp_path / "home"
    msg = f"가중치 파일 없음: {root / 'weights' / 'a.pth'} / 다른 곳: {home / 'x.txt'}"
    out = relativize_paths({"reason": msg}, root=root, home=home)
    assert str(home) not in out["reason"]
    assert "weights" in out["reason"] and "~" in out["reason"]
