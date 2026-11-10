import zipfile

from scripts.download_exdark import arrange_layout, extract_zip
from src.data.exdark import ExDarkDataset

HEADER = "% bbGt version=3\n"


def _make_zip(path, files: dict):
    with zipfile.ZipFile(path, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)


def test_extract_and_arrange_standard_names(tmp_path):
    """압축 안의 폴더 이름이 ExDark / ExDark_Annno 인 경우."""
    _make_zip(tmp_path / "img.zip", {"ExDark/Car/a.jpg": b"x", "ExDark/People/b.png": b"x"})
    _make_zip(tmp_path / "ann.zip", {"ExDark_Annno/Car/a.jpg.txt": HEADER + "Car 1 2 3 4 0 0 0 0 0 0 0\n",
                                     "ExDark_Annno/People/b.png.txt": HEADER + "People 1 1 2 2 0 0 0 0 0 0 0\n"})
    work = tmp_path / "work"
    extract_zip(tmp_path / "img.zip", work)
    extract_zip(tmp_path / "ann.zip", work)
    dest = tmp_path / "ExDark"
    arrange_layout(work, dest)
    assert (dest / "ExDark" / "Car" / "a.jpg").exists()
    assert (dest / "ExDark_Annno" / "People" / "b.png.txt").exists()


def test_arrange_handles_other_folder_names(tmp_path):
    """압축 폴더 이름이 달라도(예: 중첩 폴더) 클래스 폴더를 찾아 정해진 위치로 옮긴다."""
    work = tmp_path / "work"
    for cls in ["Car", "Bus"]:
        (work / "something" / "Images" / cls).mkdir(parents=True)
        (work / "something" / "Images" / cls / f"{cls}1.JPG").write_bytes(b"x")
        (work / "labels_v2" / cls).mkdir(parents=True)
        (work / "labels_v2" / cls / f"{cls}1.JPG.txt").write_text(HEADER + f"{cls} 0 0 5 5 0 0 0 0 0 0 0\n")
    dest = tmp_path / "ExDark"
    arrange_layout(work, dest)
    samples = ExDarkDataset(dest / "ExDark", dest / "ExDark_Annno").load()
    assert sorted(s.image_id for s in samples) == ["Bus/Bus1.JPG", "Car/Car1.JPG"]


def test_arrange_is_idempotent(tmp_path):
    """이미 정리된 상태에서 다시 실행해도 그대로 둔다."""
    dest = tmp_path / "ExDark"
    (dest / "ExDark" / "Car").mkdir(parents=True)
    (dest / "ExDark" / "Car" / "a.jpg").write_bytes(b"x")
    (dest / "ExDark_Annno" / "Car").mkdir(parents=True)
    (dest / "ExDark_Annno" / "Car" / "a.jpg.txt").write_text(HEADER)
    arrange_layout(tmp_path / "empty_work", dest)
    assert (dest / "ExDark" / "Car" / "a.jpg").exists()
