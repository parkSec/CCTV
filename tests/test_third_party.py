import json
import re
import subprocess
import sys

import pytest

from src.config import PROJECT_ROOT
from src.enhance.base import EnhancerUnavailable
from src.enhance.third_party import commit_of, load_symbol


def _write(p, text):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def test_same_module_name_two_repos(tmp_path):
    _write(tmp_path / "a" / "model.py", 'NAME = "A"\n')
    _write(tmp_path / "b" / "model.py", 'NAME = "B"\n')
    assert load_symbol(tmp_path / "a", "model", "NAME") == "A"
    assert load_symbol(tmp_path / "b", "model", "NAME") == "B"
    assert "model" not in sys.modules
    assert str(tmp_path / "a") not in sys.path


def test_intra_repo_import(tmp_path):
    _write(tmp_path / "c" / "model.py", "from helper import X\n")
    _write(tmp_path / "c" / "helper.py", "X = 3\n")
    assert load_symbol(tmp_path / "c", "model", "X") == 3
    assert "helper" not in sys.modules


def test_existing_module_not_purged(tmp_path):
    import json as json_before
    _write(tmp_path / "d" / "model.py", "import json\nY = 1\n")
    assert load_symbol(tmp_path / "d", "model", "Y") == 1
    assert sys.modules["json"] is json_before


def test_outside_module_first_imported_in_repo_is_kept(tmp_path, monkeypatch):
    """저장소 코드가 처음 불러온 외부 패키지(torchvision 등)는 지우면 안 된다. 지우면 다음 import 때
    다시 실행되어 torch 연산자 등록이 겹쳐 RuntimeError가 난다."""
    _write(tmp_path / "lib" / "extlib_once.py", "import sys\nassert 'extlib_once_loaded' not in sys.modules\n"
           "sys.modules['extlib_once_loaded'] = sys\n")
    monkeypatch.syspath_prepend(str(tmp_path / "lib"))
    monkeypatch.delitem(sys.modules, "extlib_once_loaded", raising=False)
    _write(tmp_path / "k" / "model.py", "import extlib_once\nY = 1\n")
    assert load_symbol(tmp_path / "k", "model", "Y") == 1
    assert "extlib_once" in sys.modules and "model" not in sys.modules
    import extlib_once  # noqa: F401  (다시 실행되면 assert에 걸린다)


def test_build_in_repo_relative_path_in_constructor(tmp_path, monkeypatch):
    """RUAS처럼 생성자가 저장소 기준 상대 경로('./model/denoise.pt')를 열어도 된다."""
    from src.enhance.third_party import build_in_repo
    _write(tmp_path / "m" / "model.py",
           "class Net:\n    def __init__(self):\n        self.t = open('./data/w.txt').read()\n")
    _write(tmp_path / "m" / "data" / "w.txt", "ok")
    monkeypatch.chdir(tmp_path)
    assert build_in_repo("m", "model", "Net").t == "ok"
    assert str(tmp_path) == str(__import__("pathlib").Path.cwd())


def test_missing_repo_raises(tmp_path):
    with pytest.raises(EnhancerUnavailable, match="fetch_third_party"):
        load_symbol(tmp_path / "nope", "model", "X")


def test_missing_attr_raises(tmp_path):
    _write(tmp_path / "e" / "model.py", "Z = 1\n")
    with pytest.raises(EnhancerUnavailable):
        load_symbol(tmp_path / "e", "model", "X")


def test_import_error_in_repo_raises_unavailable(tmp_path):
    _write(tmp_path / "f" / "model.py", "import package_that_does_not_exist\n")
    with pytest.raises(EnhancerUnavailable, match="package_that_does_not_exist"):
        load_symbol(tmp_path / "f", "model", "X")


def test_manifest_has_four_repos():
    m = json.loads((PROJECT_ROOT / "third_party_manifest.json").read_text(encoding="utf-8"))
    assert set(m) == {"zerodce", "sci", "ruas", "nightenh"}
    for v in m.values():
        assert v["url"].startswith("https://github.com/")
        assert re.fullmatch(r"[0-9a-f]{40}", v["commit"])


def test_commit_of(tmp_path):
    if subprocess.run(["git", "--version"], capture_output=True).returncode != 0:
        pytest.skip("git 없음")
    r = tmp_path / "r"
    r.mkdir()
    run = lambda *a: subprocess.run(["git", "-C", str(r), *a], check=True, capture_output=True)  # noqa: E731
    run("init", "-q")
    (r / "x.txt").write_text("x")
    run("add", "x.txt")
    run("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "x")
    assert re.fullmatch(r"[0-9a-f]{40}", commit_of(r))
    assert commit_of(tmp_path / "missing") is None


def test_build_in_repo_allows_lazy_import(tmp_path):
    from src.enhance.third_party import build_in_repo
    _write(tmp_path / "g" / "model.py",
           "class Net:\n    def __init__(self, k):\n        from helper import V\n        self.v = V * k\n")
    _write(tmp_path / "g" / "helper.py", "V = 5\n")
    net = build_in_repo(tmp_path / "g", "model", "Net", 2)
    assert net.v == 10
    assert "model" not in sys.modules and "helper" not in sys.modules


def test_build_in_repo_constructor_error_is_unavailable(tmp_path):
    from src.enhance.third_party import build_in_repo
    _write(tmp_path / "h" / "model.py", "class Net:\n    def __init__(self):\n        raise OSError('bad file')\n")
    with pytest.raises(EnhancerUnavailable, match="bad file"):
        build_in_repo(tmp_path / "h", "model", "Net")
