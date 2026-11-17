import subprocess

import pytest

from scripts.fetch_third_party import fetch_repo


def _git(*a, cwd=None):
    return subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def upstream(tmp_path):
    up = tmp_path / "up"
    up.mkdir()
    _git("init", "-q", cwd=up)
    for i in (1, 2):
        (up / "f.txt").write_text(str(i))
        _git("add", "f.txt", cwd=up)
        _git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", str(i), cwd=up)
    return up


def test_fetch_checks_out_pinned_commit(tmp_path, upstream):
    first = _git("rev-list", "--max-parents=0", "HEAD", cwd=upstream)
    dst = tmp_path / "tp" / "x"
    assert fetch_repo(str(upstream), first, dst) == "cloned"
    assert (dst / "f.txt").read_text() == "1"
    assert fetch_repo(str(upstream), first, dst) == "exists"


def test_fetch_bad_commit_raises(tmp_path, upstream):
    with pytest.raises(RuntimeError, match="commit"):
        fetch_repo(str(upstream), "0" * 40, tmp_path / "tp" / "y")
