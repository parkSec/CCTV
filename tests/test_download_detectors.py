from pathlib import Path

from scripts.download_detectors import download_detectors


def test_download_copies_and_skips_existing(tmp_path):
    calls = []

    def fake_download(repo_id, filename):
        calls.append((repo_id, filename))
        p = tmp_path / "cache" / repo_id.replace("/", "_") / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"w")
        return str(p)

    existing = tmp_path / "w" / "m.pt"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"old")
    dets = [{"name": "s", "weights": tmp_path / "w" / "s.pt", "hf_repo": "org/s"},
            {"name": "m", "weights": existing, "hf_repo": "org/m"}]
    done = download_detectors(dets, downloader=fake_download)
    assert calls == [("org/s", "best.pt")]
    assert (tmp_path / "w" / "s.pt").read_bytes() == b"w"
    assert existing.read_bytes() == b"old"
    assert done == {"s": "downloaded", "m": "exists"}
