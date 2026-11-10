from src.config import PROJECT_ROOT


def _requirement_names() -> set[str]:
    text = (PROJECT_ROOT / "requirements.txt").read_text(encoding="ascii")
    return {line.split("#")[0].strip().split(">")[0].split("=")[0].split("<")[0].lower()
            for line in text.splitlines() if line.split("#")[0].strip()}


def test_requirements_is_ascii():
    """Windows pip는 requirements.txt를 cp949로 읽으므로 한글이 있으면 설치가 실패한다."""
    data = (PROJECT_ROOT / "requirements.txt").read_bytes()
    assert data.isascii()


def test_requirements_use_official_realesrgan():
    """Real-ESRGAN은 공식 realesrgan 패키지(+basicsr)를 쓴다. Python 3.11/3.12 환경 전제."""
    names = _requirement_names()
    assert {"realesrgan", "basicsr"} <= names
    assert "spandrel" not in names


def test_requirements_include_hf_downloader():
    """scripts/download_bdd100k_hf.py가 huggingface_hub를 쓴다."""
    assert "huggingface_hub" in _requirement_names()
