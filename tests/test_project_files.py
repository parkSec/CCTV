from src.config import PROJECT_ROOT


def _requirement_names() -> set[str]:
    text = (PROJECT_ROOT / "requirements.txt").read_text(encoding="ascii")
    return {line.split("#")[0].strip().split(">")[0].split("=")[0].split("<")[0].lower()
            for line in text.splitlines() if line.split("#")[0].strip()}


def test_requirements_is_ascii():
    """Windows pip는 requirements.txt를 cp949로 읽으므로 한글이 있으면 설치가 실패한다."""
    data = (PROJECT_ROOT / "requirements.txt").read_bytes()
    assert data.isascii()


def test_requirements_no_realesrgan():
    """재설계에서 Real-ESRGAN을 뺐다. 화질 지표는 pyiqa로 잰다."""
    names = _requirement_names()
    assert "realesrgan" not in names and "basicsr" not in names
    assert "pyiqa" in names


def test_requirements_include_hf_downloader():
    """scripts/download_bdd100k_hf.py가 huggingface_hub를 쓴다."""
    assert "huggingface_hub" in _requirement_names()


def test_requirements_include_gdown():
    """scripts/download_exdark.py가 Google Drive에서 받을 때 gdown을 쓴다."""
    assert "gdown" in _requirement_names()


def test_gitignore_excludes_logs_and_weights():
    """로그(사용자 경로가 찍힘)와 가중치 파일(용량 큼)은 저장소에 올리지 않는다."""
    lines = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").split()
    assert "*.log" in lines and "pytest_log.txt" in lines and "*.pt" in lines and "*.pth" in lines


def test_gitignore_has_third_party():
    """외부 개선 모델 저장소(third_party/)는 우리 저장소에 올리지 않는다."""
    lines = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").split()
    assert "third_party/" in lines
