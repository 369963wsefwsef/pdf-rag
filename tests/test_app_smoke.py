"""第五步的冒烟测试：用 Streamlit 官方的 AppTest 真正跑一遍 app.py。

不启动浏览器、不需要任何 API Key：验证"没配 Key 时界面也能正常打开并给出提示"，
这正是用户第一次打开时最可能遇到的场景。
"""

from __future__ import annotations

from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = st_testing.AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


def _run_app(monkeypatch, tmp_path: Path):
    """把环境变量指到临时目录，保证测试不碰真实 data/ 目录。"""
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("EMBEDDING_BACKEND", "api")
    monkeypatch.setenv("EMBEDDING_API_KEY", "")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    app = AppTest.from_file(str(APP), default_timeout=60)
    app.run()
    return app


def test_app_boots_without_any_key(monkeypatch, tmp_path):
    app = _run_app(monkeypatch, tmp_path)

    assert not app.exception, "界面不应抛异常，缺 Key 时应友好提示：%s" % (
        [item.value for item in app.exception] if app.exception else ""
    )
    titles = [item.value for item in app.title]
    assert any("知识库问答" in title for title in titles)

    # 缺 Embedding Key 必须给出明确指引，而不是静默失败
    sidebar_text = " ".join(item.value for item in app.sidebar.markdown) + " ".join(
        item.value for item in app.sidebar.info
    )
    assert "EMBEDDING_API_KEY" in sidebar_text
    assert "DEEPSEEK_API_KEY" in sidebar_text

    # 标题栏两个区块都在（① 上传入库 / ② 提问）
    subheaders = " ".join(item.value for item in app.subheader)
    assert "上传" in subheaders
    assert "提问" in subheaders


def test_app_renders_document_stats(monkeypatch, tmp_path):
    app = _run_app(monkeypatch, tmp_path)
    assert not app.exception
    # 左侧栏两个指标：已入库文档 / 向量块数
    metrics = [item.value for item in app.sidebar.metric]
    assert "0" in metrics


def test_uploader_accepts_pdf_and_docx(monkeypatch, tmp_path):
    """上传控件必须同时接受 PDF 和 Word。"""
    app = _run_app(monkeypatch, tmp_path)
    uploaders = getattr(app, "file_uploader", None)
    if not uploaders:
        pytest.skip("当前 Streamlit 的 AppTest 不支持检查 file_uploader")
    label = uploaders[0].label
    assert "PDF" in label
    assert "Word" in label


def test_supported_suffixes_drive_upload_types():
    """允许的格式以 config.SUPPORTED_SUFFIXES 为唯一来源。"""
    from pdfrag.config import SUPPORTED_SUFFIXES

    assert SUPPORTED_SUFFIXES == (".pdf", ".docx")
    assert [suffix.lstrip(".") for suffix in SUPPORTED_SUFFIXES] == ["pdf", "docx"]
