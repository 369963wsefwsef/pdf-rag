"""提问流程的回归测试（本次修复的重点）。

用 AppTest 跑真实的 app.py，但把 RagPipeline 换成假实现，
所以不需要网络、不需要任何 API Key。

要锁住的 bug：答案只活在"点击提问那一次运行"的局部变量里，
一旦脚本重跑（或流式还没结束又提交了一次），页面上就会残留一堆
带流式光标（▌）的半截句子。修法是：历史存 session_state + chat_input
+ write_stream，页面上永远只渲染完整结果。
"""

from __future__ import annotations

from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = st_testing.AppTest

from pdfrag.pipeline import AnswerEvent, PipelineStatus   # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app.py"
CURSOR = "▌"

ANSWER_PIECES = ["本文主要围绕", "智能停车场管理系统展开", "，包含五个阶段。"]

HITS = [
    {"text": "本文围绕智能停车场管理系统展开，包含需求分析与系统测试等阶段。",
     "score": 0.71, "distance": 0.29, "doc_id": "doc_x", "source": "毕业论文.docx",
     "page": 3, "chunk_index": 4, "chunk_id": "doc_x-u3-c4", "unit": "section",
     "citation": "毕业论文.docx 第3段"},
]
CITATIONS = [
    {"index": 1, "source": "毕业论文.docx", "page": 3, "score": 0.71, "doc_id": "doc_x",
     "chunk_id": "doc_x-u3-c4", "unit": "section", "label": "[1] 毕业论文.docx 第3段"},
]


class FakePipeline:
    """替换掉真 pipeline：检索与生成都走假数据。"""

    def __init__(self, settings=None) -> None:
        self.settings = settings

    def status(self) -> PipelineStatus:
        return PipelineStatus(embedding_ready=True, llm_ready=True, documents=1,
                              chunks=3, messages=[], embedding_name="fake", embedding_dim=64)

    def documents(self):
        return [{"doc_id": "doc_x", "source": "毕业论文.docx", "chunks": 3, "pages": 2}]

    def answer_stream(self, question, **kwargs):
        yield AnswerEvent("hits", {"hits": HITS, "citations": CITATIONS, "context": "..."})
        for piece in ANSWER_PIECES:
            yield AnswerEvent("delta", {"text": piece})
        yield AnswerEvent("done", {})


class FailingPipeline(FakePipeline):
    def answer_stream(self, question, **kwargs):
        yield AnswerEvent("error", {"message": "知识库里没有找到相关内容，请先上传 PDF。"})


def _all_text(app) -> str:
    """把页面上所有可见文字收集起来（含 st.error 渲染的错误）。"""
    parts = [item.value for item in app.markdown]
    parts += [item.value for item in app.caption]
    parts += [item.value for item in getattr(app, "error", [])]
    parts += [item.label for item in getattr(app, "expander", [])]
    return "\n".join(parts)


def _boot(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("EMBEDDING_BACKEND", "api")
    monkeypatch.setenv("EMBEDDING_API_KEY", "sk-fake")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-fake")
    monkeypatch.setattr("pdfrag.pipeline.RagPipeline", FakePipeline)
    app = AppTest.from_file(str(APP), default_timeout=60)
    app.run()
    return app


def test_ask_flow_renders_full_answer_without_leftover_cursor(monkeypatch, tmp_path):
    app = _boot(monkeypatch, tmp_path)
    assert not app.exception, [item.value for item in app.exception]

    assert app.chat_input, "提问区应该使用 st.chat_input"
    app.chat_input[0].set_value("这篇论文主要讲了什么？").run()
    assert not app.exception, [item.value for item in app.exception]

    text = _all_text(app)
    assert "".join(ANSWER_PIECES) in text, "完整答案必须渲染出来"
    assert "这篇论文主要讲了什么？" in text, "提问也要显示在对话里"
    assert CURSOR not in text, "收尾后页面上不能残留流式光标"
    assert "[1] 毕业论文.docx 第3段" in text, "引用来源要跟着答案一起显示"


def test_answer_survives_rerun_and_never_shows_cursor(monkeypatch, tmp_path):
    """再重跑一次脚本（模拟点按钮/刷新），答案要从 session_state 重绘且不含光标。"""
    app = _boot(monkeypatch, tmp_path)
    app.chat_input[0].set_value("主要内容？").run()
    assert not app.exception

    app.run()                      # 不带新问题，纯粹重跑
    assert not app.exception

    text = _all_text(app)
    assert "".join(ANSWER_PIECES) in text, "重跑后答案不能丢"
    assert CURSOR not in text
    assert "[1] 毕业论文.docx 第3段" in text, "重跑后引用也要还在"


def test_second_question_keeps_both_rounds(monkeypatch, tmp_path):
    app = _boot(monkeypatch, tmp_path)
    app.chat_input[0].set_value("第一个问题").run()
    app.chat_input[0].set_value("第二个问题").run()
    assert not app.exception

    text = _all_text(app)
    assert "第一个问题" in text and "第二个问题" in text
    assert text.count("".join(ANSWER_PIECES)) == 2, "两轮回答都应保留"
    assert CURSOR not in text


def test_error_event_is_shown_and_survives_rerun(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("EMBEDDING_BACKEND", "api")
    monkeypatch.setenv("EMBEDDING_API_KEY", "sk-fake")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-fake")
    monkeypatch.setattr("pdfrag.pipeline.RagPipeline", FailingPipeline)

    app = AppTest.from_file(str(APP), default_timeout=60)
    app.run()
    app.chat_input[0].set_value("随便问问").run()
    assert not app.exception

    assert "知识库里没有找到相关内容" in _all_text(app)
    app.run()
    assert "知识库里没有找到相关内容" in _all_text(app), "重跑后错误提示也要还在"
    assert CURSOR not in _all_text(app)
