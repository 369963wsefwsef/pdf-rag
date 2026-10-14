"""第四步的集成测试：ingest -> 检索 -> 生成答案，全离线。"""

from __future__ import annotations

from pathlib import Path

import pytest
from _fakes import FakeEmbedder, FakeLLM, make_settings
from _docx_builder import build_docx, heading, paragraph
from _pdf_builder import build_pdf
from pdfrag.config import UnsupportedFormatError
from pdfrag.docx_loader import DocxLoadError
from pdfrag.pipeline import RagPipeline
from pdfrag.pdf_loader import PdfLoadError

DOCX_BLOCKS = [
    heading("营养学基础", 1),
    paragraph("苹果富含维生素 C 和膳食纤维。"),
    heading("编程语言", 1),
    paragraph("Python 是常用的数据处理语言。"),
]


def make_docx(tmp_path: Path, name: str = "kb.docx") -> Path:
    return build_docx(tmp_path / name, DOCX_BLOCKS, doc_title="知识库")

PAGES = [
    ["Apple nutrition", "Apples are rich in vitamin C and dietary fiber."],
    ["Python basics", "Python is a popular programming language for data work."],
    ["Weather", "Today is sunny and warm."],
]


def make_pdf(tmp_path: Path, name: str = "kb.pdf") -> Path:
    return build_pdf(tmp_path / name, PAGES)


def build_pipeline(tmp_path, *, llm=None, embedder=None, **overrides):
    settings = make_settings(tmp_path, **overrides)
    active_embedder = embedder or FakeEmbedder()
    store = None
    pipeline = RagPipeline(settings, embedder=active_embedder, store=store, llm=llm)
    return pipeline


def test_ingest_reports_counts_and_stores_chunks(tmp_path):
    pipeline = build_pipeline(tmp_path)
    progress = []

    result = pipeline.ingest(make_pdf(tmp_path), on_progress=progress.append)

    assert result.pages == 3
    assert result.chunks >= 1
    assert result.chars > 0
    assert result.doc_id and len(result.doc_id) == 16
    assert "读取文件" in progress[0]
    assert "入库完成" in progress[-1]
    assert pipeline.store.count() == result.chunks
    assert [item["source"] for item in pipeline.documents()] == ["kb.pdf"]


def test_ingest_is_idempotent_for_same_file(tmp_path):
    pipeline = build_pipeline(tmp_path)
    first = pipeline.ingest(make_pdf(tmp_path))
    pipeline.ingest(make_pdf(tmp_path))          # 同一份内容再来一次
    assert pipeline.store.count() == first.chunks


def test_ingest_rejects_pdf_without_text(tmp_path):
    empty = build_pdf(tmp_path / "empty.pdf", [[]])
    pipeline = build_pipeline(tmp_path)
    with pytest.raises(PdfLoadError):
        pipeline.ingest(empty)


def test_search_returns_relevant_chunk(tmp_path):
    pipeline = build_pipeline(tmp_path)
    pipeline.ingest(make_pdf(tmp_path))

    hits = pipeline.search("vitamin C", top_k=2)
    assert hits, "应该能检索到内容"
    assert "vitamin" in hits[0].text.lower()


def test_answer_includes_retrieved_context_in_prompt(tmp_path):
    llm = FakeLLM(reply="苹果富含维生素 C。[1]")
    pipeline = build_pipeline(tmp_path, llm=llm)
    pipeline.ingest(make_pdf(tmp_path))

    result = pipeline.answer("苹果有什么营养？")

    assert result["answer"] == "苹果富含维生素 C。[1]"
    assert result["citations"], "应该带上引用"
    assert "[1] 来源：" in result["context"]
    assert "vitamin" in llm.last_user_content.lower(), "资料必须出现在发给大模型的 prompt 里"
    assert "苹果有什么营养" in llm.last_user_content
    assert "只依据" in llm.messages[0]["content"], "system prompt 必须约束只依据资料回答"


def test_answer_without_documents_reports_no_hits(tmp_path):
    pipeline = build_pipeline(tmp_path, llm=FakeLLM())
    result = pipeline.answer("随便问问")
    assert result["reason"] == "no_hits"
    assert result["answer"] == ""


def test_answer_stream_event_sequence(tmp_path):
    llm = FakeLLM(reply="答案文本")
    pipeline = build_pipeline(tmp_path, llm=llm)
    pipeline.ingest(make_pdf(tmp_path))

    events = list(pipeline.answer_stream("Python 是什么？"))

    assert events[0].type == "hits", "必须先给出检索结果，界面才能先显示引用"
    assert events[-1].type == "done"
    assert all(event.type in ("hits", "delta", "done") for event in events)

    answer = "".join(event.data.get("text", "") for event in events if event.type == "delta")
    assert answer == "答案文本"
    assert len(events[0].data["citations"]) >= 1
    assert events[0].data["hits"][0]["source"] == "kb.pdf"


def test_answer_stream_errors_are_events_not_exceptions(tmp_path):
    pipeline = build_pipeline(tmp_path, llm=FakeLLM())
    events = list(pipeline.answer_stream("没有任何文档时的问题"))
    assert len(events) == 1
    assert events[0].type == "error"
    assert "上传 PDF" in events[0].data["message"]


def test_answer_stream_blank_question(tmp_path):
    pipeline = build_pipeline(tmp_path, llm=FakeLLM())
    events = list(pipeline.answer_stream("   "))
    assert events[0].type == "error"
    assert "不能为空" in events[0].data["message"]


def test_answer_stream_reports_missing_llm_key(tmp_path):
    """检索能跑，但没配 DEEPSEEK_API_KEY 时要给出清晰提示，而不是抛异常。"""
    pipeline = build_pipeline(tmp_path, llm=None, deepseek_api_key="")
    pipeline.ingest(make_pdf(tmp_path))

    events = list(pipeline.answer_stream("Apple 讲了什么？"))
    types = [event.type for event in events]
    assert types[0] == "hits"
    assert types[-1] == "error"
    assert "DEEPSEEK_API_KEY" in events[-1].data["message"]


def test_status_without_keys_explains_what_is_missing(tmp_path):
    pipeline = build_pipeline(tmp_path, llm=None)
    pipeline._embedder = None            # 强制走"按配置构建"的路径：api + 空 key
    pipeline._store = None

    status = pipeline.status()

    assert status.embedding_ready is False
    assert status.llm_ready is False
    assert any("EMBEDDING_API_KEY" in message for message in status.messages)
    assert any("DEEPSEEK_API_KEY" in message for message in status.messages)
    assert status.to_dict()["documents"] == 0


def test_status_with_working_embedder(tmp_path):
    pipeline = build_pipeline(tmp_path, llm=None, deepseek_api_key="")
    pipeline.ingest(make_pdf(tmp_path))

    status = pipeline.status()
    assert status.embedding_ready is True
    assert status.chunks > 0
    assert status.documents == 1
    assert status.llm_ready is False
    assert status.can_answer is False


def test_delete_document(tmp_path):
    pipeline = build_pipeline(tmp_path)
    pipeline.ingest(make_pdf(tmp_path))
    doc_id = pipeline.documents()[0]["doc_id"]

    assert pipeline.delete_document(doc_id) >= 1
    assert pipeline.documents() == []
    assert pipeline.store.count() == 0


def test_ingest_many(tmp_path):
    pipeline = build_pipeline(tmp_path)
    one = build_pdf(tmp_path / "one.pdf", [["Alpha content here"]])
    two = build_pdf(tmp_path / "two.pdf", [["Beta content here"]])

    results = pipeline.ingest_many([one, two])
    assert len(results) == 2
    assert sorted(item["source"] for item in pipeline.documents()) == ["one.pdf", "two.pdf"]


def test_top_k_option_is_respected(tmp_path):
    pipeline = build_pipeline(tmp_path)
    pipeline.ingest(make_pdf(tmp_path))
    assert len(pipeline.search("content", top_k=1)) == 1


# ==================== Word(.docx) 支持 ====================
def test_ingest_docx_reports_counts_and_unit(tmp_path):
    pipeline = build_pipeline(tmp_path)
    progress = []

    result = pipeline.ingest(make_docx(tmp_path), on_progress=progress.append)

    assert result.unit == "section"
    assert result.source == "kb.docx"
    assert result.pages == 2                 # 两个标题块
    assert result.chunks >= 1
    assert result.chars > 0
    assert "读取文件" in progress[0]
    assert pipeline.store.count() == result.chunks
    assert [item["source"] for item in pipeline.documents()] == ["kb.docx"]
    assert result.to_dict()["unit"] == "section"


def test_docx_ingest_is_idempotent(tmp_path):
    pipeline = build_pipeline(tmp_path)
    first = pipeline.ingest(make_docx(tmp_path))
    pipeline.ingest(make_docx(tmp_path))      # 同一份内容再来一次
    assert pipeline.store.count() == first.chunks


def test_docx_citations_use_section_wording(tmp_path):
    llm = FakeLLM(reply="苹果富含维生素 C。[1]")
    pipeline = build_pipeline(tmp_path, llm=llm)
    pipeline.ingest(make_docx(tmp_path))

    hits = pipeline.search("维生素 C", top_k=2)
    assert hits and hits[0].unit == "section"
    assert hits[0].citation == "kb.docx 第%d段" % hits[0].page

    # answer() 返回的是 Citation 对象；answer_stream() 返回的才是 dict
    result = pipeline.answer("苹果有什么营养？")
    citation = result["citations"][0]
    assert "[1] 来源：kb.docx 第 %d 段" % citation.page in result["context"]
    assert citation.label == "[1] kb.docx 第%d段" % citation.page
    assert "维生素" in llm.last_user_content          # 资料确实进了 prompt


def test_docx_stream_events_carry_unit_in_dicts(tmp_path):
    """界面走的是 answer_stream，这条链路里的 unit 必须是 dict 字段。"""
    pipeline = build_pipeline(tmp_path, llm=FakeLLM())
    pipeline.ingest(make_docx(tmp_path))

    events = list(pipeline.answer_stream("维生素"))
    assert events[0].type == "hits"
    assert events[0].data["hits"][0]["unit"] == "section"
    citation = events[0].data["citations"][0]
    assert citation["unit"] == "section"
    assert citation["label"].endswith("段")


def test_pipeline_handles_pdf_and_docx_together(tmp_path):
    pipeline = build_pipeline(tmp_path)
    pipeline.ingest(make_pdf(tmp_path))
    pipeline.ingest(make_docx(tmp_path))

    sources = sorted(item["source"] for item in pipeline.documents())
    assert sources == ["kb.docx", "kb.pdf"]

    # 两种格式的引用写法互不干扰
    pdf_hit = pipeline.search("vitamin", top_k=1)[0]
    assert pdf_hit.source == "kb.pdf" and pdf_hit.unit == "page"
    assert pdf_hit.citation.startswith("kb.pdf p.")


def test_unsupported_suffix_is_rejected_with_helpful_message(tmp_path):
    pipeline = build_pipeline(tmp_path)
    note = tmp_path / "note.txt"
    note.write_text("纯文本文件", encoding="utf-8")

    with pytest.raises(UnsupportedFormatError, match="不支持的文件格式"):
        pipeline.ingest(note)
    with pytest.raises(UnsupportedFormatError, match="\\.pdf"):
        pipeline.ingest(note)


def test_legacy_doc_gives_actionable_error(tmp_path):
    pipeline = build_pipeline(tmp_path)
    legacy = tmp_path / "old.doc"
    legacy.write_bytes(b"\xd0\xcf\x11\xe0" + b"\x00" * 32)

    with pytest.raises(DocxLoadError, match="另存为"):
        pipeline.ingest(legacy)


# ==================== 网络代理 ====================
def test_status_shows_active_proxy(tmp_path):
    """代理最容易配错（端口写错就是 10061），所以状态里必须能看到最终生效的代理。"""
    from pdfrag.net import ProxyConfig

    pipeline = build_pipeline(tmp_path, proxy=ProxyConfig(https="http://127.0.0.1:7897", source="env"))
    status = pipeline.status()

    assert any("7897" in message for message in status.messages)
    assert any(message.startswith("网络：") for message in status.messages)


def test_status_shows_proxy_off_and_warning(tmp_path):
    from pdfrag.net import ProxyConfig

    off = build_pipeline(tmp_path, proxy=ProxyConfig(enabled=False, source="off"))
    assert any("已关闭" in message for message in off.status().messages)

    bad = build_pipeline(tmp_path, proxy=ProxyConfig(enabled=False, source="invalid", warning="PDFRAG_PROXY 无法识别"))
    assert any("无法识别" in message for message in bad.status().messages)


def test_answer_stream_works_with_proxy_configured(tmp_path):
    """配了代理也不该影响业务逻辑本身。"""
    from pdfrag.net import ProxyConfig

    pipeline = build_pipeline(tmp_path, llm=FakeLLM(reply="答案 [1]"),
                              proxy=ProxyConfig(https="http://127.0.0.1:7897", source="env"))
    pipeline.ingest(make_pdf(tmp_path))
    events = list(pipeline.answer_stream("vitamin"))
    assert events[0].type == "hits"
    assert events[-1].type == "done"
