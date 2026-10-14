"""第四步（上）的测试：检索与上下文拼装。"""

from __future__ import annotations

import pytest
from _fakes import FakeEmbedder, make_chunk, make_settings
from pdfrag.retriever import Retriever
from pdfrag.vector_store import Hit, VectorStore


def build_retriever(tmp_path, *, top_k=3):
    settings = make_settings(tmp_path)
    embedder = FakeEmbedder()
    store = VectorStore(settings, embedder=embedder)
    return Retriever(store, embedder, top_k=top_k), store, embedder


def make_hit(text, *, page=1, source="a.pdf", score=0.9):
    return Hit(text=text, score=score, distance=1 - score, doc_id="doc_x",
               source=source, page=page, chunk_index=0, chunk_id="doc_x-p%d-c0" % page)


def test_retrieve_uses_embedder_and_store(tmp_path):
    retriever, store, embedder = build_retriever(tmp_path)
    store.upsert_document([make_chunk(0, "苹果 维生素"), make_chunk(1, "编程 语言", page=2)])

    hits = retriever.retrieve("苹果")
    assert hits and "苹果" in hits[0].text

    hits = retriever.retrieve("苹果", top_k=1)
    assert len(hits) == 1, "top_k 参数必须生效"


def test_retrieve_empty_question_returns_nothing(tmp_path):
    retriever, store, _ = build_retriever(tmp_path)
    store.upsert_document([make_chunk(0, "内容")])
    assert retriever.retrieve("   ") == []


def test_build_context_numbers_sources():
    retriever = Retriever(store=None, embedder=None)
    hits = [make_hit("第一段内容", page=2, source="a.pdf", score=0.81),
            make_hit("第二段内容", page=7, source="b.pdf", score=0.42)]

    context, citations = retriever.build_context(hits)

    assert "[1] 来源：a.pdf 第 2 页" in context
    assert "[2] 来源：b.pdf 第 7 页" in context
    assert "相似度 0.810" in context
    assert [item.index for item in citations] == [1, 2]
    assert citations[0].label == "[1] a.pdf p.2"
    assert citations[0].to_dict()["score"] == 0.81


def test_build_context_can_hide_scores():
    retriever = Retriever(store=None, embedder=None)
    context, _ = retriever.build_context([make_hit("内容")], with_scores=False)
    assert "相似度" not in context


def test_build_context_deduplicates_identical_text():
    retriever = Retriever(store=None, embedder=None)
    duplicated = "完全一样的一段文字" * 3
    context, citations = retriever.build_context([make_hit(duplicated), make_hit(duplicated)])
    assert len(citations) == 1, "重复片段不应重复占用上下文预算"
    assert context.count("完全一样") == 3


def test_build_context_respects_char_budget():
    retriever = Retriever(store=None, embedder=None)
    hits = [make_hit("A" * 300, page=index) for index in range(1, 6)]
    context, citations = retriever.build_context(hits, max_chars=700)

    assert len(context) <= 700 + 200, "必须遵守字符预算"
    assert 0 < len(citations) < len(hits), "预算不够时应该只放得下前几块"
    assert citations[0].index == 1


def test_build_context_always_keeps_first_block():
    """即使第一块超长，也要放进去 —— 否则会给出一个空上下文。"""
    retriever = Retriever(store=None, embedder=None)
    context, citations = retriever.build_context([make_hit("B" * 5000)], max_chars=100)
    assert len(citations) == 1 and context.strip()


def test_build_context_skips_blank_hits():
    retriever = Retriever(store=None, embedder=None)
    context, citations = retriever.build_context([make_hit("   "), make_hit("有效内容")])
    assert len(citations) == 1
    assert "[1]" in context and "有效内容" in context


# -- 定位单位：DOCX 的引用要说"第 N 段" -------------------------------
def test_context_header_uses_section_wording_for_docx() -> None:
    retriever = Retriever(store=None, embedder=None)
    hit = Hit(text="Word 段落内容", score=0.77, distance=0.23, doc_id="doc_w",
              source="手册.docx", page=4, chunk_index=0,
              chunk_id="doc_w-u4-c0", unit="section")

    context, citations = retriever.build_context([hit])

    assert "[1] 来源：手册.docx 第 4 段" in context
    assert "第 4 页" not in context
    assert citations[0].location == "第 4 段"
    assert citations[0].label == "[1] 手册.docx 第4段"
    assert citations[0].to_dict()["unit"] == "section"
    assert "[1] 手册.docx 第4段 · 相似度" not in context   # 上下文里用"第 4 段"的写法


def test_pdf_hits_still_say_page() -> None:
    retriever = Retriever(store=None, embedder=None)
    hit = Hit(text="PDF 内容", score=0.5, distance=0.5, doc_id="doc_p",
              source="a.pdf", page=2, chunk_index=0, chunk_id="doc_p-u2-c0")
    context, citations = retriever.build_context([hit])
    assert "[1] 来源：a.pdf 第 2 页" in context
    assert citations[0].label == "[1] a.pdf p.2"
    assert citations[0].location == "第 2 页"
