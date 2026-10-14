"""第二步的测试：文本 -> 带重叠、带页码的 chunk。"""

from __future__ import annotations

from pathlib import Path

import pytest
from pdfrag.chunker import chunk_pages, split_text, summarize
from _docx_builder import build_docx, heading, paragraph
from pdfrag.docx_loader import load_docx
from pdfrag.pdf_loader import LoadedPdf, PdfPage


def make_loaded(pages, doc_id: str = "doc123", source: str = "doc.pdf") -> LoadedPdf:
    return LoadedPdf(
        path=Path(source),
        doc_id=doc_id,
        title="doc",
        page_count=len(pages),
        pages=[PdfPage(page=index, text=text) for index, text in enumerate(pages, start=1)],
    )


# -- split_text 的硬性保证 ------------------------------------------
def test_every_span_respects_max_size() -> None:
    text = "。".join("这是第%d句话" % i for i in range(500))
    spans = split_text(text, size=100, overlap=20)

    assert spans[0][0] == 0
    assert spans[-1][1] == len(text)
    assert all(end - start <= 100 for start, end in spans)


def test_adjacent_spans_overlap_and_always_advance() -> None:
    spans = split_text("A" * 500, size=100, overlap=20)

    assert len(spans) > 1
    for (start_a, end_a), (start_b, end_b) in zip(spans, spans[1:]):
        assert start_b > start_a, "必须严格前进，否则死循环"
        assert start_b < end_a, "必须有重叠"
        assert end_b > end_a


def test_spans_cover_text_without_holes() -> None:
    text = "".join("句子%d。" % i for i in range(300))
    spans = split_text(text, size=64, overlap=16)

    assert spans[0][0] == 0
    assert spans[-1][1] == len(text)
    for (_, end_a), (start_b, _) in zip(spans, spans[1:]):
        assert start_b <= end_a, "不允许出现空洞"


def test_hard_split_when_there_is_no_boundary_at_all() -> None:
    spans = split_text("x" * 250, size=100, overlap=10)

    assert all(end - start <= 100 for start, end in spans)
    assert spans[-1][1] == 250


def test_short_text_produces_single_span() -> None:
    assert split_text("很短的一段话。", size=800, overlap=150) == [(0, 7)]
    assert split_text("", size=800, overlap=150) == []


def test_invalid_parameters_are_rejected() -> None:
    with pytest.raises(ValueError):
        split_text("abc", size=0, overlap=0)
    with pytest.raises(ValueError):
        split_text("abc", size=100, overlap=100)
    with pytest.raises(ValueError):
        split_text("abc", size=100, overlap=-1)


def test_prefers_sentence_boundary_over_hard_cut() -> None:
    text = "第一句话。" * 40          # 句末标点很多
    spans = split_text(text, size=50, overlap=10)

    # 切点应尽量落在句号之后
    for start, end in spans[:-1]:
        assert text[end - 1] in "。！？；; \n"


# -- chunk_pages 的元数据 -------------------------------------------
def test_chunks_carry_page_number_and_source() -> None:
    loaded = make_loaded(["第一页内容。" * 60, "第二页内容。" * 60])
    chunks = chunk_pages(loaded, size=100, overlap=20)

    assert {chunk.page for chunk in chunks} == {1, 2}
    assert chunks[0].page == 1
    assert chunks[-1].page == 2
    assert all(chunk.source == "doc.pdf" for chunk in chunks)
    assert all(chunk.doc_id == "doc123" for chunk in chunks)


def test_chunk_text_never_exceeds_size() -> None:
    loaded = make_loaded(["段落。" * 400])
    chunks = chunk_pages(loaded, size=120, overlap=30)
    assert all(len(chunk.text) <= 120 for chunk in chunks)


def test_ids_are_unique_and_reproducible() -> None:
    loaded = make_loaded(["你好世界。" * 100])
    first = chunk_pages(loaded, size=100, overlap=20)
    second = chunk_pages(loaded, size=100, overlap=20)

    assert [chunk.id for chunk in first] == [chunk.id for chunk in second]
    assert len({chunk.id for chunk in first}) == len(first)
    assert first[0].id.startswith("doc123-u1-c0")


def test_empty_or_blank_pages_yield_no_chunks() -> None:
    loaded = make_loaded(["", "   \n   ", "\n\n"])
    assert chunk_pages(loaded, size=100, overlap=20) == []


def test_tiny_tail_fragment_is_dropped() -> None:
    loaded = make_loaded(["啊" * 205])          # 切成 100 + 100 + 5
    chunks = chunk_pages(loaded, size=100, overlap=0, min_chunk_chars=20)
    assert len(chunks) == 2, "5 个字符的碎块应被丢弃"


def test_short_page_is_kept_instead_of_dropped() -> None:
    """整页都很短时也必须保留一块 —— 否则短文档上传后什么都检索不到。"""
    loaded = make_loaded(["短页面。"])
    chunks = chunk_pages(loaded, size=100, overlap=20)
    assert len(chunks) == 1
    assert chunks[0].text == "短页面。"
    assert chunks[0].page == 1


def test_short_page_does_not_affect_normal_pages() -> None:
    loaded = make_loaded(["标题", "正常长度的正文内容。" * 20])
    chunks = chunk_pages(loaded, size=100, overlap=20)
    assert chunks[0].page == 1 and chunks[0].text == "标题"
    assert any(chunk.page == 2 for chunk in chunks)


def test_metadata_is_flat_and_typed() -> None:
    loaded = make_loaded(["内容" * 100])
    chunk = chunk_pages(loaded, size=100, overlap=20)[0]
    metadata = chunk.to_metadata()

    assert set(metadata) == {"doc_id", "source", "unit", "page", "chunk_index",
                             "char_start", "char_end"}
    assert all(isinstance(value, (str, int, float, bool)) for value in metadata.values())
    assert chunk.citation == "doc.pdf p.1"


def test_summarize() -> None:
    loaded = make_loaded(["内容" * 200])
    stats = summarize(chunk_pages(loaded, size=100, overlap=20))

    assert stats["count"] > 0
    assert stats["max_chars"] <= 100
    assert stats["pages"] == 1
    assert summarize([])["count"] == 0


# -- 定位单位（unit）：PDF 是"页"，DOCX 是"段" -------------------------
def test_pdf_chunks_default_to_page_unit() -> None:
    loaded = make_loaded(["第一页内容。" * 40])
    chunk = chunk_pages(loaded, size=100, overlap=20)[0]
    assert chunk.unit == "page"
    assert chunk.unit_name == "页"
    assert chunk.citation == "doc.pdf p.1"
    assert chunk.to_metadata()["unit"] == "page"


def test_docx_chunks_carry_section_unit(tmp_path) -> None:
    """真实走一遍 load_docx，确认 chunker 认得出 DocxSection 的序号。"""
    path = build_docx(tmp_path / "kb.docx", [
        heading("第一章", 1),
        paragraph("第一章的正文内容。" * 20),
        heading("第二章", 1),
        paragraph("第二章的正文内容。" * 20),
    ])
    loaded = load_docx(path)
    chunks = chunk_pages(loaded, size=200, overlap=40)

    assert {chunk.unit for chunk in chunks} == {"section"}
    assert {chunk.page for chunk in chunks} >= {1, 2}
    assert all(chunk.citation.startswith("kb.docx 第") for chunk in chunks)
    assert all(chunk.citation.endswith("段") for chunk in chunks)
    assert all(chunk.id.startswith(loaded.doc_id + "-u") for chunk in chunks)


def test_chunker_accepts_any_object_with_pages_and_unit() -> None:
    """只要满足 LoadedDocumentLike 协议就能切块（便于将来加别的格式）。"""
    from dataclasses import dataclass

    @dataclass
    class Unit:
        index: int
        text: str

    @dataclass
    class FakeDoc:
        doc_id: str
        source: str
        unit: str
        pages: list

    doc = FakeDoc(doc_id="doc_x", source="x.md", unit="section",
                  pages=[Unit(index=1, text="内容" * 60)])
    chunks = chunk_pages(doc, size=100, overlap=20)
    assert chunks and chunks[0].page == 1 and chunks[0].unit == "section"
