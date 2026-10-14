"""Word 加载器的测试。

全部使用标准库现场拼出的假 .docx（tests/_docx_builder.py）：
不联网、不需要安装 python-docx、也不需要任何真实 Word 文件。
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from _docx_builder import build_docx, build_raw_zip, heading, paragraph, table
from pdfrag.docx_loader import (
    DocxLoadError,
    compute_doc_id,
    iter_docx_paths,
    load_docx,
)

WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def basic_docx(tmp_path: Path, name: str = "manual.docx", **kwargs) -> Path:
    return build_docx(
        tmp_path / name,
        [
            heading("第一章 概述", 1),
            paragraph("这是第一章的正文段落。"),
            heading("第二章 安装", 2),
            paragraph("先安装依赖，再运行程序。"),
        ],
        doc_title="产品手册",
        **kwargs,
    )


def custom_docx(tmp_path: Path, body_xml: str, name: str = "custom.docx") -> Path:
    xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="%s"><w:body>%s<w:sectPr/></w:body></w:document>'
        % (WORD_NS, body_xml)
    )
    return build_docx(tmp_path / name, [], document_xml=xml)


# ---------- 段落结构 ----------
def test_sections_group_heading_with_its_paragraphs(tmp_path: Path) -> None:
    loaded = load_docx(basic_docx(tmp_path))

    assert loaded.unit == "section"
    assert loaded.section_count == 2

    first, second = loaded.pages
    assert first.index == 1
    assert first.heading == "第一章 概述"
    assert first.heading_level == 1
    # 标题 + 它下面的正文合成为一块，这样检索命中时自带"这是哪一节"的上下文
    assert first.text == "第一章 概述\n这是第一章的正文段落。"

    assert second.index == 2
    assert second.heading == "第二章 安装"
    assert second.heading_level == 2
    assert "第二章 安装" in second.text


def test_paragraph_line_breaks_are_kept(tmp_path: Path) -> None:
    loaded = load_docx(build_docx(tmp_path / "br.docx", [paragraph("第一行\n第二行")]))
    assert loaded.pages[0].text == "第一行\n第二行"


def test_blank_paragraphs_are_skipped(tmp_path: Path) -> None:
    path = build_docx(tmp_path / "blank.docx", [
        paragraph("第一段"),
        {"type": "blank"},
        paragraph("第二段"),
    ])
    loaded = load_docx(path)
    assert loaded.section_count == 1
    assert loaded.pages[0].text == "第一段\n第二段"


def test_whitespace_and_nbsp_are_normalized(tmp_path: Path) -> None:
    path = build_docx(tmp_path / "ws.docx", [
        paragraph("你好\u00a0\u00a0世界\u3000\u3000你好"),
        paragraph("   前后有空格   "),
    ])
    loaded = load_docx(path)
    assert loaded.pages[0].text == "你好 世界 你好\n前后有空格"


def test_table_becomes_its_own_section(tmp_path: Path) -> None:
    path = build_docx(tmp_path / "table.docx", [
        paragraph("表格之前"),
        table([["名称", "说明"], ["A", "第一项"]]),
        paragraph("表格之后"),
    ])
    loaded = load_docx(path)
    assert [item.text for item in loaded.pages] == [
        "表格之前",
        "名称 | 说明\nA | 第一项",
        "表格之后",
    ]


def test_headings_detected_by_outline_level_only(tmp_path: Path) -> None:
    """只写了 outlineLvl、没有 Heading 样式时也要认出来（中文模板常见）。"""
    body = (
        '<w:p><w:pPr><w:outlineLvl w:val="1"/></w:pPr><w:r><w:t>第二章</w:t></w:r></w:p>'
        "<w:p><w:r><w:t>正文</w:t></w:r></w:p>"
    )
    loaded = load_docx(custom_docx(tmp_path, body))
    assert loaded.pages[0].heading == "第二章"
    assert loaded.pages[0].heading_level == 2      # outlineLvl=1 对应 2 级标题
    assert loaded.pages[0].text == "第二章\n正文"


def test_chinese_heading_style_id_is_recognized(tmp_path: Path) -> None:
    body = (
        '<w:p><w:pPr><w:pStyle w:val="标题1"/></w:pPr><w:r><w:t>中文样式标题</w:t></w:r></w:p>'
        "<w:p><w:r><w:t>内容</w:t></w:r></w:p>"
    )
    loaded = load_docx(custom_docx(tmp_path, body))
    assert loaded.pages[0].heading_level == 1
    assert loaded.pages[0].heading == "中文样式标题"


def test_long_document_without_headings_gets_bounded_sections(tmp_path: Path) -> None:
    """无标题长文档不能整篇挤成"第 1 段"，要按长度切出多个定位单元。"""
    blocks = [paragraph("第%d段落内容。" % index * 3) for index in range(12)]
    path = build_docx(tmp_path / "long.docx", blocks)

    loaded = load_docx(path, max_section_chars=60)
    assert loaded.section_count > 1
    assert [item.index for item in loaded.pages] == list(range(1, loaded.section_count + 1))
    assert all(item.heading == "" for item in loaded.pages)
    assert sum(len(item.text) for item in loaded.pages) > 100


# ---------- 标题与元数据 ----------
def test_title_comes_from_core_properties(tmp_path: Path) -> None:
    assert load_docx(basic_docx(tmp_path)).title == "产品手册"


def test_title_falls_back_to_first_heading(tmp_path: Path) -> None:
    path = build_docx(tmp_path / "no-core-title.docx", [
        heading("第一张说明书", 1),
        paragraph("正文"),
    ], doc_title="")
    assert load_docx(path).title == "第一张说明书"


def test_title_falls_back_to_filename(tmp_path: Path) -> None:
    path = build_docx(tmp_path / "无标题文档.docx", [paragraph("正文")], with_core=False)
    assert load_docx(path).title == "无标题文档"


def test_loaded_helpers_expose_sections(tmp_path: Path) -> None:
    loaded = load_docx(basic_docx(tmp_path))
    assert loaded.sections is loaded.pages
    assert loaded.section_count == loaded.page_count == 2
    assert loaded.source == "manual.docx"
    assert loaded.metadata["sections"] == 2
    assert loaded.char_count() == sum(len(item.text) for item in loaded.pages)
    assert "第一章 概述" in loaded.full_text()
    assert loaded.empty_pages == []


def test_doc_id_is_content_hash(tmp_path: Path) -> None:
    first = basic_docx(tmp_path, "a.docx")
    second = basic_docx(tmp_path, "b.docx")        # 内容相同、文件名不同
    assert load_docx(first).doc_id == load_docx(second).doc_id == compute_doc_id(first)

    renamed = tmp_path / "renamed.docx"
    shutil.copyfile(first, renamed)
    assert load_docx(renamed).doc_id == load_docx(first).doc_id

    different = build_docx(tmp_path / "d.docx", [paragraph("完全不同的内容")])
    assert load_docx(different).doc_id != load_docx(first).doc_id


def test_iter_docx_paths(tmp_path: Path) -> None:
    build_docx(tmp_path / "b.docx", [paragraph("B")])
    build_docx(tmp_path / "a.docx", [paragraph("A")])
    build_docx(tmp_path / "sub" / "c.docx", [paragraph("C")])
    (tmp_path / "note.txt").write_text("x", encoding="utf-8")

    assert [item.name for item in iter_docx_paths(tmp_path)] == ["a.docx", "b.docx", "c.docx"]
    assert iter_docx_paths(tmp_path / "missing") == []


# ---------- 异常路径 ----------
def test_empty_document_raises(tmp_path: Path) -> None:
    path = build_docx(tmp_path / "empty.docx", [])
    with pytest.raises(DocxLoadError, match="抽出任何文字"):
        load_docx(path)
    assert load_docx(path, allow_empty=True).section_count == 0


def test_legacy_doc_raises_with_hint(tmp_path: Path) -> None:
    legacy = tmp_path / "old.doc"
    legacy.write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 64)
    with pytest.raises(DocxLoadError, match="另存为"):
        load_docx(legacy)


def test_non_zip_raises(tmp_path: Path) -> None:
    fake = tmp_path / "fake.docx"
    fake.write_text("这其实是一个文本文件", encoding="utf-8")
    with pytest.raises(DocxLoadError, match="不是有效的 .docx"):
        load_docx(fake)


def test_zip_without_document_xml_raises(tmp_path: Path) -> None:
    path = build_raw_zip(tmp_path / "nope.docx", {"[Content_Types].xml": "<Types/>"})
    with pytest.raises(DocxLoadError, match="document.xml"):
        load_docx(path)


def test_broken_xml_raises(tmp_path: Path) -> None:
    path = custom_docx(tmp_path, "<w:p><w:r><w:t>没闭合", name="broken.docx")
    with pytest.raises(DocxLoadError, match="XML 解析失败"):
        load_docx(path)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(DocxLoadError, match="文件不存在"):
        load_docx(tmp_path / "nope.docx")
