"""第一步的测试：PDF -> 带页码的文本。"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from _pdf_builder import build_pdf
from pdfrag.pdf_loader import (
    PdfLoadError,
    clean_text,
    compute_doc_id,
    iter_pdf_paths,
    load_pdf,
    merge_wrapped_lines,
)

SAMPLE_PAGES = [
    ["Introduction", "This is page one.", "Hyphen-", "ation works."],
    ["Chapter Two", "This is page two."],
    ["Chapter Three", "This is page three."],
]


def make_pdf(tmp_path: Path, pages=SAMPLE_PAGES) -> Path:
    return build_pdf(tmp_path / "sample.pdf", pages)


# -- 正常路径 ------------------------------------------------------
def test_load_pdf_keeps_page_numbers(tmp_path: Path) -> None:
    loaded = load_pdf(make_pdf(tmp_path))

    assert loaded.page_count == 3
    assert [page.page for page in loaded.pages] == [1, 2, 3]
    assert "page one" in loaded.pages[0].text
    assert "page three" in loaded.pages[2].text
    assert loaded.source == "sample.pdf"
    # PDF 里没写 /Title，应回退到文件名
    assert loaded.title == "sample"
    assert loaded.char_count() > 0
    assert loaded.full_text().count("\n\n") >= 2


def test_doc_id_is_content_hash_not_filename(tmp_path: Path) -> None:
    pdf = make_pdf(tmp_path)
    loaded = load_pdf(pdf)
    assert loaded.doc_id == compute_doc_id(pdf)

    # 换个文件名、内容相同 -> 同一个 doc_id（幂等入库的基础）
    copy = tmp_path / "renamed.pdf"
    shutil.copyfile(pdf, copy)
    assert load_pdf(copy).doc_id == loaded.doc_id

    # 内容变一个字 -> doc_id 变化
    changed = build_pdf(tmp_path / "changed.pdf", [["Introduction"], ["different"]])
    assert load_pdf(changed).doc_id != loaded.doc_id


def test_multiple_pdfs_are_listed_in_stable_order(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    build_pdf(tmp_path / "b.pdf", [["B"]])
    build_pdf(tmp_path / "a.pdf", [["A"]])
    build_pdf(nested / "c.pdf", [["C"]])
    (tmp_path / "not-a-pdf.txt").write_text("ignore me", encoding="utf-8")

    found = iter_pdf_paths(tmp_path)
    assert [item.name for item in found] == ["a.pdf", "b.pdf", "c.pdf"]
    assert iter_pdf_paths(tmp_path / "missing") == []


# -- 异常路径 ------------------------------------------------------
def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(PdfLoadError, match="文件不存在"):
        load_pdf(tmp_path / "nope.pdf")


def test_corrupt_file(tmp_path: Path) -> None:
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"this is definitely not a pdf file")
    with pytest.raises(PdfLoadError):
        load_pdf(broken)


def test_scanned_pdf_reports_helpful_error(tmp_path: Path) -> None:
    # 有合法 PDF 结构但没有任何文字（近似"扫描件"）
    empty = build_pdf(tmp_path / "empty.pdf", [[]])
    with pytest.raises(PdfLoadError, match="OCR"):
        load_pdf(empty)
    # 允许空文本时可以放行，方便批量导入跳过坏文件
    assert load_pdf(empty, allow_empty=True).page_count == 1


# -- 文本清洗（中文为主，不依赖 PDF） --------------------------------
def test_clean_text_merges_cjk_lines_without_space() -> None:
    assert clean_text("第一行\n第二行") == "第一行第二行"


def test_clean_text_merges_english_lines_with_space() -> None:
    assert clean_text("hello\nworld") == "hello world"


def test_clean_text_keeps_paragraph_breaks() -> None:
    assert clean_text("第一段\n\n\n\n第二段") == "第一段\n\n第二段"


def test_clean_text_fixes_hyphenation() -> None:
    assert clean_text("inter-\nnational") == "international"


def test_clean_text_collapses_spaces_and_strips() -> None:
    assert clean_text("  a   b  \n  c  ") == "a b c"
    assert clean_text("") == ""
    assert clean_text("   \n  ") == ""


def test_merge_wrapped_lines_handles_blank_lines() -> None:
    assert merge_wrapped_lines("甲\n\n乙") == "甲\n\n乙"


def test_mixed_language_boundary_gets_space() -> None:
    # 中文 + 英文混排时用空格分隔，避免粘成一个词
    assert clean_text("使用 Python\n写脚本") == "使用 Python 写脚本"
