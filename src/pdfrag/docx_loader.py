"""第一步（Word 版）：把 .docx 读成「带段落编号的纯文本」。

为什么不用 python-docx
----------------------
.docx 本质是一个 ZIP 包，正文就在 word/document.xml 里。这里用标准库
zipfile + xml.etree 只解析需要的部分（段落 / 标题 / 表格），好处是：

* 零新增依赖 —— 装上 pypdf 就够了，不需要再装 python-docx；
* 测试完全离线，fake 数据由 tests/_docx_builder.py 用标准库现场拼出来，
  和 tests/_pdf_builder.py 手写 PDF 是同一个思路。

代价是不处理页眉页脚、脚注、批注、修订痕迹、文本框这些边角内容 —— 它们对
"检索正文回答问题"没有影响。真需要时可以换成 python-docx，接口不用变。

与 PDF 的关键差异：没有页码
--------------------------
Word 是流式文档，**没有可靠的分页概念**（同一份文档换台打印机页数都可能变），
所以定位单位用"段落块"：LoadedDocx.unit = "section"，
引用会显示成「手册.docx 第 12 段」，而不是一个假的页码。

段落结构怎么保留
----------------
一个 section = 一个标题 + 它下面的正文段落（直到下一个标题）。这样标题天然成为
该块内容的一部分，检索命中时能带上"这段属于哪一节"的上下文。
表格单独成一个 section，行内用 " | " 分隔单元格。
"""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from xml.etree import ElementTree as ET

from .pdf_loader import compute_doc_id   # 内容哈希是通用逻辑，直接复用

DOCX_SUFFIX = ".docx"
LEGACY_DOC_SUFFIX = ".doc"

_W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_DC_NS = "http://purl.org/dc/elements/1.1/"
_NS = {"w": _W_NS}
_RE_HEADING_STYLE = re.compile(r"^(?:heading|标题)\s*([1-9])$", re.IGNORECASE)
_RE_SPACES = re.compile(r"[ \t\u00a0\u3000]+")

# 单个段落块的字数上限。没有标题的文档如果整篇挤成一块，引用会永远是"第 1 段"，
# 所以超长时按长度再切开，保证"段"这个定位单位始终有意义。
DEFAULT_MAX_SECTION_CHARS = 1200


class DocxLoadError(RuntimeError):
    """Word 读取失败：文件不存在、不是有效的 .docx、抽不出文字等。"""


def legacy_doc_message(name: str) -> str:
    """老式 .doc 的可操作提示。

    放在这里是为了让 pipeline 也能复用同一句话 —— .doc 在 pipeline 的
    后缀分支就会被拦下，如果不复用，用户只能看到一句干巴巴的"格式不支持"。
    """
    return (
        "%s 是老式的 .doc 二进制格式，和 .docx 完全是两种东西，无法直接解析。"
        "请用 Word 打开后「另存为」.docx 再上传。" % name
    )


@dataclass(frozen=True)
class DocxSection:
    """一个段落块：一个标题 + 它下面的正文（或一张表格）。"""

    index: int              # 1-based 段块序号，用于引用（"第 N 段"）
    text: str
    heading: str = ""       # 本块所属标题（没有标题时为空字符串）
    heading_level: int = 0  # 0 表示不是标题块


@dataclass
class LoadedDocx:
    """Word 的加载结果。

    属性名与 LoadedPdf 保持一致（pages / page_count / empty_pages ...），
    这样 chunker 不需要为两种格式写两套逻辑；语义上它们是"定位单元"，
    DOCX 里一个 unit 就是一个段落块，单位由 unit 字段标明。
    """

    path: Path
    doc_id: str
    title: str
    page_count: int
    pages: List[DocxSection]
    unit: str = "section"
    metadata: Dict[str, Any] = field(default_factory=dict)
    failed_pages: List[int] = field(default_factory=list)

    @property
    def source(self) -> str:
        return self.path.name

    @property
    def sections(self) -> List[DocxSection]:
        return self.pages

    @property
    def section_count(self) -> int:
        return len(self.pages)

    @property
    def empty_pages(self) -> List[int]:
        return [section.index for section in self.pages if not section.text.strip()]

    def full_text(self) -> str:
        return "\n\n".join(section.text for section in self.pages if section.text.strip())

    def char_count(self) -> int:
        return sum(len(section.text) for section in self.pages)


def _localname(tag: str) -> str:
    """去掉命名空间前缀，只要标签本地名（w:p -> p）。"""
    return tag.rsplit("}", 1)[-1]


def _normalize(text: str) -> str:
    """Word 里常见的不换行空格/多余空格统一掉，段落内部的换行保留。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [_RE_SPACES.sub(" ", line).strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def _paragraph_text(paragraph: ET.Element) -> str:
    """拼出一个 <w:p> 的纯文本。

    用 iter() 遍历所有后代，所以超链接（w:hyperlink）和修订插入（w:ins）里的
    文字都能取到。w:tab 换成空格，w:br / w:cr 换成换行。
    """
    parts: List[str] = []
    for node in paragraph.iter():
        name = _localname(node.tag)
        if name == "t":
            parts.append(node.text or "")
        elif name == "tab":
            parts.append(" ")
        elif name in ("br", "cr"):
            parts.append("\n")
    return "".join(parts)


def _heading_level(paragraph: ET.Element) -> Optional[int]:
    """判断段落是否为标题并返回级别（1~9），不是标题返回 None。

    两条线索都看：pStyle（Heading1 / 标题1）和 outlineLvl（Word 给标题设置的大纲级别）。
    中文模板里 pStyle 的 id 有时就是 "1"、"2"，只靠样式名会漏判，所以用 outlineLvl 兜底。
    """
    properties = paragraph.find("w:pPr", _NS)
    if properties is None:
        return None

    style = properties.find("w:pStyle", _NS)
    if style is not None:
        value = (style.get("{%s}val" % _W_NS) or "").strip()
        if value.lower() in ("title", "标题"):
            return 1
        matched = _RE_HEADING_STYLE.match(value)
        if matched:
            return int(matched.group(1))

    outline = properties.find("w:outlineLvl", _NS)
    if outline is not None:
        raw = (outline.get("{%s}val" % _W_NS) or "").strip()
        if raw.isdigit() and 0 <= int(raw) <= 8:
            return int(raw) + 1
    return None


def _table_text(table: ET.Element) -> str:
    """把表格渲染成纯文本，行内用 " | " 分隔单元格，尽量保住二维结构。"""
    rows: List[str] = []
    for row in table.findall("w:tr", _NS):
        cells: List[str] = []
        for cell in row.findall("w:tc", _NS):
            inner = " ".join(_paragraph_text(p) for p in cell.findall("w:p", _NS))
            cells.append(_normalize(inner))
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _build_sections(
    body: ET.Element,
    max_section_chars: int = DEFAULT_MAX_SECTION_CHARS,
) -> List[Dict[str, Any]]:
    """按"标题 + 后续段落"把正文分组成段落块。"""
    sections: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None

    def flush() -> None:
        nonlocal current
        if current is not None and current["parts"]:
            text = "\n".join(current["parts"]).strip()
            if text:
                sections.append({"text": text, "heading": current["heading"],
                                 "level": current["level"]})
        current = None

    for node in body:
        name = _localname(node.tag)
        if name == "p":
            text = _normalize(_paragraph_text(node))
            level = _heading_level(node)
            if level is not None and text:
                flush()
                current = {"heading": text, "level": level, "parts": [text]}
                continue
            if not text:
                continue          # 空段落直接跳过
            if current is not None and (
                sum(len(part) for part in current["parts"]) + len(text) > max_section_chars
            ):
                # 没有标题（或标题下正文太长）时按长度切块，避免整篇只剩一个定位单元
                flush()
            if current is None:
                current = {"heading": "", "level": 0, "parts": []}
            current["parts"].append(text)
        elif name == "tbl":
            table = _normalize(_table_text(node))
            flush()
            if table:
                sections.append({"text": table, "heading": "", "level": 0})
        # 其它节点（w:sectPr 等）忽略
    flush()
    return sections


def _read_optional(archive: zipfile.ZipFile, name: str) -> bytes:
    try:
        return archive.read(name)
    except (KeyError, OSError):
        return b""


def _extract_title(core_xml: bytes, sections: List[DocxSection], path: Path) -> str:
    """标题优先级：docProps/core.xml 的 dc:title -> 文档里的第一个标题 -> 文件名。"""
    if core_xml:
        try:
            root = ET.fromstring(core_xml)
            node = root.find("{%s}title" % _DC_NS)
            if node is not None and (node.text or "").strip():
                return node.text.strip()
        except ET.ParseError:
            pass
    for section in sections:
        if section.heading_level and section.heading:
            return section.heading
    return path.stem


def load_docx(
    path: Union[Path, str],
    *,
    allow_empty: bool = False,
    max_section_chars: int = DEFAULT_MAX_SECTION_CHARS,
) -> LoadedDocx:
    """读取一个 .docx，返回带段落块编号的文本。

    allow_empty=True 时允许"抽不出文字"的文档（例如正文全是图片）而不报错，
    便于批量导入时跳过坏文件。
    """
    docx_path = Path(path).expanduser()
    if not docx_path.is_file():
        raise DocxLoadError("文件不存在：%s" % docx_path)

    suffix = docx_path.suffix.lower()
    if suffix == LEGACY_DOC_SUFFIX:
        raise DocxLoadError(legacy_doc_message(docx_path.name))
    if not zipfile.is_zipfile(docx_path):
        raise DocxLoadError(
            "%s 不是有效的 .docx：.docx 本质是 ZIP 包，文件可能已损坏或后缀不对。"
            % docx_path.name
        )

    try:
        with zipfile.ZipFile(docx_path) as archive:
            try:
                xml_bytes = archive.read("word/document.xml")
            except KeyError as exc:
                raise DocxLoadError(
                    "%s 里没有 word/document.xml，不是有效的 Word 文档。" % docx_path.name
                ) from exc
            core_xml = _read_optional(archive, "docProps/core.xml")
    except (zipfile.BadZipFile, OSError) as exc:
        raise DocxLoadError("打开 %s 失败：%s" % (docx_path.name, exc)) from exc

    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        raise DocxLoadError("%s 的正文 XML 解析失败：%s" % (docx_path.name, exc)) from exc

    body = root.find("w:body", _NS)
    if body is None:
        raise DocxLoadError("%s 的正文结构异常（找不到 w:body）。" % docx_path.name)

    raw_sections = _build_sections(body, max_section_chars)
    pages = [
        DocxSection(
            index=index,
            text=item["text"],
            heading=item["heading"],
            heading_level=int(item["level"]),
        )
        for index, item in enumerate(raw_sections, start=1)
    ]

    if not allow_empty and not pages:
        raise DocxLoadError(
            "没能从 %s 抽出任何文字（文档可能是空的，或正文全在图片/文本框里）。"
            % docx_path.name
        )

    title = _extract_title(core_xml, pages, docx_path)
    return LoadedDocx(
        path=docx_path,
        doc_id=compute_doc_id(docx_path),
        title=title,
        page_count=len(pages),
        pages=pages,
        metadata={"title": title, "sections": len(pages)},
    )


def iter_docx_paths(directory: Union[Path, str], *, recursive: bool = True) -> List[Path]:
    """列出一个目录下所有 .docx（按路径排序，结果稳定）。"""
    base = Path(directory).expanduser()
    if not base.is_dir():
        return []
    pattern = "**/*" if recursive else "*"
    return sorted(
        item for item in base.glob(pattern)
        if item.is_file() and item.suffix.lower() == DOCX_SUFFIX
    )
