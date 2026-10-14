"""用标准库现场拼出合法的 .docx，作为测试用的"假 Word 数据"。

不安装 python-docx、不需要任何真实 Word 文件：.docx 就是一个 ZIP 包，
里面放 [Content_Types].xml、_rels/.rels、word/document.xml 就够了。
zip 条目时间戳固定，保证"同样内容 -> 同样字节"，方便断言 doc_id 的稳定性。

用法：
    from _docx_builder import build_docx, heading, paragraph, table

    path = build_docx(tmp / "a.docx", [
        heading("第一章 概述", 1),
        paragraph("这是正文。"),
        table([["项目", "说明"], ["A", "第一项"]]),
    ], doc_title="我的手册")
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Union

_W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_FIXED_TIME = (2026, 1, 1, 0, 0, 0)

CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
</Types>
"""

ROOT_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
</Relationships>
"""


def heading(text: str, level: int = 1) -> Dict[str, Any]:
    return {"type": "heading", "text": text, "level": level}


def paragraph(text: str) -> Dict[str, Any]:
    return {"type": "paragraph", "text": text}


def table(rows: Sequence[Sequence[str]]) -> Dict[str, Any]:
    return {"type": "table", "rows": [list(row) for row in rows]}


def _escape(text: str) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


def _runs(text: str) -> str:
    """把文本切成 run；\n 转成 <w:br/>，这样段落内的换行也能被解析出来。"""
    pieces: List[str] = []
    for index, line in enumerate(str(text).split("\n")):
        if index:
            pieces.append("<w:r><w:br/></w:r>")
        pieces.append('<w:r><w:t xml:space="preserve">%s</w:t></w:r>' % _escape(line))
    return "".join(pieces)


def _paragraph_xml(text: str, *, style: str = "", outline: int = -1) -> str:
    properties = ""
    if style or outline >= 0:
        inner = ""
        if style:
            inner += '<w:pStyle w:val="%s"/>' % _escape(style)
        if outline >= 0:
            inner += '<w:outlineLvl w:val="%d"/>' % outline
        properties = "<w:pPr>%s</w:pPr>" % inner
    return "<w:p>%s%s</w:p>" % (properties, _runs(text))


def _table_xml(rows: Sequence[Sequence[str]]) -> str:
    body: List[str] = []
    for row in rows:
        cells = "".join(
            "<w:tc><w:p>%s</w:p></w:tc>" % _runs(cell) for cell in row
        )
        body.append("<w:tr>%s</w:tr>" % cells)
    return "<w:tbl>%s</w:tbl>" % "".join(body)


def _block_xml(block: Dict[str, Any]) -> str:
    kind = block.get("type", "paragraph")
    if kind == "heading":
        level = int(block.get("level", 1))
        # 真实 Word 用样式名 Heading1/标题1，pStyle + outlineLvl 都会写
        return _paragraph_xml(block["text"], style="Heading%d" % level, outline=level - 1)
    if kind == "paragraph":
        return _paragraph_xml(block["text"])
    if kind == "table":
        return _table_xml(block["rows"])
    if kind == "blank":
        return "<w:p/>"
    raise ValueError("未知的块类型：%r" % kind)


def build_document_xml(blocks: Iterable[Dict[str, Any]]) -> str:
    body = "".join(_block_xml(block) for block in blocks)
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<w:document xmlns:w="%s"><w:body>%s<w:sectPr/></w:body></w:document>'
        % (_W_NS, body)
    )


def build_core_xml(title: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"'
        ' xmlns:dc="http://purl.org/dc/elements/1.1/">'
        "<dc:title>%s</dc:title></cp:coreProperties>" % _escape(title)
    )


def _write_part(archive: zipfile.ZipFile, name: str, text: str) -> None:
    info = zipfile.ZipInfo(name, date_time=_FIXED_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    archive.writestr(info, text)


def build_docx(
    path: Union[Path, str],
    blocks: Iterable[Dict[str, Any]],
    *,
    doc_title: str = "",
    document_xml: str = "",
    with_core: bool = True,
) -> Path:
    """写出一个合法的 .docx，返回文件路径。

    document_xml 不为空时直接用它（用于构造畸形 XML 等边界情况）。
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    xml = document_xml or build_document_xml(blocks)

    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        _write_part(archive, "[Content_Types].xml", CONTENT_TYPES)
        _write_part(archive, "_rels/.rels", ROOT_RELS)
        _write_part(archive, "word/document.xml", xml)
        if with_core:
            _write_part(archive, "docProps/core.xml", build_core_xml(doc_title))
    return target


def build_raw_zip(path: Union[Path, str], parts: Dict[str, str]) -> Path:
    """写出一个自定义内容的 ZIP（用来模拟"不是 Word 的 zip"）。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, text in parts.items():
            _write_part(archive, name, text)
    return target
