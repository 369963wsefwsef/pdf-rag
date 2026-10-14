"""手写一个最小但合法的多页 PDF，供测试使用。

不引入 reportlab 之类的依赖：PDF 1.4 的基本结构就是
"对象 + xref 偏移表 + trailer"，自己拼装还能顺便保证 xref 偏移是算出来的、
不是写死的，避免"生成了个坏 PDF 结果测试却在测坏 PDF"。
只支持 Latin-1 文本（测试用英文），中文文本请直接测 clean_text / chunker。
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Union


def _escape(text: str) -> str:
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def build_pdf(path: Union[Path, str], pages: Sequence[Sequence[str]]) -> Path:
    """pages: 每页一个字符串列表（每个元素是一行文本）。返回写入的文件路径。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    page_count = len(pages)
    first_page_obj = 4                      # 1=Catalog 2=Pages 3=Font
    kids = " ".join("%d 0 R" % (first_page_obj + 2 * i) for i in range(page_count))

    objects: dict = {}
    objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objects[2] = ("<< /Type /Pages /Kids [%s] /Count %d >>" % (kids, page_count)).encode("latin-1")
    objects[3] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"

    for index, lines in enumerate(pages):
        page_obj = first_page_obj + 2 * index
        content_obj = page_obj + 1
        objects[page_obj] = (
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            "/Resources << /Font << /F1 3 0 R >> >> /Contents %d 0 R >>" % content_obj
        ).encode("latin-1")

        parts: List[str] = ["BT", "/F1 12 Tf", "72 720 Td", "18 TL"]
        for position, line in enumerate(lines):
            if position:
                parts.append("T*")
            parts.append("(%s) Tj" % _escape(line))
        parts.append("ET")
        stream = "\n".join(parts).encode("latin-1", "replace")
        objects[content_obj] = b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream"

    out = bytearray(b"%PDF-1.4\n")
    offsets: dict = {}
    for number in sorted(objects):
        offsets[number] = len(out)
        out += ("%d 0 obj\n" % number).encode("latin-1") + objects[number] + b"\nendobj\n"

    max_number = max(objects)
    xref_position = len(out)
    out += ("xref\n0 %d\n" % (max_number + 1)).encode("latin-1")
    out += b"0000000000 65535 f \n"
    for number in range(1, max_number + 1):
        offset = offsets.get(number)
        if offset is None:
            out += b"0000000000 65535 f \n"
        else:
            out += ("%010d 00000 n \n" % offset).encode("latin-1")
    out += (
        "trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (max_number + 1, xref_position)
    ).encode("latin-1")

    target.write_bytes(bytes(out))
    return target
