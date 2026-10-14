"""第二步：把长文本切成带重叠的小块（chunk）。

两个关键决策
------------
1) 为什么要重叠（overlap）：句子正好落在切点上时，前后两块各自都不完整。
   留一段重叠区，保证至少有一个块包含完整语义，召回率明显提升。
2) 为什么按页切：切块必须能回答"这句话在第几页"。跨页合并会让页码变模糊，
   所以这里逐页切块，代价是跨页的句子可能被分成两块（可接受的权衡）。

切分优先级：段落 \n\n > 换行 \n > 句末标点 > 逗号/空格 > 硬切。
前三级找不到合适位置时才用硬切，保证任何输入都不会产生超长块。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from typing import Any, Protocol


class LoadedDocumentLike(Protocol):
    """chunker 需要的最小接口：PDF 和 DOCX 的加载结果都满足它。

    pages 是"定位单元"列表：PDF 里是 PdfPage（带 .page），DOCX 里是
    DocxSection（带 .index）。序号由 _unit_index() 统一取出，
    所以 chunker 不需要为两种格式写两套逻辑。
    """

    doc_id: str
    source: str
    pages: Sequence[Any]


def _unit_index(unit: Any) -> int:
    """取出定位单元的序号：PDF 的页用 .page，DOCX 的段落块用 .index。"""
    for attribute in ("page", "index"):
        value = getattr(unit, attribute, None)
        if isinstance(value, int):
            return value
    return 0

PARAGRAPH_BREAK = "\n\n"
SENTENCE_ENDS = "。！？!?；;"
SOFT_ENDS = "，,、）)】」》 "

# 从"最体面"到"最将就"的切点候选
_BREAK_GROUPS: Tuple[str, ...] = ("\n\n", "\n", SENTENCE_ENDS, SOFT_ENDS)


@dataclass(frozen=True)
class Chunk:
    """一个可检索的最小单元。"""

    id: str
    text: str
    doc_id: str
    source: str
    page: int
    chunk_index: int
    char_start: int   # 定位单元内的起始偏移
    char_end: int     # 定位单元内的结束偏移
    unit: str = "page"   # 定位单位：page（PDF 页）/ section（Word 段落块）

    @property
    def unit_name(self) -> str:
        """定位单位的中文名，给界面展示用。"""
        return "页" if self.unit == "page" else "段"

    @property
    def citation(self) -> str:
        """给用户看的引用标签：PDF 是 a.pdf p.3，DOCX 是 a.docx 第3段。"""
        if self.unit == "page":
            return "%s p.%d" % (self.source, self.page)
        return "%s 第%d段" % (self.source, self.page)

    def to_metadata(self) -> Dict[str, object]:
        """转成向量库能存的元数据（只允许 str / int / float / bool）。"""
        return {
            "doc_id": self.doc_id,
            "source": self.source,
            "unit": self.unit,
            "page": int(self.page),
            "chunk_index": int(self.chunk_index),
            "char_start": int(self.char_start),
            "char_end": int(self.char_end),
        }


def _rfind_any(window: str, chars: str) -> int:
    return max((window.rfind(char) for char in chars), default=-1)


def _skip_leading_space(text: str, position: int, limit: int) -> int:
    while position < limit and text[position] in " \t\n":
        position += 1
    return position


def find_break(text: str, start: int, hard_end: int, min_length: int) -> int:
    """在允许的窗口里从右往左找一个体面的切点，找不到就按 hard_end 硬切。"""
    lower = start + min_length
    window = text[lower:hard_end]
    if not window:
        return hard_end

    for group in _BREAK_GROUPS:
        if len(group) > 1:
            index = window.rfind(group)
            if index >= 0:
                return lower + index + len(group)
        else:
            index = _rfind_any(window, group)
            if index >= 0:
                return lower + index + 1
    return hard_end


def split_text(text: str, size: int, overlap: int) -> List[Tuple[int, int]]:
    """把文本切成 [(start, end), ...]，相邻块之间有 overlap 个字符重叠。

    保证：1) 每块长度 <= size；2) 覆盖全文无空洞；3) 每次都严格前进（不死循环）。
    """
    if size <= 0:
        raise ValueError("size 必须为正整数")
    if overlap < 0:
        raise ValueError("overlap 不能为负")
    if overlap >= size:
        raise ValueError("overlap(%d) 必须小于 size(%d)" % (overlap, size))
    if not text:
        return []

    min_length = max(1, size // 3)
    total = len(text)
    spans: List[Tuple[int, int]] = []
    start = 0

    while start < total:
        end = min(start + size, total)
        if end < total:
            candidate = find_break(text, start, end, min_length)
            end = candidate if candidate > start else min(start + size, total)
        spans.append((start, end))
        if end >= total:
            break
        nxt = end - overlap
        if nxt <= start:                      # 兜底：保证严格前进
            nxt = start + 1
        start = _skip_leading_space(text, nxt, end)

    return spans


def chunk_pages(
    loaded: LoadedDocumentLike,
    *,
    size: int,
    overlap: int,
    min_chunk_chars: int = 20,
) -> List[Chunk]:
    """把一个已加载的文档切成 chunk 列表（逐个定位单元切，保留单位序号）。

    定位单位由 loaded.unit 决定：PDF 是"页"，DOCX 是"段落块"。
    逐单元切而不跨单元合并，是为了让引用（第几页 / 第几段）永远准确。
    """
    chunks: List[Chunk] = []
    unit_kind = str(getattr(loaded, "unit", "page") or "page")
    index = 0
    for page in loaded.pages:
        number = _unit_index(page)
        text = getattr(page, "text", "")
        if not text.strip():
            continue
        candidates = []
        for start, end in split_text(text, size, overlap):
            piece = text[start:end].strip()
            if piece:
                candidates.append((start, end, piece))

        # 正常情况下丢掉页尾的零碎残片（留着只会污染检索结果）；
        # 但如果整页都很短（例如只有一行标题），过滤后会一块不剩 —— 那就全部保留，
        # 否则短文档会被整个丢掉，用户上传后什么也检索不到。
        kept = [item for item in candidates if len(item[2]) >= min_chunk_chars]
        if not kept:
            kept = candidates

        for start, end, piece in kept:
            chunks.append(
                Chunk(
                    id="%s-u%d-c%d" % (loaded.doc_id, number, index),
                    text=piece,
                    doc_id=loaded.doc_id,
                    source=loaded.source,
                    page=number,
                    chunk_index=index,
                    char_start=start,
                    char_end=end,
                    unit=unit_kind,
                )
            )
            index += 1
    return chunks


def summarize(chunks: Sequence[Chunk]) -> Dict[str, float]:
    """给界面展示的切块统计。"""
    if not chunks:
        return {"count": 0, "total_chars": 0, "avg_chars": 0, "min_chars": 0,
                "max_chars": 0, "pages": 0}
    sizes = [len(chunk.text) for chunk in chunks]
    return {
        "count": len(chunks),
        "total_chars": sum(sizes),
        "avg_chars": round(sum(sizes) / len(sizes), 1),
        "min_chars": min(sizes),
        "max_chars": max(sizes),
        "pages": len({chunk.page for chunk in chunks}),
    }
