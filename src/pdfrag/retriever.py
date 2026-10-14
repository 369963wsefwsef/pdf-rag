"""第四步（上半）：检索 + 组装上下文。

流程：问题 -> 向量 -> 在向量库里找最相似的 top_k 个块 -> 编号拼成一段可引用的上下文。

拼上下文时做了两件事：
* 编号：让大模型能用 [1][2] 这样的角标引用来源，前端再把编号映射回文件名和页码；
* 预算：限制总字符数，避免把 top_k 个超长块全塞进去把 token 打爆。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .vector_store import Hit, VectorStore

DEFAULT_MAX_CONTEXT_CHARS = 6000
SEPARATOR = "\n\n"


def format_location(unit: str, index: int) -> str:
    """统一渲染定位信息：PDF 是"第 3 页"，DOCX 是"第 3 段"。"""
    return ("第 %d 段" % index) if unit != "page" else ("第 %d 页" % index)


@dataclass(frozen=True)
class Citation:
    index: int
    source: str
    page: int              # 定位单元序号：PDF 页码 / DOCX 段落块号
    score: float
    doc_id: str
    chunk_id: str
    unit: str = "page"     # 定位单位：page / section

    @property
    def label(self) -> str:
        """给界面展示的一行引用标签。"""
        if self.unit == "page":
            return "[%d] %s p.%d" % (self.index, self.source, self.page)
        return "[%d] %s 第%d段" % (self.index, self.source, self.page)

    @property
    def location(self) -> str:
        return format_location(self.unit, self.page)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "source": self.source,
            "unit": self.unit,
            "page": self.page,
            "score": round(self.score, 4),
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "label": self.label,
        }


class Retriever:
    def __init__(self, store: VectorStore, embedder: Any, *, top_k: int = 5) -> None:
        self.store = store
        self.embedder = embedder
        self.top_k = int(top_k)

    def retrieve(
        self,
        question: str,
        *,
        top_k: Optional[int] = None,
        doc_ids: Optional[Sequence[str]] = None,
    ) -> List[Hit]:
        text = (question or "").strip()
        if not text:
            return []
        vector = self.embedder.embed_query(text)
        return self.store.search(vector, top_k=int(top_k or self.top_k), doc_ids=doc_ids)

    def build_context(
        self,
        hits: Sequence[Hit],
        *,
        max_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
        with_scores: bool = True,
    ) -> Tuple[str, List[Citation]]:
        """把命中片段拼成带编号的上下文，返回 (上下文文本, 引用列表)。

        引用编号与上下文里的 [n] 一一对应，前端据此展示"来自哪个文件的第几页/第几段"。
        """
        blocks: List[str] = []
        citations: List[Citation] = []
        used_chars = 0
        seen = set()

        for hit in hits:
            body = (hit.text or "").strip()
            if not body:
                continue
            fingerprint = body[:120]      # 同一段文字被切了两块时去重，别浪费预算
            if fingerprint in seen:
                continue
            seen.add(fingerprint)

            index = len(citations) + 1
            unit = str(getattr(hit, "unit", "page") or "page")
            header = "[%d] 来源：%s %s" % (index, hit.source, format_location(unit, hit.page))
            if with_scores:
                header += "（相似度 %.3f）" % hit.score
            block = header + "\n" + body

            if used_chars + len(block) > max_chars and citations:
                break
            blocks.append(block)
            citations.append(
                Citation(
                    index=index,
                    source=hit.source,
                    page=hit.page,
                    score=hit.score,
                    doc_id=hit.doc_id,
                    chunk_id=hit.chunk_id,
                    unit=unit,
                )
            )
            used_chars += len(block) + len(SEPARATOR)

        return SEPARATOR.join(blocks), citations
