"""把前四步串成两条链路：ingest（入库）和 answer（问答）。

    ingest:  PDF 文件 -> load_pdf -> chunk_pages -> embed -> vector_store
    answer:  问题 -> embed -> vector_store.search -> 拼上下文 -> DeepSeek -> 答案 + 引用

设计要点
--------
* 所有依赖都是惰性构建：没配 key 时对象也能造出来，只是 status() 会告诉你缺什么，
  界面因此可以正常打开并给出指引，而不是一启动就崩。
* answer_stream() 产出结构化事件（hits / delta / done / error），
  前端拿它做"先显示引用、再逐字出答案"的效果，也便于测试。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Union

from .chunker import chunk_pages, summarize
from .config import (
    SUPPORTED_SUFFIXES,
    ConfigError,
    Settings,
    UnsupportedFormatError,
)
from .docx_loader import (
    DOCX_SUFFIX,
    LEGACY_DOC_SUFFIX,
    DocxLoadError,
    LoadedDocx,
    legacy_doc_message,
    load_docx,
)
from .embedder import BaseEmbedder, EmbeddingError, build_embedder
from .llm import DeepSeekClient, LLMError, build_rag_messages
from .pdf_loader import PDF_SUFFIX, LoadedPdf, PdfLoadError, load_pdf
from .retriever import DEFAULT_MAX_CONTEXT_CHARS, Retriever
from .vector_store import Hit, VectorStore, VectorStoreError

# 两种加载结果的结构一致（doc_id / title / source / page_count / pages / unit），
# 所以 ingest 之后的所有逻辑完全共用，只有"用哪个加载器"这一步按后缀分支。
LoadedDocument = Union[LoadedPdf, LoadedDocx]


@dataclass
class IngestResult:
    doc_id: str
    source: str
    title: str
    pages: int                # 定位单元数量：PDF 是页数，DOCX 是段落块数
    chunks: int
    chars: int
    skipped_pages: List[int] = field(default_factory=list)
    unit: str = "page"        # 定位单位：page / section

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "source": self.source,
            "title": self.title,
            "pages": self.pages,
            "chunks": self.chunks,
            "chars": self.chars,
            "skipped_pages": list(self.skipped_pages),
            "unit": self.unit,
        }


@dataclass
class AnswerEvent:
    type: str                     # hits | delta | done | error
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineStatus:
    embedding_ready: bool
    llm_ready: bool
    documents: int
    chunks: int
    messages: List[str]
    embedding_name: str = ""
    embedding_dim: int = 0

    @property
    def can_answer(self) -> bool:
        return self.embedding_ready and self.llm_ready and self.chunks > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "embedding_ready": self.embedding_ready,
            "llm_ready": self.llm_ready,
            "documents": self.documents,
            "chunks": self.chunks,
            "messages": list(self.messages),
            "embedding_name": self.embedding_name,
            "embedding_dim": self.embedding_dim,
        }


class RagPipeline:
    def __init__(
        self,
        settings: Settings,
        *,
        embedder: Optional[BaseEmbedder] = None,
        store: Optional[VectorStore] = None,
        llm: Optional[Any] = None,
    ) -> None:
        self.settings = settings
        self._embedder = embedder
        self._store = store
        self._llm = llm

    # -- 惰性依赖 ----------------------------------------------------
    @property
    def embedder(self) -> BaseEmbedder:
        if self._embedder is None:
            self._embedder = build_embedder(self.settings)
        return self._embedder

    @property
    def store(self) -> VectorStore:
        if self._store is None:
            self._store = VectorStore(self.settings, embedder=self.embedder)
        return self._store

    def require_llm(self) -> Any:
        if self._llm is None:
            self._llm = DeepSeekClient(
                self.settings.deepseek_api_key,
                base_url=self.settings.deepseek_base_url,
                model=self.settings.deepseek_model,
                proxy=self.settings.proxy,
            )
        return self._llm

    # -- 状态 --------------------------------------------------------
    def status(self) -> PipelineStatus:
        """永远不抛异常：缺什么就以消息形式列出来，界面照常打开。"""
        messages: List[str] = []
        embedding_ready = False
        embedding_name = ""
        embedding_dim = 0

        try:
            embedder = self.embedder
            embedding_ready = True
            embedding_name = str(getattr(embedder, "name", ""))
            embedding_dim = int(getattr(embedder, "dim", 0) or 0)
        except EmbeddingError as exc:
            messages.append(str(exc))

        llm_ready = bool(self.settings.deepseek_api_key)
        if not llm_ready:
            messages.append(
                "缺少 DEEPSEEK_API_KEY：可以入库和检索，但还不能生成答案。请在 .env 里补上。"
            )

        # 代理最容易出问题（配错端口就是 WinError 10061），所以直接把最终生效的设置摆出来
        proxy = self.settings.proxy
        if proxy.warning:
            messages.append(proxy.warning)
        for line in proxy.describe():
            messages.append("网络：" + line)

        documents = chunks = 0
        if embedding_ready:
            try:
                documents = len(self.store.list_documents())
                chunks = self.store.count()
            except VectorStoreError as exc:
                messages.append(str(exc))

        return PipelineStatus(
            embedding_ready=embedding_ready,
            llm_ready=llm_ready,
            documents=documents,
            chunks=chunks,
            messages=messages,
            embedding_name=embedding_name,
            embedding_dim=embedding_dim,
        )

    # -- ingest ------------------------------------------------------
    def ingest(
        self,
        source_path: Union[Path, str],
        *,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> IngestResult:
        """把一个 PDF / Word 文档切块并写进向量库（按后缀自动选加载器）。"""

        def progress(message: str) -> None:
            if on_progress:
                on_progress(message)

        path = Path(source_path)
        progress("读取文件：%s" % path.name)

        # 按后缀分发到对应加载器。返回结构一致，后面的切块 / 向量化 / 入库完全共用。
        suffix = path.suffix.lower()
        if suffix == DOCX_SUFFIX:
            loaded: LoadedDocument = load_docx(path)
            load_error = DocxLoadError
        elif suffix == PDF_SUFFIX:
            loaded = load_pdf(path)
            load_error = PdfLoadError
        elif suffix == LEGACY_DOC_SUFFIX:
            # 老式 .doc 单独给一条可操作的提示（复用语 docx_loader 里的同一句话）
            raise DocxLoadError(legacy_doc_message(path.name))
        else:
            raise UnsupportedFormatError(
                "不支持的文件格式：%s。目前支持 %s。"
                % (path.name, "、".join(SUPPORTED_SUFFIXES))
            )

        unit_kind = str(getattr(loaded, "unit", "page") or "page")
        unit_label = "页" if unit_kind == "page" else "段"
        progress("切块（%d %s）" % (loaded.page_count, unit_label))
        chunks = chunk_pages(
            loaded,
            size=self.settings.chunk_size,
            overlap=self.settings.chunk_overlap,
        )
        if not chunks:
            raise load_error("没能从 %s 切出任何有效的文本块" % path.name)

        stats = summarize(chunks)
        progress("向量化 %d 个块（%s）" % (stats["count"], self.embedder.name))
        self.store.upsert_document(chunks, embedder=self.embedder)
        progress("入库完成：%s" % path.name)

        return IngestResult(
            doc_id=loaded.doc_id,
            source=loaded.source,
            title=loaded.title,
            pages=loaded.page_count,
            chunks=int(stats["count"]),
            chars=int(stats["total_chars"]),
            skipped_pages=list(loaded.empty_pages),
            unit=unit_kind,
        )

    def ingest_many(
        self,
        paths: Sequence[Union[Path, str]],
        *,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> List[IngestResult]:
        return [self.ingest(item, on_progress=on_progress) for item in paths]

    # -- query -------------------------------------------------------
    def retriever(self, *, top_k: Optional[int] = None) -> Retriever:
        return Retriever(self.store, self.embedder, top_k=int(top_k or self.settings.top_k))

    def search(
        self,
        question: str,
        *,
        top_k: Optional[int] = None,
        doc_ids: Optional[Sequence[str]] = None,
    ) -> List[Hit]:
        return self.retriever(top_k=top_k).retrieve(question, doc_ids=doc_ids)

    def build_context(
        self,
        hits: Sequence[Hit],
        *,
        max_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    ):
        return self.retriever().build_context(hits, max_chars=max_chars)

    def answer(
        self,
        question: str,
        *,
        top_k: Optional[int] = None,
        temperature: Optional[float] = None,
        doc_ids: Optional[Sequence[str]] = None,
        max_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    ) -> Dict[str, Any]:
        """非流式问答，返回 {answer, citations, hits}，方便测试和命令行使用。"""
        hits = self.search(question, top_k=top_k, doc_ids=doc_ids)
        if not hits:
            return {"answer": "", "citations": [], "hits": [], "reason": "no_hits"}
        context_text, citations = self.build_context(hits, max_chars=max_chars)
        client = self.require_llm()
        messages = build_rag_messages(question, context_text)
        result = client.chat(messages, temperature=temperature)
        return {
            "answer": (result.get("content") or "").strip(),
            "citations": citations,
            "hits": hits,
            "context": context_text,
            "messages": messages,
            "usage": result.get("usage") or {},
            "reason": "",
        }

    def answer_stream(
        self,
        question: str,
        *,
        top_k: Optional[int] = None,
        temperature: Optional[float] = None,
        doc_ids: Optional[Sequence[str]] = None,
        max_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    ) -> Iterator[AnswerEvent]:
        """流式问答：先给检索结果（hits），再逐段给答案文本（delta）。"""
        text = (question or "").strip()
        if not text:
            yield AnswerEvent("error", {"message": "问题不能为空。"})
            return

        try:
            hits = self.search(text, top_k=top_k, doc_ids=doc_ids)
        except (EmbeddingError, VectorStoreError, ConfigError) as exc:
            yield AnswerEvent("error", {"message": str(exc)})
            return

        if not hits:
            yield AnswerEvent("error", {"message": "知识库里没有找到相关内容，请先上传 PDF。"})
            return

        context_text, citations = self.build_context(hits, max_chars=max_chars)
        yield AnswerEvent(
            "hits",
            {
                "hits": [hit.to_dict() for hit in hits],
                "citations": [item.to_dict() for item in citations],
                "context": context_text,
            },
        )

        try:
            client = self.require_llm()
        except (ConfigError, LLMError) as exc:
            yield AnswerEvent("error", {"message": str(exc)})
            return

        messages = build_rag_messages(text, context_text)
        try:
            for piece in client.stream(messages, temperature=temperature):
                yield AnswerEvent("delta", {"text": piece})
        except LLMError as exc:
            yield AnswerEvent("error", {"message": str(exc)})
            return

        yield AnswerEvent("done", {})

    # -- 文档管理 ----------------------------------------------------
    def documents(self) -> List[Dict[str, Any]]:
        try:
            return self.store.list_documents()
        except (VectorStoreError, EmbeddingError):
            return []

    def delete_document(self, doc_id: str) -> int:
        return self.store.delete_document(doc_id)

    def reset(self) -> None:
        self.store.reset()
