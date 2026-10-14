r"""第五步：Streamlit 网页界面。

启动：
    .\.venv\Scripts\python.exe -m streamlit run app.py

界面分三个区域：
* 左侧栏：配置状态（缺哪个 key 会明确提示）、检索参数、文档清单 + 删除
* 主区域左：上传 PDF 并入库
* 主区域右：提问，流式显示答案，并列出引用来源（文件名 + 页码 + 原文）
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from pdfrag.config import SUPPORTED_SUFFIXES, Settings, UnsupportedFormatError  # noqa: E402
from pdfrag.docx_loader import DocxLoadError    # noqa: E402
from pdfrag.embedder import EmbeddingError      # noqa: E402
from pdfrag.pdf_loader import PdfLoadError      # noqa: E402
from pdfrag.pipeline import RagPipeline         # noqa: E402

# 界面上允许上传的格式，从 config 派生（结果是 ["pdf", "docx"]），避免两处各写一遍
UPLOAD_TYPES = [suffix.lstrip(".") for suffix in SUPPORTED_SUFFIXES]

st.set_page_config(page_title="文档知识库问答", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner=False)
def get_pipeline(settings: Settings) -> RagPipeline:
    """按配置缓存一个 RagPipeline（用 settings 做缓存键，配置变了会自动重建）。"""
    settings.ensure_dirs()
    return RagPipeline(settings)


def save_upload(uploaded, target_dir: Path) -> Path:
    """把 Streamlit 上传的文件落盘，并做重名保护。"""
    target_dir.mkdir(parents=True, exist_ok=True)
    name = Path(uploaded.name).name
    candidate = target_dir / name
    if candidate.exists():
        candidate = target_dir / ("%s-%d%s" % (candidate.stem, int(time.time()), candidate.suffix))
    candidate.write_bytes(uploaded.getbuffer())
    return candidate


def render_sidebar(settings: Settings, pipeline: RagPipeline) -> dict:
    st.sidebar.header("⚙️ 运行状态")
    status = pipeline.status()

    first, second = st.sidebar.columns(2)
    first.metric("已入库文档", status.documents)
    second.metric("向量块数", status.chunks)

    if status.embedding_ready:
        st.sidebar.success("Embedding 就绪：%s（%d 维）" % (status.embedding_name, status.embedding_dim))
    else:
        st.sidebar.error("Embedding 未就绪")

    if status.llm_ready:
        st.sidebar.success("大模型就绪：%s" % settings.deepseek_model)
    else:
        st.sidebar.warning("大模型未配置（只能检索，不能生成答案）")

    for message in status.messages:
        st.sidebar.info(message)

    with st.sidebar.expander("查看 / 调试配置"):
        for key, value in settings.describe().items():
            st.write("**%s**：%s" % (key, value))

    st.sidebar.divider()
    st.sidebar.subheader("🔧 检索参数")
    top_k = st.sidebar.slider("检索条数 top_k", 1, 20, int(settings.top_k))
    with st.sidebar.expander("切块参数（只影响之后新入库的文档）"):
        st.caption("当前：%d / %d 字符" % (settings.chunk_size, settings.chunk_overlap))
        st.caption("要修改请编辑 .env 里的 CHUNK_SIZE / CHUNK_OVERLAP，然后重启应用。")

    st.sidebar.divider()
    st.sidebar.subheader("📄 文档管理")
    documents = pipeline.documents()
    if not documents:
        st.sidebar.caption("还没有任何文档。")
    for document in documents:
        columns = st.sidebar.columns([4, 1])
        columns[0].write("**%s**  \n%d 块 / %d 页"
                         % (document["source"], document["chunks"], document["pages"]))
        if columns[1].button("🗑", key="del_%s" % document["doc_id"], help="删除这份文档的向量"):
            pipeline.delete_document(document["doc_id"])
            st.rerun()

    if documents and st.sidebar.button("清空整个知识库"):
        pipeline.reset()
        st.rerun()

    if st.session_state.get("chat_messages") and st.sidebar.button("🧹 清空对话记录"):
        st.session_state.chat_messages = []
        st.rerun()

    return {"top_k": top_k}


def render_upload(pipeline: RagPipeline, settings: Settings) -> None:
    st.subheader("① 上传文档入库")
    uploads = st.file_uploader(
        "可一次选多个 PDF / Word 文档（同一份文件重复上传会替换旧向量，不会重复）",
        type=UPLOAD_TYPES,          # ["pdf", "docx"]
        accept_multiple_files=True,
    )
    if not uploads:
        return
    if not pipeline.status().embedding_ready:
        st.error("Embedding 还没配置好，无法入库。请先在 .env 里填 EMBEDDING_API_KEY。")
        return

    if st.button("开始入库", type="primary"):
        for uploaded in uploads:
            try:
                path = save_upload(uploaded, settings.upload_dir)
                with st.status("正在处理 %s" % path.name, expanded=True) as box:
                    messages = []
                    result = pipeline.ingest(path, on_progress=lambda item: messages.append(item))
                    for message in messages:
                        st.write(message)
                    unit_label = "页" if result.unit == "page" else "段"
                    st.write("完成：%d %s / %d 块 / %d 字符"
                             % (result.pages, unit_label, result.chunks, result.chars))
                    box.update(label="✅ %s 已入库" % path.name, state="complete")
                st.success("**%s** 入库成功（doc_id=%s）" % (result.source, result.doc_id))
            except (PdfLoadError, DocxLoadError, UnsupportedFormatError, EmbeddingError) as exc:
                st.error("处理 %s 失败：%s" % (uploaded.name, exc))
            except Exception as exc:   # 兜底：单个坏文件不该让整个界面崩掉
                st.error("处理 %s 时出现未预期错误：%r" % (uploaded.name, exc))
        st.rerun()


def render_citations(hits, citations) -> None:
    if not hits:
        return
    st.markdown("**📌 引用来源**")
    for hit, citation in zip(hits, citations):
        # label 由 retriever 生成，已经区分了"第 N 页"（PDF）和"第 N 段"（Word）
        header = "%s · 相似度 %.3f" % (citation["label"], citation["score"])
        with st.expander(header):
            st.caption("chunk：%s（第 %d 块）" % (hit["chunk_id"], hit["chunk_index"]))
            st.write(hit["text"])


# 流式输出时的光标：只在"生成过程中"出现，收尾那次渲染会去掉它。
CURSOR = "▌"


def _chat_messages() -> list:
    """对话历史存在 session_state 里。

    这一点是修掉"页面上残留一堆带光标的半截句子"的关键：
    脚本每次重跑（点按钮、改滑块、刷新页面、上一次还没输出完又提交了一次）时，
    页面上的问答全部由这里的**完整结果**重绘出来，而不是由某次流式输出的中间态负责渲染。
    """
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    return st.session_state.chat_messages


def _render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message.get("content") or "_（没有生成内容）_")
        for error in message.get("errors") or []:
            st.error(error)
        if message.get("citations"):
            render_citations(message.get("hits") or [], message["citations"])


def render_ask(pipeline: RagPipeline, options: dict) -> None:
    st.subheader("② 提问")
    status = pipeline.status()
    if status.chunks == 0:
        st.info("请先上传并入库至少一份 PDF / Word 文档。")
        return
    if not status.llm_ready:
        st.warning("还没配置 DEEPSEEK_API_KEY，暂时无法生成答案（检索本身是可用的）。")

    # 先把已完成的问答原样重绘出来（它们只来自 session_state，永远不含光标）
    for message in _chat_messages():
        _render_message(message)

    # 用 chat_input 而不是 text_input + button：
    # 它在回答生成期间是禁用的，天然避免"上一次还在流式输出、又提交了一次提问"——
    # 那种情况下旧运行的半成品会以带光标的中间态留在页面上。
    question = (st.chat_input("输入你的问题，例如：这份文档主要讲了什么？") or "").strip()
    if not question:
        return

    _chat_messages().append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    hits: list = []
    citations: list = []
    errors: list = []

    def text_chunks():
        """把事件流里的文本片段交给 st.write_stream，顺带收集引用来源与错误。"""
        for event in pipeline.answer_stream(question, top_k=options.get("top_k")):
            if event.type == "hits":
                hits.clear()
                hits.extend(event.data.get("hits") or [])
                citations.clear()
                citations.extend(event.data.get("citations") or [])
            elif event.type == "delta":
                yield event.data.get("text", "")
            elif event.type == "error":
                errors.append(event.data.get("message", "未知错误"))
                return
            elif event.type == "done":
                return

    with st.chat_message("assistant"):
        # write_stream 是 Streamlit 官方的流式渲染 API：内部同样用 st.empty() 替换，
        # 但它会处理"半个 markdown 语法"（例如未闭合的 ** 粗体），比自己拼字符串稳。
        answer = st.write_stream(text_chunks(), cursor=CURSOR) or ""
        for message in errors:
            st.error(message)
        if citations:
            render_citations(hits, citations)

    # 关键一步：把**完整结果**写进 session_state。此后任何重绘都从这里取，
    # 光标和中间态再也不会出现在页面上。
    _chat_messages().append(
        {
            "role": "assistant",
            "content": answer,
            "hits": hits,
            "citations": citations,
            "errors": errors,
        }
    )


def main() -> None:
    settings = Settings.from_env()
    pipeline = get_pipeline(settings)

    st.title("📚 文档知识库问答")
    st.caption(
        "上传 PDF / Word(.docx) → 自动切块向量化 → 提问时先检索相关片段，"
        "再由 DeepSeek 依据这些片段作答，并给出引用来源。"
    )

    options = render_sidebar(settings, pipeline)

    left, right = st.columns([1, 1])
    with left:
        render_upload(pipeline, settings)
    with right:
        render_ask(pipeline, options)


if __name__ == "__main__":
    main()
