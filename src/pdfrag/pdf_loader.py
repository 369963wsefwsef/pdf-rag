"""第一步：把本地 PDF 读成「带页码的纯文本」。

设计要点
--------
* 保留页码：做引用来源（citation）时必须能说出"来自第几页"。
* doc_id 用文件内容哈希：同一个文件重复上传天然幂等；文件改一个字就是新文档，
  旧向量不会和新向量混在一起（幂等入库的基础）。
* 只做纯文本抽取：扫描件（图片型 PDF）抽不出文字，这里会明确报错，
  而不是悄悄返回空串（OCR 属于后续可选项）。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pypdf import PdfReader

_CJK_RANGES = r"\u3000-\u303f\u4e00-\u9fff\uff00-\uffef"
_RE_CJK = re.compile("[" + _CJK_RANGES + "]")
_RE_HYPHEN_BREAK = re.compile(r"([A-Za-z])-\n([a-z])")
_RE_BLANK_RUN = re.compile(r"\n{3,}")
_RE_MULTI_SPACE = re.compile(r"[ \t]{2,}")


PDF_SUFFIX = ".pdf"


class PdfLoadError(RuntimeError):
    """PDF 读取失败：文件不存在、损坏、加密、或抽不出任何文字。"""


@dataclass(frozen=True)
class PdfPage:
    """一页文本。page 从 1 开始计数，直接用于展示引用。"""

    page: int
    text: str


@dataclass
class LoadedPdf:
    path: Path
    doc_id: str
    title: str
    page_count: int
    pages: List[PdfPage]
    metadata: Dict[str, Any] = field(default_factory=dict)
    failed_pages: List[int] = field(default_factory=list)

    @property
    def source(self) -> str:
        """用于引用展示的文件名。"""
        return self.path.name

    @property
    def empty_pages(self) -> List[int]:
        return [page.page for page in self.pages if not page.text.strip()]

    def full_text(self) -> str:
        return "\n\n".join(page.text for page in self.pages if page.text.strip())

    def char_count(self) -> int:
        return sum(len(page.text) for page in self.pages)


def compute_doc_id(path: Union[Path, str], *, block_size: int = 1 << 20) -> str:
    """按文件内容计算稳定 id（sha256 前 16 位）。

    用内容而不是文件名：同一个 PDF 改名后仍然认得出来，
    避免"同一份文件被存了两遍、检索结果重复"。
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()[:16]


def _is_cjk(char: str) -> bool:
    return bool(_RE_CJK.match(char))


def merge_wrapped_lines(text: str) -> str:
    """把 PDF 的硬换行还原成正常段落。

    PDF 里每一行都是排版结果，不是语义换行：一句中文可能被切成好几行。
    中文换行处直接拼接（中文本来就不用空格分词），英文换行处补一个空格，
    否则会把两个单词粘成一个。
    """
    out: List[str] = []
    for line in text.split("\n"):
        if not out or not out[-1].strip() or not line.strip():
            out.append(line)
            continue
        previous = out[-1]
        joiner = "" if (_is_cjk(previous[-1]) and _is_cjk(line[0])) else " "
        out[-1] = previous + joiner + line
    return "\n".join(out)


def clean_text(raw: str) -> str:
    """标准化抽取出来的文本，供切块使用。"""
    if not raw:
        return ""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    # 英文断词还原：inter-\nnational -> international
    text = _RE_HYPHEN_BREAK.sub(r"\1\2", text)
    text = merge_wrapped_lines(text)
    text = _RE_MULTI_SPACE.sub(" ", text)
    text = _RE_BLANK_RUN.sub("\n\n", text)
    return text.strip()


def _try_decrypt(reader: PdfReader, password: str) -> bool:
    try:
        return bool(reader.decrypt(password or ""))
    except Exception:
        return False


def _extract_title(reader: PdfReader, fallback: Path) -> str:
    try:
        metadata = reader.metadata or {}
    except Exception:
        metadata = {}
    for key in ("/Title", "title"):
        try:
            value = metadata.get(key)
        except Exception:
            value = None
        if value and str(value).strip():
            return str(value).strip()
    return fallback.stem


def load_pdf(
    path: Union[Path, str],
    *,
    password: Optional[str] = None,
    allow_empty: bool = False,
) -> LoadedPdf:
    """读取一个 PDF，返回带页码的文本。

    allow_empty=True 时允许"抽不出文字"（例如扫描件）而不报错，
    便于批量导入时跳过坏文件。
    """
    pdf_path = Path(path).expanduser()
    if not pdf_path.is_file():
        raise PdfLoadError("文件不存在：%s" % pdf_path)

    try:
        reader = PdfReader(str(pdf_path))
    except Exception as exc:
        raise PdfLoadError(
            "无法解析 PDF：%s（%s: %s）" % (pdf_path.name, type(exc).__name__, exc)
        ) from exc

    if reader.is_encrypted:
        # 很多 PDF 只是设置了权限位，空密码即可打开
        candidates = [password] if password else []
        candidates.append("")
        if not any(_try_decrypt(reader, candidate) for candidate in candidates if candidate is not None):
            raise PdfLoadError("PDF 已加密，需要提供密码：%s" % pdf_path.name)

    pages: List[PdfPage] = []
    failed: List[int] = []
    try:
        total = len(reader.pages)
    except Exception as exc:
        raise PdfLoadError("无法读取页数：%s（%s）" % (pdf_path.name, exc)) from exc

    for index in range(total):
        page_number = index + 1
        try:
            raw = reader.pages[index].extract_text() or ""
        except Exception:
            # 单页异常不该让整篇失败，跳过并记录
            raw = ""
            failed.append(page_number)
        pages.append(PdfPage(page=page_number, text=clean_text(raw)))

    if not allow_empty and not any(page.text for page in pages):
        raise PdfLoadError(
            "没能从 %s 抽出任何文字：可能是扫描件（图片型 PDF），需要先做 OCR"
            % pdf_path.name
        )

    try:
        metadata = {str(key): str(value) for key, value in dict(reader.metadata or {}).items()}
    except Exception:
        metadata = {}

    return LoadedPdf(
        path=pdf_path,
        doc_id=compute_doc_id(pdf_path),
        title=_extract_title(reader, pdf_path),
        page_count=total,
        pages=pages,
        metadata=metadata,
        failed_pages=failed,
    )


def iter_pdf_paths(directory: Union[Path, str], *, recursive: bool = True) -> List[Path]:
    """列出一个目录下所有 PDF（按路径排序，保证结果稳定）。"""
    base = Path(directory).expanduser()
    if not base.is_dir():
        return []
    pattern = "**/*" if recursive else "*"
    return sorted(
        item for item in base.glob(pattern)
        if item.is_file() and item.suffix.lower() == ".pdf"
    )
