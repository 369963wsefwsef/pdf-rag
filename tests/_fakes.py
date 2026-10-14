"""测试用的替身：假 Embedder、假大模型、构造 Settings 的小工具。

它们让整套测试**完全离线**：不联网、不下载模型、不花 API 费用，而且很快。

FakeEmbedder 用的是"词袋 + md5 哈希桶"：同一个词永远落到同一个维度上，
所以两段文本共享的词越多，余弦相似度越高 —— 足以断言检索排序是否正确。
"""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

from pdfrag.config import Settings
from pdfrag.net import ProxyConfig

_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> List[str]:
    """中文按字、英文/数字按词切分。"""
    return _TOKEN_RE.findall((text or "").lower())


class FakeEmbedder:
    """确定性词袋向量（默认 64 维），不依赖任何外部服务。"""

    name = "fake:bag-of-words"

    def __init__(self, dim: int = 64) -> None:
        self.dim = int(dim)
        self.calls = 0
        self.embedded_texts: List[str] = []

    def _vector(self, text: str) -> List[float]:
        vector = [0.0] * self.dim
        for token in tokenize(text):
            bucket = int(hashlib.md5(token.encode("utf-8")).hexdigest()[:8], 16) % self.dim
            vector[bucket] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        items = list(texts)
        self.calls += 1
        self.embedded_texts.extend(items)
        return [self._vector(item) for item in items]

    def embed_query(self, text: str) -> List[float]:
        self.calls += 1
        self.embedded_texts.append(text)
        return self._vector(text)


class FakeLLM:
    """假大模型：记录收到的 messages，按固定文本"流式"吐出。"""

    def __init__(self, reply: str = "根据资料，答案是 42。[1]") -> None:
        self.reply = reply
        self.messages: List[Dict[str, str]] = []
        self.calls = 0

    def _remember(self, messages: Sequence[Dict[str, str]]) -> None:
        self.calls += 1
        self.messages = [dict(item) for item in messages]

    def stream(self, messages: Sequence[Dict[str, str]], **_: Any):
        self._remember(messages)
        for index in range(0, len(self.reply), 4):
            yield self.reply[index:index + 4]

    def chat(self, messages: Sequence[Dict[str, str]], **_: Any) -> Dict[str, Any]:
        self._remember(messages)
        return {
            "content": self.reply,
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "model": "fake",
        }

    @property
    def last_user_content(self) -> str:
        for message in reversed(self.messages):
            if message.get("role") == "user":
                return message.get("content", "")
        return ""


def make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    """造一个指向临时目录的 Settings，供各测试复用。"""
    base: Dict[str, Any] = dict(
        deepseek_api_key="",
        deepseek_base_url="https://api.deepseek.com",
        deepseek_model="deepseek-chat",
        embedding_backend="api",
        embedding_api_key="",
        embedding_base_url="https://embedding.invalid/v1",
        embedding_model="fake-model",
        embedding_dim=64,
        chunk_size=120,
        chunk_overlap=30,
        top_k=3,
        collection_name="pytest_collection",
        data_dir=tmp_path,
        upload_dir=tmp_path / "uploads",
        chroma_dir=tmp_path / "chroma",
        proxy=ProxyConfig(enabled=False),   # 测试里不碰真实环境变量/系统代理
    )
    base.update(overrides)
    return Settings(**base)


def make_chunk(index: int, text: str, *, page: int = 1, doc_id: str = "doc_aaaa",
               source: str = "sample.pdf"):
    """直接造一个 Chunk（跳过 PDF 解析），用于向量库/检索的单元测试。"""
    from pdfrag.chunker import Chunk

    return Chunk(
        id="%s-p%d-c%d" % (doc_id, page, index),
        text=text,
        doc_id=doc_id,
        source=source,
        page=page,
        chunk_index=index,
        char_start=0,
        char_end=len(text),
    )
