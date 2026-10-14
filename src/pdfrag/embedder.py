"""第三步（上半）：把文本变成向量。

为什么需要可插拔后端
--------------------
DeepSeek 官方 API 没有 embeddings 接口，向量化必须由别的模型完成。
这里定义统一的 Embedder 协议，上层（retriever / pipeline）不关心具体用的是谁：

    api    -> 任意的 OpenAI 兼容 /embeddings 接口（硅基流动 / DashScope / OpenAI / Ollama）
    local  -> sentence-transformers 本地模型（离线，需要装 torch）
    chroma -> Chroma 自带的 ONNX MiniLM（零配置，但英文模型，中文效果一般）

一个硬性约束：**向量维度必须一致**。换模型等于换维度（bge-m3=1024、MiniLM=384），
往同一个 collection 里混不同维度的向量，检索结果会变成垃圾，所以 vector_store 会做校验。
"""

from __future__ import annotations

import json
import random
import time
from typing import Any, Callable, Dict, List, Optional, Sequence
from urllib import error as urlerror
from urllib import request as urlrequest

from .config import Settings
from .net import (
    ProxyConfig,
    build_opener,
    connection_error_hint,
    is_connection_refused,
    read_proxy_config,
)

DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_RETRIES = 3


class EmbeddingError(RuntimeError):
    """向量化失败：没配 key、接口报错、返回维度不对等。"""


class BaseEmbedder:
    """公共实现：批处理 + 进度回调 + 单条查询。

    子类只实现 _embed_batch()（一次真实调用）即可：批处理、进度、单条查询都在基类里，
    每个后端不必重复实现一遍，也避免"embed 调 embed_batched、后者又回调 embed"这种递归。
    """

    name: str = "base"
    dim: int = 0
    batch_size: int = DEFAULT_BATCH_SIZE

    def _embed_batch(self, batch: Sequence[str]) -> List[List[float]]:  # pragma: no cover
        raise NotImplementedError("子类必须实现 _embed_batch()")

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        return self.embed_batched(texts)

    def embed_query(self, text: str) -> List[float]:
        """检索时给"一个问题"求向量：等价于只 embed 一条。"""
        vectors = self.embed([text])
        if not vectors:
            raise EmbeddingError("向量化返回为空")
        return vectors[0]

    def embed_batched(
        self,
        texts: Sequence[str],
        *,
        batch_size: Optional[int] = None,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> List[List[float]]:
        """按批处理，避免一次发太多把接口或内存打爆。"""
        items = list(texts)
        if not items:
            return []
        step = max(1, int(batch_size or self.batch_size or DEFAULT_BATCH_SIZE))
        vectors: List[List[float]] = []
        for start in range(0, len(items), step):
            vectors.extend(self._embed_batch(items[start:start + step]))
            if on_progress:
                on_progress(len(vectors), len(items))
        return vectors

    def describe(self) -> Dict[str, Any]:
        return {"backend": type(self).__name__, "name": self.name, "dim": self.dim}


class ApiEmbedder(BaseEmbedder):
    """OpenAI 兼容的 /embeddings 接口。

    只要对方实现了 {"model": ..., "input": [...]} -> {"data": [{"embedding": [...]}]}
    就能用，包括硅基流动、阿里云 DashScope 兼容模式、OpenAI、本地 Ollama / LM Studio。
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        dim: int = 0,
        timeout: float = 60.0,
        max_retries: int = DEFAULT_MAX_RETRIES,
        batch_size: int = DEFAULT_BATCH_SIZE,
        debug: bool = False,
        proxy: Optional[ProxyConfig] = None,
    ) -> None:
        if not api_key:
            raise EmbeddingError(
                "Embedding 后端是 api，但 EMBEDDING_API_KEY 是空的。请到 "
                "https://cloud.siliconflow.cn 申请 Key 填进 .env 的 EMBEDDING_API_KEY，"
                "或把 EMBEDDING_BACKEND 改成 chroma 先零配置跑通流程。"
            )
        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.dim = int(dim or 0)
        self.timeout = float(timeout)
        self.max_retries = max(0, int(max_retries))
        self.batch_size = max(1, int(batch_size))
        self.debug = debug
        self.name = "api:" + model

        # 代理在构造时就定下来，并据此建一个**属于本客户端**的 opener。
        # 不用 urllib.request.urlopen 的默认 opener，是因为它只在首次请求时构建并缓存
        # （urllib.request._opener），"设置代理环境变量"与"第一次发请求"的先后顺序
        # 会决定代理到底生不生效 —— 这正是最难排查的那类问题。
        self.proxy = proxy if proxy is not None else read_proxy_config()
        self._opener = build_opener(self.proxy, target_url=self.endpoint)

    @property
    def endpoint(self) -> str:
        return self.base_url + "/embeddings"

    def _payload(self, batch: Sequence[str]) -> bytes:
        payload = {
            "model": self.model,
            "input": list(batch),
            "encoding_format": "float",
        }
        return json.dumps(payload, ensure_ascii=False).encode("utf-8", "replace")

    def _embed_batch(self, batch: Sequence[str]) -> List[List[float]]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": "Bearer " + self.api_key,
            "User-Agent": "pdf-rag/0.1",
        }
        body = self._payload(batch)
        delay = 0.6
        last_error: Optional[EmbeddingError] = None

        for attempt in range(self.max_retries + 1):
            try:
                request = urlrequest.Request(self.endpoint, data=body, headers=headers, method="POST")
                with self._opener.open(request, timeout=self.timeout) as response:
                    raw = response.read().decode("utf-8", "replace")
                return self._parse(json.loads(raw), len(batch))
            except urlerror.HTTPError as exc:
                detail = ""
                try:
                    detail = exc.read().decode("utf-8", "replace")[:300]
                except Exception:
                    pass
                if exc.code in (401, 403):
                    raise EmbeddingError(
                        "Embedding 鉴权失败（HTTP %d）：检查 EMBEDDING_API_KEY 是否正确。%s"
                        % (exc.code, detail)
                    ) from exc
                last_error = EmbeddingError("Embedding 接口返回 HTTP %d：%s" % (exc.code, detail))
                retryable = exc.code == 429 or exc.code >= 500
                if not retryable:
                    raise last_error from exc
            except EmbeddingError:
                raise
            except (urlerror.URLError, TimeoutError, OSError) as exc:
                last_error = EmbeddingError(
                    "Embedding " + connection_error_hint(exc, url=self.endpoint, proxy=self.proxy)
                )
                # 端口没人监听这种"直接拒绝"，重试三次只是白等，立刻报出去
                if is_connection_refused(exc):
                    raise last_error from exc
            except json.JSONDecodeError as exc:
                last_error = EmbeddingError("Embedding 返回的不是 JSON：%s" % exc)

            if attempt >= self.max_retries:
                break
            time.sleep(delay + random.uniform(0, 0.3))
            delay = min(delay * 2, 8.0)

        raise last_error or EmbeddingError("Embedding 请求失败")

    def _parse(self, data: Any, expected: int) -> List[List[float]]:
        if not isinstance(data, dict):
            raise EmbeddingError("Embedding 接口返回的不是 JSON 对象")
        error = data.get("error")
        if isinstance(error, (dict, str)):
            message = error.get("message") if isinstance(error, dict) else error
            raise EmbeddingError("Embedding 接口报错：%s" % message)

        items = data.get("data") or []
        if not isinstance(items, list) or len(items) != expected:
            raise EmbeddingError(
                "Embedding 接口返回条数不对：期望 %d，实际 %d" % (expected, len(items))
            )
        try:
            items = sorted(items, key=lambda item: item.get("index", 0))
        except Exception:
            pass

        vectors: List[List[float]] = []
        for item in items:
            vector = item.get("embedding") if isinstance(item, dict) else None
            if not isinstance(vector, (list, tuple)) or not vector:
                raise EmbeddingError("Embedding 接口返回的 embedding 字段格式不对")
            vectors.append([float(value) for value in vector])

        actual_dim = len(vectors[0])
        if self.dim and actual_dim != self.dim:
            raise EmbeddingError(
                "Embedding 维度与配置不符：模型返回 %d 维，但 EMBEDDING_DIM=%d。"
                "请把 EMBEDDING_DIM 改成 %d（或删掉这一行让程序自动识别）。"
                % (actual_dim, self.dim, actual_dim)
            )
        if not self.dim:
            self.dim = actual_dim
        return vectors


class ChromaEmbedder(BaseEmbedder):
    """Chroma 自带的 ONNX MiniLM：零配置、离线，但英文为主，中文检索效果差。

    首次使用会下载约 80MB 模型权重（之后走本地缓存）。
    """

    def __init__(self, *, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from chromadb.utils import embedding_functions
        except Exception as exc:  # pragma: no cover - 依赖缺失
            raise EmbeddingError("chromadb 未安装或导入失败：%s" % exc) from exc
        try:
            self._function = embedding_functions.DefaultEmbeddingFunction()
        except Exception as exc:
            raise EmbeddingError(
                "初始化 Chroma 默认 Embedding 失败（首次使用需要联网下载模型）：%s" % exc
            ) from exc
        self.model_name = model_name
        self.name = "chroma:default-minilm"
        self.dim = 384

    def _embed_batch(self, batch: Sequence[str]) -> List[List[float]]:
        items = list(batch)
        if not items:
            return []
        vectors = self._function(items)
        return [[float(value) for value in vector] for vector in vectors]


class LocalEmbedder(BaseEmbedder):
    """sentence-transformers 本地模型：离线、中文效果好，代价是要装 torch（约 2GB）。"""

    def __init__(self, *, model_name: str = "BAAI/bge-small-zh-v1.5", dim: int = 0) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            raise EmbeddingError(
                "EMBEDDING_BACKEND=local 需要先安装 sentence-transformers："
                "pip install sentence-transformers（会一并安装 torch，约 2GB）"
            ) from exc
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self.dim = int(dim or self._model.get_sentence_embedding_dimension() or 0)
        self.name = "local:" + model_name

    def _embed_batch(self, batch: Sequence[str]) -> List[List[float]]:
        items = list(batch)
        if not items:
            return []
        vectors = self._model.encode(items, normalize_embeddings=True, show_progress_bar=False)
        return [[float(value) for value in vector] for vector in vectors]


def build_embedder(settings: Settings) -> BaseEmbedder:
    """根据配置造出对应的 Embedder（工厂）。"""
    backend = (settings.embedding_backend or "").lower()
    if backend == "api":
        return ApiEmbedder(
            api_key=settings.embedding_api_key,
            base_url=settings.embedding_base_url,
            model=settings.embedding_model,
            dim=settings.embedding_dim,
            proxy=settings.proxy,
        )
    if backend == "local":
        return LocalEmbedder(model_name=settings.embedding_model)
    if backend == "chroma":
        return ChromaEmbedder()
    raise EmbeddingError("未知的 EMBEDDING_BACKEND：%r（可选 api / local / chroma）" % backend)
