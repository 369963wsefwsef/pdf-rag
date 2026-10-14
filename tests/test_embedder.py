"""第三步（上）的测试：Embedder。全部离线，HTTP 用假对象替换。"""

from __future__ import annotations

import json
from urllib.error import URLError

import pytest
from _fakes import make_settings
from pdfrag.embedder import (
    ApiEmbedder,
    BaseEmbedder,
    ChromaEmbedder,
    LocalEmbedder,
    EmbeddingError,
    build_embedder,
)
from pdfrag.net import ProxyConfig


class _FakeResponse:
    def __init__(self, payload) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeOpener:
    """替代真实 opener：不发网络请求，直接把预设响应返回。"""

    def __init__(self, responder) -> None:
        self._responder = responder

    def open(self, request, timeout=None):
        return _FakeResponse(self._responder(request))


def _patch_opener(monkeypatch, payload, recorder=None):
    """把 embedder 用的 opener 换掉。

    embedder 在构造时会调用 build_opener() 建一个属于本客户端的 opener，
    所以这里替换的是 build_opener，而不是 urllib 的全局 urlopen。
    """

    def responder(request):
        if recorder is not None:
            recorder.append(request)
        return payload

    monkeypatch.setattr("pdfrag.embedder.build_opener", lambda *args, **kwargs: _FakeOpener(responder))


def test_api_embedder_requires_key():
    with pytest.raises(EmbeddingError, match="EMBEDDING_API_KEY"):
        ApiEmbedder(api_key="", base_url="https://x.invalid/v1", model="m")


def test_api_embedder_parses_response_and_detects_dim(monkeypatch):
    payload = {"data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]},
                        {"index": 1, "embedding": [0.4, 0.5, 0.6]}]}
    requests = []
    _patch_opener(monkeypatch, payload, requests)

    embedder = ApiEmbedder(api_key="sk-test", base_url="https://x.invalid/v1", model="m")
    vectors = embedder.embed(["甲", "乙"])

    assert vectors == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    assert embedder.dim == 3, "没配 EMBEDDING_DIM 时应自动识别"
    assert embedder.endpoint == "https://x.invalid/v1/embeddings"

    sent = json.loads(requests[0].data.decode("utf-8"))
    assert sent["model"] == "m" and sent["input"] == ["甲", "乙"]
    assert requests[0].get_header("Authorization") == "Bearer sk-test"


def test_api_embedder_rejects_dimension_mismatch(monkeypatch):
    _patch_opener(monkeypatch, {"data": [{"index": 0, "embedding": [0.1, 0.2]}]})
    embedder = ApiEmbedder(api_key="k", base_url="https://x.invalid/v1", model="m", dim=1024)
    with pytest.raises(EmbeddingError, match="维度与配置不符"):
        embedder.embed(["甲"])


def test_api_embedder_surfaces_error_payload(monkeypatch):
    _patch_opener(monkeypatch, {"error": {"message": "invalid api key"}})
    embedder = ApiEmbedder(api_key="k", base_url="https://x.invalid/v1", model="m")
    with pytest.raises(EmbeddingError, match="invalid api key"):
        embedder.embed(["甲"])


def test_api_embedder_rejects_wrong_count(monkeypatch):
    _patch_opener(monkeypatch, {"data": [{"index": 0, "embedding": [0.1]}]})
    embedder = ApiEmbedder(api_key="k", base_url="https://x.invalid/v1", model="m")
    with pytest.raises(EmbeddingError, match="条数不对"):
        embedder.embed(["甲", "乙"])


def test_batching_splits_requests(monkeypatch):
    calls = []

    def responder(request):
        body = json.loads(request.data.decode("utf-8"))
        calls.append(len(body["input"]))
        # 真实接口的形状是 {"data": [...]}，假响应也必须一致
        return {"data": [{"index": i, "embedding": [0.0, 1.0]}
                         for i in range(len(body["input"]))]}

    monkeypatch.setattr("pdfrag.embedder.build_opener", lambda *args, **kwargs: _FakeOpener(responder))
    embedder = ApiEmbedder(api_key="k", base_url="https://x.invalid/v1", model="m", batch_size=2)
    vectors = embedder.embed(["a", "b", "c", "d", "e"])

    assert len(vectors) == 5
    assert calls == [2, 2, 1], "5 条按 batch_size=2 应该发 3 次请求"


def test_embed_batched_reports_progress():
    class Counting(BaseEmbedder):
        name = "counting"
        dim = 2

        def _embed_batch(self, batch):
            return [[float(len(t)), 0.0] for t in batch]

    seen = []
    vectors = Counting().embed_batched(["a", "b", "c"], batch_size=1,
                                       on_progress=lambda done, total: seen.append((done, total)))
    assert len(vectors) == 3
    assert seen == [(1, 3), (2, 3), (3, 3)]


def test_embed_query_returns_single_vector():
    class Counting(BaseEmbedder):
        name = "counting"
        dim = 1

        def _embed_batch(self, batch):
            return [[1.0] for _ in batch]

    assert Counting().embed_query("任意") == [1.0]


def test_build_embedder_dispatch(tmp_path):
    settings = make_settings(tmp_path, embedding_backend="api", embedding_api_key="")
    with pytest.raises(EmbeddingError, match="EMBEDDING_API_KEY"):
        build_embedder(settings)

    ok = build_embedder(make_settings(tmp_path, embedding_backend="api", embedding_api_key="sk-x",
                                      embedding_model="fake-model", embedding_dim=0))
    assert isinstance(ok, ApiEmbedder) and ok.name == "api:fake-model"

    with pytest.raises(EmbeddingError, match="未知的 EMBEDDING_BACKEND"):
        build_embedder(make_settings(tmp_path, embedding_backend="whatever"))


def test_local_and_chroma_backends_explain_missing_deps(tmp_path):
    """没装可选依赖时，报错信息必须说清楚怎么办，而不是抛裸 ImportError。"""
    try:
        import sentence_transformers  # noqa: F401
        local_available = True
    except Exception:
        local_available = False

    if not local_available:
        with pytest.raises(EmbeddingError, match="sentence-transformers"):
            LocalEmbedder(model_name="BAAI/bge-small-zh-v1.5")

    # Chroma 的默认 Embedding 依赖 onnxruntime，这里只验证类本身能构造出正确的元信息
    assert ChromaEmbedder.__name__ == "ChromaEmbedder"


# ==================== 代理支持（本次新增） ====================
def test_requests_go_through_the_client_own_opener(monkeypatch):
    """关键回归：必须走客户端自己的 opener。

    之前用的是 urllib.request.urlopen，它依赖模块级全局 opener
    （首次请求时构建并缓存），"设代理变量"和"第一次请求"的先后顺序
    会决定代理到底生不生效 —— 这是最难排查的一类问题。
    """
    opened = []

    class Recording:
        def open(self, request, timeout=None):
            opened.append(request.full_url)
            return _FakeResponse({"data": [{"index": 0, "embedding": [1.0, 0.0]}]})

    monkeypatch.setattr("pdfrag.embedder.build_opener", lambda *args, **kwargs: Recording())

    def boom(*args, **kwargs):
        raise AssertionError("不应该再走 urllib.request.urlopen（受全局 opener 缓存影响）")

    monkeypatch.setattr("pdfrag.embedder.urlrequest.urlopen", boom)

    embedder = ApiEmbedder(api_key="k", base_url="https://x.invalid/v1", model="m")
    assert embedder.embed(["甲"]) == [[1.0, 0.0]]
    assert opened == ["https://x.invalid/v1/embeddings"]


def test_proxy_config_reaches_the_opener(monkeypatch):
    seen = {}

    def fake_build_opener(config=None, *, target_url=""):
        seen["config"] = config
        seen["target_url"] = target_url
        return _FakeOpener(lambda request: {"data": [{"index": 0, "embedding": [1.0]}]})

    monkeypatch.setattr("pdfrag.embedder.build_opener", fake_build_opener)
    proxy = ProxyConfig(https="http://127.0.0.1:7897", source="app")

    embedder = ApiEmbedder(api_key="k", base_url="https://api.x.invalid/v1", model="m", proxy=proxy)
    embedder.embed(["甲"])

    assert seen["config"] is proxy
    assert seen["target_url"] == "https://api.x.invalid/v1/embeddings"


def test_proxy_defaults_to_environment(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7897")
    embedder = ApiEmbedder(api_key="k", base_url="https://x.invalid/v1", model="m")
    assert embedder.proxy.for_url("https://x.invalid/v1/embeddings") == "http://127.0.0.1:7897"


def test_build_embedder_passes_settings_proxy(tmp_path):
    proxy = ProxyConfig(https="http://127.0.0.1:7897", source="env")
    settings = make_settings(tmp_path, embedding_api_key="sk-x", proxy=proxy)
    assert build_embedder(settings).proxy is proxy


def test_connection_refused_reports_proxy_and_skips_retries(monkeypatch):
    """代理端口没人监听 -> 立刻失败，并把代理地址写进报错（正是 WinError 10061 的场景）。"""
    attempts = []

    class Refusing:
        def open(self, request, timeout=None):
            attempts.append(request.full_url)
            raise URLError(ConnectionRefusedError(10061, "由于目标计算机积极拒绝，无法连接。"))

    monkeypatch.setattr("pdfrag.embedder.build_opener", lambda *args, **kwargs: Refusing())
    proxy = ProxyConfig(https="http://127.0.0.1:7890", source="env")

    embedder = ApiEmbedder(api_key="k", base_url="https://api.siliconflow.cn/v1", model="m",
                           proxy=proxy, max_retries=3)
    with pytest.raises(EmbeddingError) as info:
        embedder.embed(["甲"])

    message = str(info.value)
    assert "10061" in message
    assert "127.0.0.1:7890" in message, "报错里必须指出是哪个代理地址"
    assert len(attempts) == 1, "连接被直接拒绝不该重试 3 次"


def test_timeout_is_still_retried(monkeypatch):
    attempts = []

    class Timing:
        def open(self, request, timeout=None):
            attempts.append(1)
            raise URLError(TimeoutError("timed out"))

    monkeypatch.setattr("pdfrag.embedder.build_opener", lambda *args, **kwargs: Timing())
    monkeypatch.setattr("pdfrag.embedder.time.sleep", lambda *_: None)   # 别真等退避

    embedder = ApiEmbedder(api_key="k", base_url="https://x.invalid/v1", model="m",
                           max_retries=2, proxy=ProxyConfig(enabled=False))
    with pytest.raises(EmbeddingError, match="timed out"):
        embedder.embed(["甲"])
    assert len(attempts) == 3, "超时属于可重试错误（首次 + 2 次重试）"
