"""第四步（下半）：调用 DeepSeek 生成答案。

沿用项目 1 的写法：标准库 urllib + SSE 逐行解析，不依赖 openai SDK。
这里额外提供两件事：
* stream() 是真正的生成器（逐块 yield），Streamlit 才能一边收一边显示；
* build_rag_messages() 负责把检索到的资料塞进 prompt，并约束模型"只依据资料回答、标注编号"。
"""

from __future__ import annotations

import json
import random
import time
from typing import Any, Dict, Iterator, List, Optional, Sequence
from urllib import error as urlerror
from urllib import request as urlrequest

from .net import ProxyConfig, build_opener, connection_error_hint, read_proxy_config

DEFAULT_TEMPERATURE = 0.2   # 知识库问答要稳，比闲聊低得多

SYSTEM_PROMPT = """你是一个严谨的知识库问答助手。请遵守以下规则：
1. 只依据【参考资料】中的内容回答，不要编造资料里没有的信息。
2. 如果参考资料不足以回答，就明确说"资料中没有找到相关内容"，不要猜测。
3. 引用资料时用方括号标注编号，例如 [1] 或 [1][3]，编号对应参考资料的序号。
4. 用简体中文回答，条理清晰、直接给结论，不要复述问题。
"""


class LLMError(RuntimeError):
    """调用大模型失败。"""


def build_rag_messages(question: str, context_text: str) -> List[Dict[str, str]]:
    """组装发给大模型的消息：system 定规则，user 给资料 + 问题。"""
    content = (
        "【参考资料】\n%s\n\n【问题】\n%s\n\n"
        "请只依据上面的参考资料回答；回答里用 [编号] 标注依据；"
        "如果资料不足，直接说明资料中没有相关内容。"
    ) % (context_text.strip() or "（没有检索到任何资料）", question.strip())
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


class DeepSeekClient:
    """DeepSeek Chat Completions 客户端（OpenAI 兼容协议）。"""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        timeout: float = 90.0,
        max_retries: int = 3,
        temperature: float = DEFAULT_TEMPERATURE,
        debug: bool = False,
        proxy: Optional[ProxyConfig] = None,
    ) -> None:
        if not api_key or not api_key.strip():
            raise LLMError("缺少 DEEPSEEK_API_KEY，无法生成答案。请在 .env 里填好。")
        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = float(timeout)
        self.max_retries = max(0, int(max_retries))
        self.temperature = float(temperature)
        self.debug = debug

        # 与 ApiEmbedder 同样的理由：用本客户端自己的 opener，而不是 urllib 的全局缓存
        self.proxy = proxy if proxy is not None else read_proxy_config()
        self._opener = build_opener(self.proxy, target_url=self.endpoint)

    @property
    def endpoint(self) -> str:
        return self.base_url + "/chat/completions"

    # -- 请求 --------------------------------------------------------
    def _payload(
        self,
        messages: Sequence[Dict[str, str]],
        *,
        stream: bool,
        temperature: Optional[float],
        max_tokens: Optional[int],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "stream": bool(stream),
            "temperature": float(self.temperature if temperature is None else temperature),
        }
        if max_tokens:
            payload["max_tokens"] = int(max_tokens)
        if stream:
            # 不加这个，流式响应里不会带 usage
            payload["stream_options"] = {"include_usage": True}
        return payload

    def _open(self, payload: Dict[str, Any], *, stream: bool) -> Any:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8", "replace")
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if stream else "application/json",
            "Authorization": "Bearer " + self.api_key,
            "User-Agent": "pdf-rag/0.1",
        }
        request = urlrequest.Request(self.endpoint, data=body, headers=headers, method="POST")
        try:
            return self._opener.open(request, timeout=self.timeout)
        except urlerror.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:300]
            except Exception:
                pass
            if exc.code in (401, 403):
                raise LLMError(
                    "大模型鉴权失败（HTTP %d）：检查 DEEPSEEK_API_KEY。%s" % (exc.code, detail)
                ) from exc
            raise LLMError("大模型接口返回 HTTP %d：%s" % (exc.code, detail)) from exc
        except (urlerror.URLError, TimeoutError, OSError) as exc:
            raise LLMError(
                "大模型 " + connection_error_hint(exc, url=self.endpoint, proxy=self.proxy)
            ) from exc

    def _open_with_retry(self, payload: Dict[str, Any], *, stream: bool) -> Any:
        """只在"连接还没建立起来"时重试；已经开始输出的流不能重来。"""
        delay = 0.6
        last: Optional[LLMError] = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._open(payload, stream=stream)
            except LLMError as exc:
                last = exc
                message = str(exc)
                # 代理端口没人监听这类"直接拒绝"重试没有意义；其余连接类错误（超时等）才退避重试
                refused = "10061" in message or "被拒绝" in message
                retryable = (not refused) and (
                    "HTTP 429" in message
                    or "HTTP 5" in message
                    or "连接" in message
                )
                if not retryable or attempt >= self.max_retries:
                    raise
            time.sleep(delay + random.uniform(0, 0.3))
            delay = min(delay * 2, 8.0)
        raise last or LLMError("调用大模型失败")

    # -- 对外接口 ----------------------------------------------------
    def stream(
        self,
        messages: Sequence[Dict[str, str]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Iterator[str]:
        """真正的流式：逐段 yield 文本（生成器）。"""
        payload = self._payload(messages, stream=True, temperature=temperature, max_tokens=max_tokens)
        with self._open_with_retry(payload, stream=True) as response:
            for line in response:
                text = line.decode("utf-8", "replace").strip()
                if not text or text.startswith(":"):
                    continue
                if text.startswith("data:"):
                    text = text[5:].strip()
                elif not text.startswith("{"):
                    continue
                if text == "[DONE]":
                    break
                try:
                    chunk = json.loads(text)
                except json.JSONDecodeError:
                    continue
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    piece = delta.get("content")
                    if piece:
                        yield piece

    def chat(
        self,
        messages: Sequence[Dict[str, str]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """一次性拿到完整答案（stream=True 时内部复用流式通道，只有一条代码路径）。"""
        if stream:
            parts = list(self.stream(messages, temperature=temperature, max_tokens=max_tokens))
            return {"content": "".join(parts), "usage": {}, "model": self.model}

        payload = self._payload(messages, stream=False, temperature=temperature, max_tokens=max_tokens)
        with self._open_with_retry(payload, stream=False) as response:
            raw = response.read().decode("utf-8", "replace")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMError("大模型返回的不是合法 JSON：%s" % raw[:200]) from exc
        choices = data.get("choices") or [{}]
        message = (choices[0] or {}).get("message") or {}
        return {
            "content": message.get("content") or "",
            "usage": data.get("usage") or {},
            "model": data.get("model") or self.model,
            "finish_reason": (choices[0] or {}).get("finish_reason"),
        }
