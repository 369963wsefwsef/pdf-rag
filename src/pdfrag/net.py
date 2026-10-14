"""网络层公共设施：代理解析 + 显式 opener + 连接错误提示。

为什么不能只依赖 urllib 的默认行为
----------------------------------
urllib.request.urlopen **确实**会读 HTTP_PROXY / HTTPS_PROXY / NO_PROXY
（Windows 上的解析顺序是"环境变量优先，其次才是注册表里的系统代理"，
见 CPython 的 getproxies() = getproxies_environment() or getproxies_registry()）。
但实践中有两个坑：

1) 它用的是**模块级全局 opener**，只在第一次请求时构建一次并缓存
   （urllib.request._opener）。如果在这之前有任何代码发过 HTTP 请求，
   之后再设置代理环境变量就完全不生效 —— 对"先 import、后读 .env"的程序
   来说，这是个很难发现的顺序陷阱。
2) 它**不会告诉你最终用了哪个代理**。出错时报的是一句
   "WinError 10061 由于目标计算机积极拒绝"，既没说走没走代理，
   也没说是哪个地址在拒绝，只能靠猜。

所以这里把代理解析做成显式的：自己读环境变量、自己建 ProxyHandler、
每个客户端持有自己的 opener（与全局缓存完全无关），
并且把"最终生效的代理"暴露出来给界面和报错信息用。

配置来源与优先级
----------------
    PDFRAG_PROXY=off                  -> 强制直连，忽略下面所有代理变量
    PDFRAG_PROXY=http://127.0.0.1:7897 -> 同时作为 http / https 代理（简写 host:port 也认）
    HTTPS_PROXY / HTTP_PROXY / ALL_PROXY -> 标准变量，按协议分别生效
    NO_PROXY                          -> 逗号分隔的域名/IP，命中则直连（* 表示全部直连）
    本机地址（localhost / 127.0.0.1 / ::1）永远直连，避免把本地服务（Ollama 等）送去代理
"""

from __future__ import annotations

import os
import urllib.request as urllib_request
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence
from urllib.parse import urlsplit

# 应用自己的开关（优先级最高，方便临时排查）
APP_PROXY_ENV = "PDFRAG_PROXY"
DISABLE_VALUES = frozenset({"off", "none", "no", "disable", "disabled", "direct", "0", "false"})
LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0", "[::1]"})
MASK = "****"


def _lookup(env: Mapping[str, str], name: str) -> str:
    """大小写不敏感地取一个环境变量（.env 里写小写也要认）。"""
    for key in (name, name.lower(), name.upper()):
        value = env.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def mask_proxy(url: str) -> str:
    """打码代理 URL 里的密码（代理也可能带 user:pass）。"""
    if not url:
        return ""
    parts = urlsplit(url)
    if not parts.hostname:
        return url
    auth = ""
    if parts.username:
        auth = parts.username + ((":" + MASK) if parts.password else "") + "@"
    port = (":" + str(parts.port)) if parts.port else ""
    scheme = (parts.scheme + "://") if parts.scheme else ""
    return "%s%s%s%s" % (scheme, auth, parts.hostname, port)


@dataclass(frozen=True)
class ProxyConfig:
    """最终生效的代理设置。"""

    http: str = ""
    https: str = ""
    no_proxy: str = ""
    enabled: bool = True
    source: str = "none"      # app | env | none | off | invalid
    warning: str = ""

    @property
    def active(self) -> bool:
        return self.enabled and bool(self.http or self.https)

    def for_url(self, url: str) -> str:
        """返回访问这个 URL 时应该用的代理；空串表示直连。"""
        if not self.enabled:
            return ""
        scheme = urlsplit(url or "").scheme.lower()
        if scheme == "https":
            return self.https or self.http
        if scheme == "http":
            return self.http or self.https
        return self.http or self.https

    def should_bypass(self, host: str) -> bool:
        """该主机是否绕过代理（本机地址，或命中 NO_PROXY）。"""
        name = (host or "").strip().lower()
        if not name:
            return False
        if name in LOCAL_HOSTS:
            return True          # 本地服务永远直连
        entries = [item.strip().lower() for item in (self.no_proxy or "").split(",")]
        for entry in entries:
            if not entry:
                continue
            if entry == "*":
                return True
            if name == entry or name.endswith("." + entry.lstrip(".")):
                return True
        return False

    def describe(self) -> List[str]:
        """给界面展示的行（代理地址已打码）。"""
        if self.source == "off":
            return ["代理：已关闭（PDFRAG_PROXY=off），全部直连"]
        if not self.active:
            lines = ["代理：未配置（直连）"]
            if self.no_proxy:
                lines.append("NO_PROXY：%s" % self.no_proxy)
            return lines
        lines = []
        if self.https:
            lines.append("HTTPS 代理：%s" % mask_proxy(self.https))
        if self.http:
            lines.append("HTTP 代理：%s" % mask_proxy(self.http))
        if self.no_proxy:
            lines.append("NO_PROXY：%s" % self.no_proxy)
        lines.append("来源：%s" % ("PDFRAG_PROXY" if self.source == "app" else "环境变量"))
        return lines


def read_proxy_config(env: Optional[Mapping[str, str]] = None) -> ProxyConfig:
    """从环境变量解析代理配置（不抛异常，问题通过 warning 字段返回）。"""
    source_env: Mapping[str, str] = os.environ if env is None else env
    no_proxy = _lookup(source_env, "NO_PROXY")

    app_value = _lookup(source_env, APP_PROXY_ENV)
    if app_value:
        lowered = app_value.lower()
        if lowered in DISABLE_VALUES:
            # 强制直连：忽略 HTTP_PROXY / HTTPS_PROXY，也不理会注册表里的系统代理
            return ProxyConfig(no_proxy=no_proxy, enabled=False, source="off")
        if "://" in app_value:
            url = app_value
        elif ":" in app_value:
            url = "http://" + app_value           # 允许写 127.0.0.1:7897 这种简写
        else:
            return ProxyConfig(
                no_proxy=no_proxy, enabled=False, source="invalid",
                warning="%s=%r 无法识别：请写 off，或写成 http://host:port。当前按直连处理。"
                        % (APP_PROXY_ENV, app_value),
            )
        return ProxyConfig(http=url, https=url, no_proxy=no_proxy, source="app")

    http = _lookup(source_env, "HTTP_PROXY")
    https = _lookup(source_env, "HTTPS_PROXY")
    all_proxy = _lookup(source_env, "ALL_PROXY")
    if all_proxy:
        https = https or all_proxy
        http = http or all_proxy
    return ProxyConfig(
        http=http, https=https, no_proxy=no_proxy,
        source="env" if (http or https) else "none",
    )


def build_opener(
    config: Optional[ProxyConfig] = None,
    *,
    target_url: str = "",
) -> urllib_request.OpenerDirector:
    """按配置构建 opener。

    * 该 URL 需要走代理 -> 装一个真的 ProxyHandler（https 会自动走 CONNECT 隧道）；
    * 代理关闭 / 命中 NO_PROXY / 目标是本机地址 -> 装一个**空的** ProxyHandler。
      空 handler 不会拦截任何请求，等于强制直连 —— 这一步很关键：如果系统里
      有代理（环境变量或注册表），不显式关掉的话请求还是会被代理劫走。
    """
    active = config if config is not None else read_proxy_config()
    proxy_url = active.for_url(target_url)
    if not proxy_url or active.should_bypass(urlsplit(target_url or "").hostname or ""):
        return urllib_request.build_opener(urllib_request.ProxyHandler({}))
    scheme = urlsplit(target_url).scheme.lower() or "http"
    return urllib_request.build_opener(urllib_request.ProxyHandler({scheme: proxy_url}))


def is_connection_refused(exc: BaseException) -> bool:
    """连接是否被**直接拒绝**（ConnectionRefusedError / WinError 10061）。

    这种情况重试没有意义 —— 最常见的原因是代理端口没有人监听，
    等一秒再试一次还是同一个结果，不如立刻把原因告诉用户。
    """
    reason = getattr(exc, "reason", exc)
    if isinstance(reason, ConnectionRefusedError):
        return True
    text = str(reason)
    return "10061" in text or "refused" in text.lower()


def connection_error_hint(
    exc: BaseException,
    *,
    url: str = "",
    proxy: Optional[ProxyConfig] = None,
) -> str:
    """把底层连接异常翻译成一句能照着做的提示。"""
    reason = getattr(exc, "reason", exc)
    text = str(reason)
    refused = is_connection_refused(exc)
    host = urlsplit(url or "").hostname or url or "目标主机"
    # 注意：如果这个主机本来就会被跳过代理（本机地址 / 命中 NO_PROXY），
    # 那连接失败跟代理无关，不能把责任推给代理 —— 否则排查方向会被带偏。
    bypassed = proxy is not None and proxy.should_bypass(host)
    proxy_url = "" if (proxy is None or bypassed) else proxy.for_url(url)

    if refused and proxy_url:
        return (
            "连接被拒绝（WinError 10061）：代理 %s 上没有服务在监听。"
            "请确认代理客户端已启动、端口填对了（Clash 常见 7897，V2Ray 常见 10809），"
            "或者改成直连：在 .env 里设 PDFRAG_PROXY=off。" % mask_proxy(proxy_url)
        )
    if refused and host in LOCAL_HOSTS:
        return (
            "连接被拒绝（WinError 10061）：本机 %s 上没有服务在监听"
            "（本机地址自动直连、不走代理）。如果这是本地模型服务（Ollama / LM Studio），"
            "先确认它已经启动。" % host
        )
    if refused:
        return (
            "连接被拒绝（WinError 10061）：%s 拒绝了这个连接。"
            "若本机开了代理客户端，试着在 .env 里设置 HTTPS_PROXY（例如 http://127.0.0.1:7897）；"
            "若走代理反而不通，设 PDFRAG_PROXY=off 直连。" % host
        )
    if proxy_url:
        return "网络错误（走代理 %s）：%s" % (mask_proxy(proxy_url), text)
    return "网络错误：%s" % text
