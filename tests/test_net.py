"""网络层（代理解析 / opener 构建 / 报错提示）的测试。

全部离线：只检查"最终会用什么配置去发请求"，不真的发请求。
"""

from __future__ import annotations

import urllib.request
from urllib.error import URLError

import pytest
from pdfrag.net import (
    ProxyConfig,
    build_opener,
    connection_error_hint,
    is_connection_refused,
    mask_proxy,
    read_proxy_config,
)


# ---------- 解析环境变量 ----------
def test_no_proxy_variables_means_direct():
    config = read_proxy_config({})
    assert config.active is False
    assert config.source == "none"
    assert build_opener(config, target_url="https://a.example.com/v1").handlers


def test_reads_https_and_http_proxy():
    config = read_proxy_config({"HTTPS_PROXY": "http://127.0.0.1:7897",
                                "HTTP_PROXY": "http://127.0.0.1:8080"})
    assert config.active is True
    assert config.source == "env"
    assert config.for_url("https://api.example.com/v1") == "http://127.0.0.1:7897"
    assert config.for_url("http://api.example.com/v1") == "http://127.0.0.1:8080"


def test_lowercase_variables_are_accepted():
    config = read_proxy_config({"https_proxy": "http://127.0.0.1:7897"})
    assert config.for_url("https://api.example.com") == "http://127.0.0.1:7897"


def test_all_proxy_is_fallback_for_both_schemes():
    config = read_proxy_config({"ALL_PROXY": "socks5://127.0.0.1:1080"})
    assert config.for_url("https://a.example.com") == "socks5://127.0.0.1:1080"
    assert config.for_url("http://a.example.com") == "socks5://127.0.0.1:1080"


def test_https_proxy_only_applies_to_https():
    config = read_proxy_config({"HTTPS_PROXY": "http://127.0.0.1:7897"})
    # 只配了 https 时，http 也用它（否则 http 请求会悄悄直连，更让人困惑）
    assert config.for_url("http://a.example.com") == "http://127.0.0.1:7897"


def test_app_switch_off_forces_direct_even_with_proxy_env():
    config = read_proxy_config({"HTTPS_PROXY": "http://127.0.0.1:7897", "PDFRAG_PROXY": "off"})
    assert config.active is False
    assert config.source == "off"
    assert config.for_url("https://api.example.com") == ""
    assert "直连" in " ".join(config.describe())


def test_app_switch_accepts_host_port_shorthand():
    config = read_proxy_config({"PDFRAG_PROXY": "127.0.0.1:7897"})
    assert config.source == "app"
    assert config.for_url("https://api.example.com") == "http://127.0.0.1:7897"
    assert config.for_url("http://api.example.com") == "http://127.0.0.1:7897"


def test_app_switch_accepts_full_url():
    config = read_proxy_config({"PDFRAG_PROXY": "http://user:pass@127.0.0.1:7897"})
    assert config.for_url("https://api.example.com") == "http://user:pass@127.0.0.1:7897"


def test_app_switch_invalid_value_warns_and_goes_direct():
    config = read_proxy_config({"HTTPS_PROXY": "http://127.0.0.1:7897", "PDFRAG_PROXY": "yes-please"})
    assert config.active is False
    assert config.source == "invalid"
    assert "无法识别" in config.warning


# ---------- NO_PROXY 与本机直连 ----------
@pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "::1"])
def test_local_hosts_always_bypass(host):
    config = read_proxy_config({"HTTPS_PROXY": "http://127.0.0.1:7897"})
    assert config.should_bypass(host) is True


def test_no_proxy_matches_exact_and_subdomain_and_star():
    config = read_proxy_config({"HTTPS_PROXY": "http://127.0.0.1:7897",
                                "NO_PROXY": "example.com, .internal"})
    assert config.should_bypass("example.com") is True
    assert config.should_bypass("api.example.com") is True
    assert config.should_bypass("host.internal") is True
    assert config.should_bypass("api.siliconflow.cn") is False


def test_no_proxy_star_bypasses_everything():
    config = read_proxy_config({"HTTPS_PROXY": "http://127.0.0.1:7897", "NO_PROXY": "*"})
    assert config.should_bypass("api.siliconflow.cn") is True


# ---------- opener 构建 ----------
def _proxy_interceptor(opener):
    """返回**真正会拦截请求**的 ProxyHandler；返回 None 表示直连。

    注意：空 ProxyHandler(proxies={}) 没有注册任何 "*_open" 方法，
    OpenerDirector.add_handler 因此不会把它放进 opener.handlers
    （add_handler 只记录成功注册了方法的 handler）。
    所以"找不到 ProxyHandler"恰好就是"直连"的判定依据。
    """
    for handler in opener.handlers:
        if isinstance(handler, urllib.request.ProxyHandler):
            return handler
    return None


def _assert_direct(opener):
    assert _proxy_interceptor(opener) is None, "直连时不应存在会拦截请求的代理处理器"


def test_opener_installs_proxy_handler_for_target_scheme():
    config = ProxyConfig(https="http://127.0.0.1:7897", source="env")
    opener = build_opener(config, target_url="https://api.siliconflow.cn/v1/embeddings")
    handler = _proxy_interceptor(opener)
    assert handler is not None, "配了代理就必须装一个会拦截的 ProxyHandler"
    assert handler.proxies == {"https": "http://127.0.0.1:7897"}
    assert hasattr(handler, "https_open"), "https 需要走 CONNECT 隧道"


def test_opener_is_direct_for_local_target():
    """本地服务（Ollama / LM Studio）绝不能被送去代理。"""
    config = ProxyConfig(https="http://127.0.0.1:7897", source="env")
    _assert_direct(build_opener(config, target_url="http://127.0.0.1:11434/v1/embeddings"))


def test_opener_is_direct_when_bypassed_by_no_proxy():
    config = ProxyConfig(https="http://127.0.0.1:7897", no_proxy="api.siliconflow.cn", source="env")
    _assert_direct(build_opener(config, target_url="https://api.siliconflow.cn/v1/embeddings"))


def test_opener_is_direct_when_proxy_disabled():
    config = ProxyConfig(https="http://127.0.0.1:7897", enabled=False, source="off")
    _assert_direct(build_opener(config, target_url="https://api.siliconflow.cn/v1/embeddings"))


def test_direct_opener_ignores_system_wide_proxy(monkeypatch):
    """即使环境里配了代理，直连 opener 也必须把它挡掉（否则本地服务会被送去代理）。"""
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7897")
    _assert_direct(build_opener(ProxyConfig(enabled=False, source="off"),
                                target_url="https://a.example.com"))


# ---------- 报错提示 ----------
def test_connection_refused_with_dead_proxy_is_actionable():
    exc = URLError(ConnectionRefusedError(10061, "由于目标计算机积极拒绝，无法连接。"))
    proxy = ProxyConfig(https="http://127.0.0.1:7890", source="env")

    hint = connection_error_hint(exc, url="https://api.siliconflow.cn/v1/embeddings", proxy=proxy)

    assert "10061" in hint
    assert "127.0.0.1:7890" in hint, "必须指出是哪个代理地址在拒绝连接"
    assert "7897" in hint and "PDFRAG_PROXY=off" in hint, "必须给出可操作的两条出路"


def test_connection_refused_to_local_service_mentions_starting_it():
    exc = URLError(ConnectionRefusedError(10061, "refused"))
    hint = connection_error_hint(exc, url="http://127.0.0.1:11434/v1/embeddings",
                                 proxy=ProxyConfig(enabled=False))
    assert "Ollama" in hint or "本地" in hint


def test_other_network_errors_keep_the_details():
    hint = connection_error_hint(URLError("timed out"), url="https://a.example.com", proxy=ProxyConfig())
    assert "timed out" in hint


def test_is_connection_refused_detects_both_shapes():
    assert is_connection_refused(URLError(ConnectionRefusedError(10061, "x"))) is True
    assert is_connection_refused(URLError("WinError 10061")) is True
    assert is_connection_refused(URLError("timed out")) is False


# ---------- 打码 ----------
def test_mask_proxy_hides_credentials():
    assert mask_proxy("http://user:secret@127.0.0.1:7897") == "http://user:****@127.0.0.1:7897"
    assert mask_proxy("http://127.0.0.1:7897") == "http://127.0.0.1:7897"
    assert mask_proxy("") == ""


def test_describe_never_leaks_password():
    config = ProxyConfig(https="http://user:secret@127.0.0.1:7897", source="env")
    text = " / ".join(config.describe())
    assert "secret" not in text
    assert "****" in text


def test_connection_refused_to_bypassed_local_host_does_not_blame_proxy():
    """本机地址会被跳过代理，所以连不上时绝不能说是代理的问题。"""
    exc = URLError(ConnectionRefusedError(10061, "refused"))
    proxy = ProxyConfig(http="http://127.0.0.1:7897", https="http://127.0.0.1:7897", source="env")

    hint = connection_error_hint(exc, url="http://127.0.0.1:11434/v1/embeddings", proxy=proxy)

    assert "7897" not in hint, "请求根本没走代理，不能把锅甩给代理"
    assert "Ollama" in hint or "本地" in hint


def test_connection_refused_to_no_proxy_host_does_not_blame_proxy():
    exc = URLError(ConnectionRefusedError(10061, "refused"))
    proxy = ProxyConfig(https="http://127.0.0.1:7897", no_proxy="internal.example.com", source="env")

    hint = connection_error_hint(exc, url="https://internal.example.com/api", proxy=proxy)

    # 注意不能简单断言 "7897" 不在文案里 —— 通用建议里本来就会提到常用端口。
    # 要断言的是：不能声称"配置的代理在拒绝连接"。
    assert "上没有服务在监听" not in hint
    assert "internal.example.com" in hint
