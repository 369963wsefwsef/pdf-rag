"""配置集中管理。

约定：所有环境变量只在本模块读一次，其它模块统一依赖 Settings 对象。
好处是测试可以直接构造 Settings，不必去摆弄真实环境变量。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .net import ProxyConfig, read_proxy_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EMBEDDING_BACKENDS = ("api", "local", "chroma")

# 支持入库的文件格式。界面上的上传类型、pipeline 的后缀分发都以此为准。
SUPPORTED_SUFFIXES = (".pdf", ".docx")


class UnsupportedFormatError(RuntimeError):
    """文件格式不支持（例如老式 .doc、.txt、图片等）。"""


DEFAULTS = {
    "deepseek_base_url": "https://api.deepseek.com",
    "deepseek_model": "deepseek-chat",
    "embedding_base_url": "https://api.siliconflow.cn/v1",
    "embedding_model": "BAAI/bge-m3",
    "embedding_dim": "1024",
    "chunk_size": "800",
    "chunk_overlap": "150",
    "top_k": "5",
    "collection_name": "pdf_knowledge_base",
}


class ConfigError(RuntimeError):
    """配置缺失或非法。"""


def load_dotenv(path: Optional[Path] = None) -> None:
    """极简 .env 读取：只补缺失的环境变量，不覆盖已有的。"""
    env_path = path or (PROJECT_ROOT / ".env")
    if not env_path.is_file():
        return
    try:
        lines = env_path.read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _env(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return value.strip()


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = _env(name, str(default))
    try:
        value = int(raw)
    except ValueError:
        raise ConfigError("环境变量 %s 必须是整数，当前值：%r" % (name, raw)) from None
    if value < minimum:
        raise ConfigError("环境变量 %s 必须 >= %d，当前值：%d" % (name, minimum, value))
    return value


def mask_secret(secret: str) -> str:
    """给界面展示用：只露头尾，绝不打印完整密钥。"""
    if not secret:
        return "(未设置)"
    if len(secret) <= 8:
        return "*" * len(secret)
    return secret[:3] + "*" * 6 + secret[-4:]


@dataclass(frozen=True)
class Settings:
    deepseek_api_key: str
    deepseek_base_url: str
    deepseek_model: str
    embedding_backend: str
    embedding_api_key: str
    embedding_base_url: str
    embedding_model: str
    embedding_dim: int
    chunk_size: int
    chunk_overlap: int
    top_k: int
    collection_name: str
    data_dir: Path
    upload_dir: Path
    chroma_dir: Path
    proxy: ProxyConfig = field(default_factory=read_proxy_config)

    def __post_init__(self) -> None:
        if self.chunk_overlap >= self.chunk_size:
            raise ConfigError(
                "CHUNK_OVERLAP(%d) 必须小于 CHUNK_SIZE(%d)，否则切块无法前进"
                % (self.chunk_overlap, self.chunk_size)
            )

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()

        backend = _env("EMBEDDING_BACKEND").lower()
        embedding_api_key = _env("EMBEDDING_API_KEY")
        if backend not in EMBEDDING_BACKENDS:
            # 没显式指定时按"有没有配 key"自动选，保证零配置也能先跑起来
            backend = "api" if embedding_api_key else "chroma"

        data_dir = Path(_env("DATA_DIR", str(PROJECT_ROOT / "data"))).expanduser()

        return cls(
            deepseek_api_key=_env("DEEPSEEK_API_KEY"),
            deepseek_base_url=_env("DEEPSEEK_BASE_URL", DEFAULTS["deepseek_base_url"]).rstrip("/"),
            deepseek_model=_env("DEEPSEEK_MODEL", DEFAULTS["deepseek_model"]),
            embedding_backend=backend,
            embedding_api_key=embedding_api_key,
            embedding_base_url=_env("EMBEDDING_BASE_URL", DEFAULTS["embedding_base_url"]).rstrip("/"),
            embedding_model=_env("EMBEDDING_MODEL", DEFAULTS["embedding_model"]),
            embedding_dim=_env_int("EMBEDDING_DIM", int(DEFAULTS["embedding_dim"])),
            chunk_size=_env_int("CHUNK_SIZE", int(DEFAULTS["chunk_size"]), minimum=100),
            chunk_overlap=_env_int("CHUNK_OVERLAP", int(DEFAULTS["chunk_overlap"]), minimum=0),
            top_k=_env_int("TOP_K", int(DEFAULTS["top_k"])),
            collection_name=_env("COLLECTION_NAME", DEFAULTS["collection_name"]),
            data_dir=data_dir,
            upload_dir=data_dir / "uploads",
            chroma_dir=data_dir / "chroma",
        )

    # -- 校验与展示 --------------------------------------------------
    @property
    def has_llm_key(self) -> bool:
        return bool(self.deepseek_api_key)

    def require_llm_key(self) -> None:
        if not self.has_llm_key:
            raise ConfigError(
                "缺少 DEEPSEEK_API_KEY：请在项目目录的 .env 或系统环境变量里配置后再提问。"
            )

    def ensure_dirs(self) -> None:
        for target in (self.data_dir, self.upload_dir, self.chroma_dir):
            target.mkdir(parents=True, exist_ok=True)

    def describe(self) -> dict:
        """给 Streamlit 侧边栏展示的配置摘要（密钥一律打码）。"""
        return {
            "大模型": "%s @ %s" % (self.deepseek_model, self.deepseek_base_url),
            "大模型 Key": mask_secret(self.deepseek_api_key),
            "Embedding 后端": self.embedding_backend,
            "Embedding 模型": self.embedding_model if self.embedding_backend == "api" else "(由后端决定)",
            "Embedding Key": mask_secret(self.embedding_api_key),
            "切块 大小/重叠": "%d / %d 字符" % (self.chunk_size, self.chunk_overlap),
            "检索条数 top_k": self.top_k,
            "向量库目录": str(self.chroma_dir),
            "网络代理": " / ".join(self.proxy.describe()),
        }
