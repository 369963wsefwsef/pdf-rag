"""第三步（下半）：Chroma 向量库封装。

三个关键决策
------------
1) embedding_function=None：向量由我们自己的 Embedder 算好再传进去。
   这样不会触发 Chroma 偷偷下载它自带的 ONNX 模型，而且"用哪个 embedding 模型"
   完全由 .env 决定，不会出现两套向量混在一个库里。
2) 幂等入库：doc_id 来自 PDF 内容哈希，重复上传时先按 doc_id 删除旧向量再写入，
   不会产生重复条目（内容没变则结果等价）。
3) 模型/维度校验：建库时把 embedding 模型名和维度写进 collection 元数据，
   之后如果换了模型，检索前就会明确报错，而不是静默返回一堆错的相似度。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .config import Settings

# Chroma 1.x 仍支持这种写法（已实测无弃用警告）；cosine 对归一化向量最直观
_BASE_METADATA: Dict[str, Any] = {"hnsw:space": "cosine"}
_MIN_NAME_LENGTH = 3
_MAX_NAME_LENGTH = 512


class VectorStoreError(RuntimeError):
    """向量库相关错误（名字非法、模型不匹配、Chroma 未安装等）。"""


@dataclass(frozen=True)
class Hit:
    """一次检索命中的片段。"""

    text: str
    score: float           # 相似度，越大越相关（= 1 - cosine 距离）
    distance: float
    doc_id: str
    source: str
    page: int              # 定位单元序号：PDF 是页码，DOCX 是段落块号
    chunk_index: int
    chunk_id: str
    unit: str = "page"     # 定位单位：page / section

    @property
    def citation(self) -> str:
        if self.unit == "page":
            return "%s p.%d" % (self.source, self.page)
        return "%s 第%d段" % (self.source, self.page)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "score": round(self.score, 4),
            "distance": round(self.distance, 4),
            "doc_id": self.doc_id,
            "source": self.source,
            "unit": self.unit,
            "page": self.page,
            "chunk_index": self.chunk_index,
            "chunk_id": self.chunk_id,
            "citation": self.citation,
        }


class VectorStore:
    """Chroma 持久化向量库的薄封装。"""

    def __init__(
        self,
        settings: Settings,
        *,
        embedder: Optional[Any] = None,
        collection_name: Optional[str] = None,
    ) -> None:
        self.settings = settings
        self.collection_name = collection_name or settings.collection_name
        self.embedder = embedder
        self._client: Optional[Any] = None
        self._collection: Optional[Any] = None
        self._signature_checked = False

    # -- 连接 --------------------------------------------------------
    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                import chromadb
            except Exception as exc:  # pragma: no cover - 依赖缺失
                raise VectorStoreError("未安装 chromadb：pip install chromadb") from exc
            Path(self.settings.chroma_dir).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.settings.chroma_dir))
        return self._client

    def _signature(self) -> Dict[str, Any]:
        if self.embedder is None:
            return {}
        return {
            "embedding_model": str(getattr(self.embedder, "name", "unknown")),
            "embedding_dim": int(getattr(self.embedder, "dim", 0) or 0),
        }

    @property
    def collection(self) -> Any:
        if self._collection is None:
            name = (self.collection_name or "").strip()
            if not (_MIN_NAME_LENGTH <= len(name) <= _MAX_NAME_LENGTH):
                raise VectorStoreError(
                    "collection 名字必须 %d~%d 个字符（当前 %r 长度 %d）"
                    % (_MIN_NAME_LENGTH, _MAX_NAME_LENGTH, name, len(name))
                )
            metadata = dict(_BASE_METADATA)
            metadata.update(self._signature())
            self._collection = self.client.get_or_create_collection(
                name=name,
                embedding_function=None,
                metadata=metadata,
                configuration={"hnsw": {"space": "cosine"}},
            )
        if not self._signature_checked:
            self._verify_signature()
            self._signature_checked = True
        return self._collection

    def _verify_signature(self) -> None:
        """建库时用的 embedding 模型和现在要用的不一致就报错。"""
        stored = dict(self._collection.metadata or {})
        current = self._signature()
        if not current or not stored.get("embedding_model"):
            return
        same_model = str(stored.get("embedding_model")) == current["embedding_model"]
        same_dim = int(stored.get("embedding_dim") or 0) == int(current["embedding_dim"] or 0)
        if same_model and same_dim:
            return
        # 注意：这里必须直接读 Chroma 的原始计数，不能走 self.count() ——
        # self.count() 会再访问 collection 属性，从而再次进入本方法，形成无限递归。
        if int(self._collection.count()) == 0:
            self._stamp_signature(current)   # 空库无所谓，顺手更新签名
            return
        raise VectorStoreError(
            "向量库是用 %s（%d 维）建的，当前配置是 %s（%d 维），两者不兼容。\n"
            "换 embedding 模型必须重建向量库：删掉目录 %s 后重新上传文档。"
            % (
                stored.get("embedding_model"), int(stored.get("embedding_dim") or 0),
                current["embedding_model"], int(current["embedding_dim"] or 0),
                self.settings.chroma_dir,
            )
        )

    def _stamp_signature(self, signature: Dict[str, Any]) -> None:
        """把签名补写进元数据。注意不能带上 hnsw:*，否则 Chroma 会拒绝修改。"""
        try:
            stored = {k: v for k, v in dict(self._collection.metadata or {}).items()
                      if not str(k).startswith("hnsw:")}
            stored.update(signature)
            self._collection.modify(metadata=stored)
        except Exception:
            pass   # 打标只是辅助手段，失败不影响主流程

    # -- 读写 --------------------------------------------------------
    def count(self) -> int:
        try:
            return int(self.collection.count())
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError("读取向量库条数失败：%s" % exc) from exc

    def upsert_document(self, chunks: Sequence[Any], *, embedder: Optional[Any] = None) -> int:
        """把一个文档的所有 chunk 写入向量库（先删旧、再写新，保证幂等）。"""
        items = list(chunks)
        if not items:
            return 0
        active = embedder or self.embedder
        if active is None:
            raise VectorStoreError("没有可用的 Embedder，无法写入向量库")

        doc_id = items[0].doc_id
        if any(chunk.doc_id != doc_id for chunk in items):
            raise VectorStoreError("一次 upsert 只能处理同一个文档的 chunk")

        self.delete_document(doc_id)
        vectors = active.embed([chunk.text for chunk in items])
        if len(vectors) != len(items):
            raise VectorStoreError(
                "向量数量与 chunk 数量不一致：%d vs %d" % (len(vectors), len(items))
            )
        self.collection.add(
            ids=[chunk.id for chunk in items],
            documents=[chunk.text for chunk in items],
            embeddings=vectors,
            metadatas=[chunk.to_metadata() for chunk in items],
        )
        return len(items)

    def search(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int = 5,
        doc_ids: Optional[Sequence[str]] = None,
    ) -> List[Hit]:
        total = self.count()
        if total == 0:
            return []
        n_results = max(1, min(int(top_k), total))

        where: Optional[Dict[str, Any]] = None
        if doc_ids:
            unique = list(dict.fromkeys(str(item) for item in doc_ids))
            where = {"doc_id": unique[0]} if len(unique) == 1 else {"doc_id": {"$in": unique}}

        result = self.collection.query(
            query_embeddings=[list(query_vector)],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        ids = (result.get("ids") or [[]])[0]

        hits: List[Hit] = []
        for index, text in enumerate(documents):
            metadata = metadatas[index] if index < len(metadatas) else {}
            distance = float(distances[index]) if index < len(distances) else 1.0
            hits.append(
                Hit(
                    text=str(text or ""),
                    score=1.0 - distance,
                    distance=distance,
                    doc_id=str(metadata.get("doc_id", "")),
                    source=str(metadata.get("source", "")),
                    unit=str(metadata.get("unit", "page") or "page"),
                    page=int(metadata.get("page", 0) or 0),
                    chunk_index=int(metadata.get("chunk_index", 0) or 0),
                    chunk_id=str(ids[index]) if index < len(ids) else "",
                )
            )
        return hits

    def delete_document(self, doc_id: str) -> int:
        if not doc_id:
            return 0
        before = self.count()
        self.collection.delete(where={"doc_id": str(doc_id)})
        return max(0, before - self.count())

    def list_documents(self) -> List[Dict[str, Any]]:
        """按 doc_id 聚合出文档清单（文件名/页数/块数）。"""
        if self.count() == 0:
            return []
        try:
            data = self.collection.get(include=["metadatas"])
        except Exception as exc:
            raise VectorStoreError("读取文档清单失败：%s" % exc) from exc

        aggregated: Dict[str, Dict[str, Any]] = {}
        for metadata in data.get("metadatas") or []:
            doc_id = str(metadata.get("doc_id", ""))
            if not doc_id:
                continue
            entry = aggregated.setdefault(
                doc_id,
                {"doc_id": doc_id, "source": str(metadata.get("source", "")),
                 "chunks": 0, "pages": set()},
            )
            entry["chunks"] += 1
            page = metadata.get("page")
            if isinstance(page, (int, float)):
                entry["pages"].add(int(page))

        documents = [
            {"doc_id": entry["doc_id"], "source": entry["source"],
             "chunks": entry["chunks"], "pages": len(entry["pages"])}
            for entry in aggregated.values()
        ]
        return sorted(documents, key=lambda item: item["source"])

    def reset(self) -> None:
        """清空整个知识库（删掉 collection 再让它自动重建）。"""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._collection = None
        self._signature_checked = False

    def describe(self) -> Dict[str, Any]:
        return {
            "collection": self.collection_name,
            "path": str(self.settings.chroma_dir),
            "chunks": self.count(),
            "documents": len(self.list_documents()),
        }
