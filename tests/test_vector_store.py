"""第三步（下）的测试：Chroma 向量库封装（真实 Chroma，落在临时目录）。"""

from __future__ import annotations

import pytest
from dataclasses import replace

from _fakes import FakeEmbedder, make_chunk, make_settings
from pdfrag.vector_store import VectorStore, VectorStoreError


def build_store(tmp_path, *, dim=64, name="pytest_collection", embedder=None):
    settings = make_settings(tmp_path, collection_name=name)
    active = embedder or FakeEmbedder(dim=dim)
    return VectorStore(settings, embedder=active), active


def test_upsert_count_and_search_ranking(tmp_path):
    store, embedder = build_store(tmp_path)
    chunks = [
        make_chunk(0, "苹果 富含 维生素 和 膳食纤维"),
        make_chunk(1, "Python 是一门 编程语言", page=2),
        make_chunk(2, "今天 天气 很好", page=3),
    ]
    assert store.upsert_document(chunks) == 3
    assert store.count() == 3

    query = embedder.embed_query("苹果 维生素")
    hits = store.search(query, top_k=2)

    assert len(hits) == 2
    assert "苹果" in hits[0].text, "最相关的应该是苹果那一块"
    assert hits[0].score >= hits[1].score
    assert hits[0].page == 1
    assert hits[0].source == "sample.pdf"
    assert hits[0].citation == "sample.pdf p.1"


def test_search_can_filter_by_doc_id(tmp_path):
    store, embedder = build_store(tmp_path)
    store.upsert_document([make_chunk(0, "文档一的 内容 关于 苹果", doc_id="doc_one", source="one.pdf")])
    store.upsert_document([make_chunk(0, "文档二的 内容 关于 苹果", doc_id="doc_two", source="two.pdf")])

    vector = embedder.embed_query("苹果")
    assert len(store.search(vector, top_k=5)) == 2
    filtered = store.search(vector, top_k=5, doc_ids=["doc_two"])
    assert [hit.source for hit in filtered] == ["two.pdf"]
    assert store.search(vector, top_k=5, doc_ids=["doc_one", "doc_two"]) != []


def test_reingest_same_document_is_idempotent(tmp_path):
    store, _ = build_store(tmp_path)
    chunks = [make_chunk(0, "第一段 内容"), make_chunk(1, "第二段 内容")]

    store.upsert_document(chunks)
    assert store.count() == 2
    store.upsert_document(chunks)          # 同一份文件（同 doc_id）重复入库
    assert store.count() == 2, "重复入库不应产生重复条目"


def test_reingest_replaces_changed_content(tmp_path):
    store, _ = build_store(tmp_path)
    store.upsert_document([make_chunk(0, "旧 内容"), make_chunk(1, "旧 内容 之二")])
    store.upsert_document([make_chunk(0, "新 内容")])
    assert store.count() == 1, "内容变了但 doc_id 相同也应替换掉旧块"


def test_delete_and_list_documents(tmp_path):
    store, _ = build_store(tmp_path)
    store.upsert_document([make_chunk(0, "甲 内容", doc_id="doc_aaa", source="a.pdf")])
    store.upsert_document([make_chunk(0, "乙 内容", doc_id="doc_bbb", source="b.pdf", page=1),
                           make_chunk(1, "乙 内容 续", doc_id="doc_bbb", source="b.pdf", page=2)])

    documents = store.list_documents()
    assert [item["source"] for item in documents] == ["a.pdf", "b.pdf"]
    assert documents[1]["chunks"] == 2 and documents[1]["pages"] == 2

    assert store.delete_document("doc_aaa") == 1
    assert store.count() == 2
    assert [item["doc_id"] for item in store.list_documents()] == ["doc_bbb"]


def test_reset_clears_everything(tmp_path):
    store, _ = build_store(tmp_path)
    store.upsert_document([make_chunk(0, "内容")])
    store.reset()
    assert store.count() == 0
    assert store.list_documents() == []


def test_empty_store_search_returns_nothing(tmp_path):
    store, embedder = build_store(tmp_path)
    assert store.count() == 0
    assert store.search(embedder.embed_query("任意问题")) == []


def test_collection_name_must_be_long_enough(tmp_path):
    store, _ = build_store(tmp_path, name="ab")     # Chroma 要求 3~512 字符
    with pytest.raises(VectorStoreError, match="3~512"):
        store.count()


def test_embedding_model_change_is_detected(tmp_path):
    """换 embedding 模型（维度变了）必须报错，而不是静默返回垃圾检索结果。"""
    first, _ = build_store(tmp_path, dim=64)
    first.upsert_document([make_chunk(0, "内容")])

    second, _ = build_store(tmp_path, dim=128)      # 同一个 chroma 目录、不同维度
    with pytest.raises(VectorStoreError, match="不兼容"):
        second.count()


def test_upsert_rejects_mixed_documents(tmp_path):
    store, _ = build_store(tmp_path)
    mixed = [make_chunk(0, "甲", doc_id="doc_aaa"), make_chunk(1, "乙", doc_id="doc_bbb")]
    with pytest.raises(VectorStoreError, match="同一个文档"):
        store.upsert_document(mixed)


def test_search_top_k_larger_than_store(tmp_path):
    store, embedder = build_store(tmp_path)
    store.upsert_document([make_chunk(0, "只有 一块 内容")])
    hits = store.search(embedder.embed_query("内容"), top_k=99)
    assert len(hits) == 1, "top_k 超过库容量时应返回现有全部，而不是报错"


def test_unit_round_trips_through_metadata(tmp_path):
    """unit 要能写进 Chroma 元数据再读回来，否则 DOCX 的引用会退化成"页"。"""
    store, embedder = build_store(tmp_path)
    chunks = [make_chunk(0, "Word 段落", doc_id="doc_word", source="w.docx")]
    chunks = [replace(chunk, unit="section") for chunk in chunks]
    store.upsert_document(chunks)

    hits = store.search(embedder.embed_query("Word 段落"), top_k=1)
    assert hits[0].unit == "section"
    assert hits[0].citation == "w.docx 第1段"
    assert hits[0].to_dict()["unit"] == "section"
