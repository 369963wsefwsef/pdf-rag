"""pdfrag：本地 PDF 知识库问答（RAG）。

模块划分（按数据流顺序）：

    pdf_loader  ->  chunker  ->  embedder  ->  vector_store
                                                  |
                                   retriever  <---+
                                       |
                                     llm  ->  pipeline  ->  app(Streamlit)

每一步都能单独测试，不依赖后面的步骤。
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
