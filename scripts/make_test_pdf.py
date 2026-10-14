"""生成一个多页测试 PDF（纯标准库，不需要 reportlab）。

用法：
    python scripts/make_test_pdf.py                    # 默认写到 data/sample.pdf
    python scripts/make_test_pdf.py demo.pdf --pages 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from _pdf_builder import build_pdf  # noqa: E402

DEMO_PAGES = [
    [
        "Retrieval Augmented Generation (RAG) Demo",
        "Page 1 of the sample document.",
        "It exists so that tests and manual runs have a real PDF to work with.",
    ],
    [
        "How RAG works",
        "RAG retrieves the most relevant chunks from a knowledge base,",
        "then hands them to a large language model together with the question.",
        "This keeps answers grounded in your own documents.",
    ],
    [
        "Why chunk overlap matters",
        "A sentence that happens to land on a chunk boundary",
        "would be incomplete in both chunks without an overlap region.",
    ],
]


def main() -> int:
    parser = argparse.ArgumentParser(description="生成测试用 PDF")
    parser.add_argument("output", nargs="?", default="data/sample.pdf")
    parser.add_argument("--pages", type=int, default=0, help="只保留前 N 页（0 = 全部）")
    args = parser.parse_args()

    pages = DEMO_PAGES[: args.pages] if args.pages else DEMO_PAGES
    path = build_pdf(Path(args.output), pages)
    print("已生成：%s（%d 页，%d 字节）" % (path, len(pages), path.stat().st_size))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
