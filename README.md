# pdf-rag · 本地 PDF 知识库问答（RAG）

把本地 PDF 变成可以提问的知识库：上传 PDF → 切块 → 向量化 → 存进 Chroma →
提问时先检索最相关的几块 → 连同问题一起交给 DeepSeek 生成答案，并给出引用来源（文件名 + 页码 + 原文）。

## 一、项目结构

    pdf-rag/
    ├── app.py                        # Streamlit 网页界面（上传 / 提问 / 引用展示 / 文档管理）
    ├── run_app.ps1                   # 一键启动脚本
    ├── requirements.txt
    ├── pyproject.toml                # pytest 配置（pythonpath / 不写 cache）
    ├── .env                          # 你的实际配置（已被 .gitignore，不要提交）
    ├── .env.example                  # 配置样例
    ├── README.md                     # 本文件
    ├── src/pdfrag/
    │   ├── config.py                 # 配置集中管理（环境变量只在这里读一次）
    │   ├── pdf_loader.py             # 第 1 步：PDF → 带页码的文本
    │   ├── docx_loader.py            # 第 1 步：Word(.docx) → 带段落编号的文本（标准库实现，零新依赖）
    │   ├── chunker.py                # 第 2 步：文本 → 带重叠的 chunk
    │   ├── embedder.py               # 第 3 步：chunk → 向量（api / local / chroma）
    │   ├── vector_store.py           # 第 3 步：Chroma 封装（幂等入库、模型校验、元数据过滤）
    │   ├── retriever.py              # 第 4 步：检索 + 编号拼上下文
    │   ├── llm.py                    # 第 4 步：DeepSeek 流式生成
    │   └── pipeline.py               # 编排 ingest / answer 两条链路
    ├── scripts/make_test_pdf.py      # 生成测试用多页 PDF
    ├── tests/                        # 全部离线可跑
    └── data/                         # 运行时生成（已 gitignore）
        ├── uploads/                  # 上传的 PDF
        └── chroma/                   # 向量库持久化目录

数据流：

    ingest:  PDF → load_pdf → chunk_pages → embedder → Chroma
    answer:  问题 → embedder → Chroma.search → 拼上下文 → DeepSeek → 答案 + 引用

UI 只依赖 `pipeline`，换 UI（CLI / FastAPI）不影响核心逻辑；每一步都能单独测试。

## 二、技术选型（含一个关键前提）

### ⚠️ DeepSeek 没有 Embedding 接口

DeepSeek 官方 API 只有 chat 类接口（`chat/completions`、`/models`、`/files`），
**没有 `/embeddings`**。所以「把文本变向量」必须由别的模型承担，这是本项目唯一需要额外配置的地方。
代码里做成了可插拔后端，改 `.env` 一行即可切换：

| 后端 | 中文效果 | 需要额外 Key | 说明 |
|---|---|---|---|
| `api`（当前配置） | 好 | 要 | 任意 OpenAI 兼容 `/embeddings`：硅基流动 BGE-M3、DashScope、OpenAI、本地 Ollama |
| `local` | 好 | 不要 | sentence-transformers 本地模型，需装 torch（约 2GB） |
| `chroma` | 差 | 不要 | Chroma 自带 ONNX MiniLM（英文模型），零配置跑通流程用 |

| 环节 | 选型 | 备注 |
|---|---|---|
| PDF 解析 | pypdf | 纯 Python。复杂双栏/表格可换 pdfplumber；扫描件需另配 OCR |
| Word 解析 | 标准库 zipfile + xml.etree | .docx 本质是 ZIP，正文在 word/document.xml；不用装 python-docx |
| 切块 | 自研递归切分器 | 段落 > 换行 > 句末标点 > 逗号空格 > 硬切，带重叠 |
| 向量库 | Chroma PersistentClient | 本地文件持久化、不用起服务、支持元数据过滤 |
| 大模型 | DeepSeek `deepseek-chat` | 标准库 urllib + SSE 实现流式，不依赖 openai SDK |
| 界面 | Streamlit | 上传 / 提问 / 流式答案 / 引用来源 |
| 测试 | pytest | 假 Embedder + 假大模型 + 现场拼的假 PDF / 假 docx，不联网、不花钱 |

### Word(.docx) 支持

支持 `.docx`（Word 2007+；.docm 同结构），**不需要额外依赖**：直接用标准库
`zipfile` 打开 ZIP、用 `xml.etree` 解析 `word/document.xml`。
会保留段落结构，并按下面的规则切块：

* **一个段落块 = 一个标题 + 它下面的正文段落**（直到下一个标题）。标题因此成为该块的一部分，
  检索命中时自带「这段属于哪一节」的上下文；
* 表格单独成块，行内用 ` | ` 分隔单元格；
* 段落内的换行原样保留；空段落跳过；不换行空格 / 全角空格统一成普通空格；
* 标题识别同时看 `pStyle`（Heading1 / 标题1）和 `outlineLvl`（Word 的大纲级别），
  中文模板也能认出来；
* 没有标题的长文档会按长度（默认 1200 字）继续切块，避免整篇挤成「第 1 段」。

**引用单位是「段」而不是「页」**：Word 是流式文档，同一份文件换台打印机页数都会变，
所以定位单位用段落块，引用显示为 `手册.docx 第12段`。页眉页脚、脚注、批注、
修订痕迹、文本框不会被抽取 —— 它们对「检索正文回答问题」没有影响。

不支持老式 `.doc`（97-2003 二进制格式，和 .docx 完全是两种东西）：上传会给出
明确提示，用 Word 打开后「另存为 .docx」即可。

## 三、环境准备

    cd C:\Users\lsh\Documents\ChatGPT\pdf-rag

    # 虚拟环境（已建好；如果删了或移动过目录，重建）
    python -m venv .venv

    # 安装依赖
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt

## 四、配置 .env

项目根目录的 `.env`（已被 gitignore）当前内容：

    EMBEDDING_BACKEND=api
    EMBEDDING_API_KEY=
    EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1
    EMBEDDING_MODEL=BAAI/bge-m3

**你还差两个 Key，填好就能完整使用：**

1. `EMBEDDING_API_KEY` —— 向量化用。去 https://cloud.siliconflow.cn 注册拿 Key（有免费额度），填进去：

       EMBEDDING_API_KEY=sk-你的硅基流动key

   不想申请就先留空，或把 `EMBEDDING_BACKEND` 改成 `chroma` 零配置跑通流程（中文检索效果会打折）。

2. `DEEPSEEK_API_KEY` —— 生成答案用。加到同一个文件里：

       DEEPSEEK_API_KEY=sk-你的DeepSeek key

   如果这个变量已经存在于系统环境变量里，不写进 `.env` 也能生效：程序只在环境变量缺失时才读 `.env`。

改完 `.env` 需要**重启应用**才生效。

## 五、运行

### 启动网页界面

    # 方式 A：一键脚本（自动使用项目内的 .venv）
    .\run_app.ps1

    # 方式 B：手动
    .\.venv\Scripts\python.exe -m streamlit run app.py

浏览器会自动打开 http://localhost:8501。停止服务按 `Ctrl+C`。
换端口：`.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8502`。

### 界面操作流程

1. **左侧栏**先看「运行状态」：Embedding / 大模型是否就绪，缺什么会直接告诉你；下面可改检索条数 top_k、查看文档清单、删除单篇文档或清空知识库。
2. **左上「① 上传 PDF 入库」**：选一个或多个 PDF → 点「开始入库」。会实时显示读取 PDF → 切块 → 向量化 → 入库完成，以及页数 / 块数 / 字符数。同一份文件重复上传会自动替换旧向量，不会重复。
3. **右上「② 提问」**：输入问题 → 点「提问」。答案逐字流式显示；下方「📌 引用来源」列出每一块的 `[编号] 文件名 · 第 N 页 · 相似度`，点开即可看到原文。

### 跑测试

    .\.venv\Scripts\python.exe -m pytest                          # 全部用例
    .\.venv\Scripts\python.exe -m pytest -v                       # 显示每个用例名
    .\.venv\Scripts\python.exe -m pytest tests/test_pipeline.py   # 只跑某一步

测试**完全离线**：用 `FakeEmbedder`（确定性词袋向量）和 `FakeLLM`（记录 prompt 的假模型），
不联网、不下载模型、不花 API 费用，全量约 2 秒。

### 生成测试用 PDF

    .\.venv\Scripts\python.exe scripts\make_test_pdf.py data\sample.pdf

## 六、几个关键设计决策

1. **doc_id 用文件内容哈希**（sha256 前 16 位）：同一份 PDF 改名后仍认得出来，重复上传天然幂等；内容改一个字就是新文档。
2. **幂等入库**：写入前先按 `doc_id` 删除旧向量，所以「重新上传修改后的同一份文件」等于覆盖更新，不会新旧混杂。
3. **向量库不绑 Embedding 函数**：建 collection 时传 `embedding_function=None`，向量由我们自己的 Embedder 算好再写入。好处是不会触发 Chroma 偷偷下载自带模型，换模型只需改 `.env`。
4. **模型 / 维度校验**：建库时把 embedding 模型名和维度写进 collection 元数据。之后换模型（bge-m3 是 1024 维、MiniLM 是 384 维）会**明确报错并提示重建**，而不是静默返回一堆错误的相似度。
5. **切块按页进行**：保证引用页码 100% 准确，代价是跨页的句子可能被切成两块。
6. **上下文有字符预算**（默认 6000 字）：避免 top_k 个超长块把 token 打爆；预算不够时保留前面的（更相关的）块，并去重相同片段。
7. **prompt 约束**：system prompt 明确要求「只依据参考资料回答、用 [编号] 标注来源、资料不足就直说」，把幻觉压到最低。
8. **temperature=0.2**：知识库问答要稳，比闲聊低得多。
9. **错误是事件不是异常**：`answer_stream()` 把错误也做成事件，界面不会因为一次检索失败就崩溃。

## 七、常见问题

**打开界面提示「Embedding 未就绪 / EMBEDDING_API_KEY 是空的」**
按第四节第 1 条填 Key，然后重启应用。

**提示「向量库是用 xxx 建的，当前配置是 yyy，两者不兼容」**
你换了 embedding 模型。删除 `data/chroma` 目录（或点界面上的「清空整个知识库」）后重新上传文档即可。

**报错「collection 名字必须 3~512 个字符」**
Chroma 的硬性限制，改 `.env` 里的 `COLLECTION_NAME`（默认值没问题）。

**扫描件 PDF 报「可能是扫描件，需要先做 OCR」**
pypdf 只能抽文本层。要支持扫描件需另加 `pypdfium2`（转图）+ OCR 引擎，属于可选扩展。

**中文检索结果不准**
确认用的是 `BAAI/bge-m3` 这类中文模型；`chroma` 兜底后端用的是英文模型，中文效果差是正常的。

**首次向量化很慢**
`api` 后端受网络与接口速率影响；`chroma` 后端首次会下载约 80MB 模型。

**费用**
一次问答 ≈ 1 次 embedding（问题很短）+ 1 次 chat（输入 = 检索到的几块文本）。bge-m3 在硅基流动有免费额度，DeepSeek 按 token 计费。

## 八、Word(.docx) 常见问题

**上传 .docx 报「不是有效的 .docx」**
.docx 本质是 ZIP 包：文件可能已损坏，或者其实是把 .doc / .pdf 改名成了 .docx。
可以先改回真实后缀，或用 Word 重新「另存为」.docx。

**上传 .doc 报「老式的 .doc 二进制格式」**
这是 97-2003 的旧格式，和 .docx 完全不同。用 Word 打开 →「另存为」→ 选 .docx 再上传。

**为什么引用显示「第 12 段」而不是页码？**
Word 没有可靠的分页信息（分页取决于字体、纸张和打印机），所以定位单位改用段落块。
一个段落块 = 一个标题 + 它下面的正文；文档很长时还会按 1200 字上限继续切分。

**标题没被识别成标题**
标题要用 Word 内置的「标题 1 / 2 / 3」样式，或确保段落带大纲级别。手动调成
「加粗大字」的段落，在文件里只是普通段落，程序无法区分。

**解析 docx 需要额外装什么吗？**
不需要。解析用的是 Python 标准库，装好 requirements.txt 里的依赖即可。

## 九、网络代理与 WinError 10061

### 先说结论：urllib 本来就认 HTTP_PROXY / HTTPS_PROXY

`urllib.request.urlopen` 默认就会读 `HTTP_PROXY / HTTPS_PROXY / NO_PROXY`（CPython 里 Windows 的解析顺序是「环境变量优先，其次才是注册表里的系统代理」）。
所以报 `[WinError 10061] 由于目标计算机积极拒绝，无法连接` 时，**几乎不是「没用代理」，而是「代理地址/端口上没有服务在监听」** ——
代理客户端没启动，或者端口填错了（Clash 默认 7890，Clash Verge 混合端口常见 7897，V2Ray 常见 10809）。

实测对照（同一台机器、同一个 API）：

| 配置 | 结果 |
| --- | --- |
| 无代理变量（直连） | 正常返回 HTTP 401（能连上） |
| `HTTPS_PROXY=http://127.0.0.1:7897`（代理在跑） | 正常返回 HTTP 401 |
| `HTTPS_PROXY=http://127.0.0.1:7890`（端口没人监听） | **WinError 10061** |

### 本项目做了什么

urllib 虽然认环境变量，但有两个实践中的坑，所以这里改成了**显式**实现：

1. **全局 opener 缓存**：`urlopen` 用的是模块级全局 opener，只在**第一次请求时**构建一次并缓存（`urllib.request._opener`）。
   如果在这之前已经有代码发过 HTTP 请求，之后再设置代理环境变量就**完全不生效** —— 「先 import、后读 .env」的程序很容易踩中。
   现在每个客户端（`ApiEmbedder / DeepSeekClient`）在构造时就按当前配置建一个**属于自己的 opener**，与全局缓存无关。
2. **看不见用了哪个代理**：现在侧边栏和报错信息都会明确写出最终生效的代理地址，出现 10061 时直接告诉你「是哪个地址在拒绝连接」以及两条出路。

另外两点改进：

* **本机地址永远直连**（`localhost / 127.0.0.1 / ::1`）—— 配了代理后用本地 Ollama / LM Studio 时不会被送去代理；
* **连接被直接拒绝时不重试**（原来会白等 3 次退避），超时仍然照常重试。

### 怎么配

    HTTP_PROXY=http://127.0.0.1:7897
    HTTPS_PROXY=http://127.0.0.1:7897
    NO_PROXY=localhost,127.0.0.1

    PDFRAG_PROXY=off              # 想强制直连时用这个（优先级最高，忽略上面所有设置）
    PDFRAG_PROXY=127.0.0.1:7897   # 一次性给 http/https 都设上（省得写两行）

改完 `.env` 要**重启应用**。

### 排查三步

    # 1) 看 Python 最终会用什么代理（含系统代理）
    .\.venv\Scripts\python.exe -c "import urllib.request as u; print(u.getproxies())"

    # 2) 看那个端口到底有没有人监听（把 7897 换成你的端口）
    Test-NetConnection 127.0.0.1 -Port 7897

    # 3) 不确定就绕过代理直连试试
    #    在 .env 里加：PDFRAG_PROXY=off

**注意**：环境变量优先于 Windows 设置里的系统代理。所以如果你的终端里导出过 `HTTPS_PROXY` 指向一个已经关掉的代理客户端，
即使系统设置里的代理是关的，程序也会去连那个死端口。
