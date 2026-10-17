# 纯 Python + DeepSeek 的简单 RAG

适合约 15 份短文档的小型知识库和课程实验。没有 LangChain、LlamaIndex 或向量数据库：读取、分块、检索和提示词等公共能力都在 `rag.py`，便于理解和复用。`rag.py` 仅作为公共函数库；`BGE/run.py` 与 `BM25/run.py` 各自提供入库、搜索、问答、聊天和评估命令。

流程：`文档 → 共同分块 → BGE 向量检索或 BM25 词项检索 → Top-K → DeepSeek → 带引用的回答`。`search`、`evaluate` 在检索后结束，`ask`、`chat` 才生成答案。

## 安装（Windows PowerShell）

建议使用 Python 3.11 或 3.12。普通 Python 安装后执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-experiment.txt
```

不需要激活虚拟环境，也不受 PowerShell 脚本执行策略影响。如果 `python` 指向 Windows 商店占位程序，请使用实际 Python 路径执行第一条命令。

当前工作区已安装 `.venv`。`requirements-lock.txt` 保留原有基础环境的版本记录；完整实验依赖见 `requirements-experiment.txt`，每次实验的实际版本记录在结果元数据中。

本次验证（2026-10-15）：7 项本地测试通过；两份示例文档已生成 `storage/demo.json`（512 维向量）；真实调用 DeepSeek 完成带来源的问答，并在资料没有发布日期时明确拒答。现已加入 15 份公开来源的中文整理语料，分为天文与太阳系、海洋与自然环境、计算机与 Web 基础三个领域，每领域 5 份；共 9 篇长文、6 篇短文。正式索引 `storage/index.json` 已生成，共 33 个块。来源、原文快照、字数和许可详见 [data/README.md](data/README.md)。

本项目根目录 `.env` 已配置本次提供的 DeepSeek 凭据。迁移到其他机器时，复制 `.env.example` 为 `.env` 并填入自己的 key。`.env`、虚拟环境和本地索引已加入 `.gitignore`；本次公开语料的三个目录允许版本控制，`data/docs` 中其他目录和文件仍默认忽略。外部环境变量优先于 `.env`，程序不会打印密钥。

```dotenv
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-flash
EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
EMBEDDING_DEVICE=cpu
```

首次运行编码器需要联网下载模型；默认缓存到项目 `.cache/huggingface`，外部 `HF_HOME` 配置优先。之后使用本地缓存。CPU 即可运行。也可把 `EMBEDDING_MODEL` 改为已下载模型的本地目录。依赖安装会安装 PyTorch，体积较大。

## GPU 编码

当前 `.env` 已设置 `EMBEDDING_DEVICE=cuda`，实际验证编码器运行于 `cuda:0`；峰值显存约 102 MiB，与原 CPU 向量的最大绝对差约 1.6e-7，示例检索首位来源一致。仍可直接使用下方原有的 `ingest`、`search`、`ask`、`chat` 命令。

本机显卡为 RTX 3060 Laptop（6GB），NVIDIA 驱动版本 546.92。`requirements-gpu.txt` 指定官方 PyTorch 2.7.1 + CUDA 11.8 安装包，适配本机现有驱动：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu.txt
```

安装后在 `.env` 中设置 `EMBEDDING_DEVICE=cuda`，即可让文档和问题的向量编码在 GPU 上运行。启动时会显示编码器实际使用的设备。改回 `EMBEDDING_DEVICE=cpu` 可使用 CPU；GPU 版 PyTorch 同样支持 CPU。

切换计算设备不改变编码器或向量维度，可以继续使用原有索引；浮点计算可能有细微差异。DeepSeek 回答生成仍通过 API 完成。GPU 安装包包含所需 CUDA 运行时，使用预编译包不需要另外安装 CUDA Toolkit。

## 先体验现有语料

BGE 已有向量索引，可以直接搜索；BM25 使用独立词项索引，先运行一次入库：

```powershell
.\.venv\Scripts\python.exe -m BGE.run search "太阳核心的能量从哪里来？"
.\.venv\Scripts\python.exe -m BM25.run ingest
.\.venv\Scripts\python.exe -m BM25.run search "太阳核心的能量从哪里来？"
```

两个方法的入口都提供以下命令：

|命令|作用|
|---|---|
|`ingest`|BGE 编码并保存向量；BM25 分词并保存词项；两边共用分块函数|
|`search "问题"`|只显示检索结果，不调用 DeepSeek|
|`ask "问题"`|用本方法检索，再调用公共生成函数|
|`chat`|循环独立问答，输入 `/exit` 结束|
|`evaluate`|只评价检索，计算 Recall/Hit 并导出失败案例|

例如，分别用两种方法问答：

```powershell
.\.venv\Scripts\python.exe -m BGE.run ask "太阳核心的能量从哪里来？" --show-context
.\.venv\Scripts\python.exe -m BM25.run ask "太阳核心的能量从哪里来？" --show-context
.\.venv\Scripts\python.exe -m BGE.run chat
.\.venv\Scripts\python.exe -m BM25.run chat
```

只有问答中的问题和检索片段会发送给 DeepSeek。查看某个命令的参数，可使用 `python -m BGE.run search --help` 或对应 BM25 命令。

## 使用自己的文档

1. 将正式文档放入 `data/docs`（支持子目录）。支持 `.txt`、`.md` 和文字版 `.pdf`；TXT/Markdown 支持 UTF-8 和 GB18030。扫描 PDF 要先做 OCR。
2. 建立索引，然后提问：

```powershell
.\.venv\Scripts\python.exe -m BGE.run ingest
.\.venv\Scripts\python.exe -m BM25.run ingest
.\.venv\Scripts\python.exe -m BGE.run ask "你的问题" --show-context
.\.venv\Scripts\python.exe -m BGE.run chat
```

`chat` 是循环独立问答，每个问题需自包含，历史记录不会作为下一次问答的上下文。输入 `/exit` 退出。向量索引默认保存在 `storage/index.json`，BM25 的 Chunk 和词项默认保存在 `storage/bm25.json`。修改文档或分块参数后应分别重新入库，修改编码器只影响 BGE。BGE 查询使用索引记录的编码器，BM25 入库与搜索不加载 BGE。

每次成功完成 `ask`、`chat` 或 `search` 查询后，会自动追加到项目的 `data/queryhistory.json`（本机为 `E:\codex_project\thesis_1\data\queryhistory.json`）。文件是 UTF-8 JSON 数组，新记录包含检索方法 `method`（BGE 或 BM25）、UTC 时间 `timestamp`、命令 `command`、问题 `question`、检索结果 `retrieval_results`（编号、来源、页码、片段编号、原文及检索分数）、完整提示词消息 `prompt` 和答案 `answer`。`search` 不生成答案，其 `prompt` 和 `answer` 为 `null`；未检索到片段的问答仍保存构造的提示词和资料不足的答案，但不调用 API。文件不存在时自动创建，后续查询保留已有记录；生成失败的查询不写入。历史文件包含问题与资料原文，适合本地留存；同时运行多个查询进程写入同一文件暂不支持。

默认每块最多 350 个字符，重叠 60 个字符，检索 Top-3。保存文档相对路径、PDF 页码和片段编号，回答用 `[1]` 等编号对应终端中的检索来源。可以调整：

```powershell
.\.venv\Scripts\python.exe -m BGE.run ingest --chunk-size 300 --overlap 50
.\.venv\Scripts\python.exe -m BGE.run ask "你的问题" --top-k 4 --min-score 0.6
```

注意：分块长度按字符计算；程序会检查编码器 token 上限，过长时提示缩小片段。余弦分数不是概率，`0.6` 仅演示用法，需要在自己的语料上校准；默认不过滤。没有片段通过阈值时，程序直接返回资料不足，不请求 API。存在片段时通过提示词要求拒答和标注来源，但并不能保证模型判断或引用完全正确。做实验时应人工核对证据，并准备答案不在语料中的问题。

## 文件与验证

- `rag.py`：RAG 和检索实验的公共函数库，提供数据结构、文档处理、编码、检索、生成与评价函数。
- `BGE/run.py`：BGE 的检索器、向量入库和五个命令入口。
- `BM25/run.py`：BM25 的检索器、词项入库和五个命令入口。
- `run_experiments.py`：共同实验参数、两方法统一评估和报告汇总。
- `data/docs/`：15 份公开来源的中文整理语料，按三个领域分目录。
- `data/README.md`、`data/sources.json`：文档清单、来源链接、处理说明、字数和许可。
- `data/raw/`：15 个原始网页的文字快照，不加入默认索引。
- `storage/`：BGE 向量索引与 BM25 词项索引。
- `tests/test_rag.py`：分块、编码格式、检索、来源、空证据与损坏索引检查。

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

依赖用 Sentence Transformers 加载现成编码器，用 OpenAI SDK 直接调用 DeepSeek 的兼容接口；它们不是 RAG 编排框架。文档在本地编码，生成部分按 DeepSeek API 计费。

## Windows DLL 加载问题

如果启动时报 `WinError 126`，并提到 `torch/lib/shm.dll`，可能是它依赖的 Visual C++ 运行库缺失，而不是 `shm.dll` 本身丢失。本机曾缺少 `VCRUNTIME140_THREADS.dll`；已将电脑上已有、微软数字签名有效的副本补入当前 `.venv/Lib/site-packages/torch/lib/`，使项目不再依赖 Codex 的额外 PATH。

如果重建 `.venv` 或换电脑，建议安装或更新 [微软官方 x64 Visual C++ 运行库](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)，然后重新运行。无需重新建立文档索引。

官方参考：[DeepSeek 接口](https://api-docs.deepseek.com/zh-cn/)、[BGE 模型及查询指令](https://huggingface.co/BAAI/bge-small-zh-v1.5)。

## BM25 与 BGE Dense Retrieval 检索实验

公共代码集中在 `rag.py`：原有文档加载、分块、BGE 编码、JSON 索引和余弦检索继续复用，新增 Chunk ID、测试集校验、Recall/Hit、结果导出与对比图。该函数库没有 `main()` 或直接运行入口，不解析命令行参数；实验函数通过普通参数接收数据。`BGE/run.py` 和 `BM25/run.py` 各自组织入库、搜索、问答、聊天和评估；`run_experiments.py` 负责实验参数解析、统一运行和对比。以后增加方法时，可建立新方法目录，提供 `chunk_ids`、`config` 和 `search(question, k)`，复用公共函数，再注册到总入口。

例如，在自己的实验代码中直接使用公共函数：

```python
import rag
from BM25.run import BM25Retrieval

index, dataset, metadata = rag.load_experiment_data(
    rag.ROOT / "storage/index.json", rag.ROOT / "data/test_queries.json")
report, result_dir = rag.run_retrieval_experiment(
    "BM25", BM25Retrieval, index, dataset, metadata)
```

### 直接体验

本机实验依赖已安装。新环境在项目根目录安装：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-experiment.txt
```

统一运行两种方法，计算 Recall@1/3/5 与 Hit@1/3/5，并生成一份可直接打开的 HTML 报告：

```powershell
.\.venv\Scripts\python.exe run_experiments.py
```

也可以分别评估：

```powershell
.\.venv\Scripts\python.exe -m BGE.run evaluate
.\.venv\Scripts\python.exe -m BM25.run evaluate
```

这里 `-m BGE.run` 表示以模块方式运行 `BGE/run.py`。请在项目根目录运行。不写子命令时仍默认执行 evaluate，保留之前的实验用法。BM25 的检索和评估不初始化 BGE；ask/chat 共用 DeepSeek 生成。evaluate 全程不调用 DeepSeek，不构造生成提示词，不写 `data/queryhistory.json`，也不重建或改写现有索引。BGE 查询设备沿用原配置，可用 `--device cpu` 或 `--device cuda` 覆盖。

### 共同语料与测试集

evaluate 默认固定使用 `storage/index.json` 中的 15 份文档、33 个 Chunk。BM25 也可用 `evaluate --index storage/bm25.json` 评价自己的索引；Chunk 指纹必须与测试集一致。两种方法读取相同索引里的原文，ID 形如 `astronomy/01_sun.md::pnone::c0001`，由相对路径、页码和文档内编号产生，原索引格式保持不变。实验检查 Chunk ID 集合、行顺序和文本，并记录包含行顺序与文本的语料指纹。若修改文档后重新建索引，应重新审核测试题；程序不会自动迁移旧标注。

`data/test_queries.json` 包含 48 道 AI 辅助整理的题目，三个领域各 16 道，9 篇长文各 4 道、6 篇短文各 2 道。题目覆盖术语、事实、数字表达、语义改写、条件查询和概念辨析，覆盖全部 33 个 Chunk。题目与标注在首次运行检索前依据语料和原文证据确定，不通过检索结果生成或调整。

**当前 48 道题的 `review_status` 均为 `pending`，结果属于预实验，不能表述为人工标注测试集上的正式结果。** 为方便先体验，默认允许运行并显式标记 `preliminary_ai_assisted`。原有 `data/retrieval_checks.json` 仍是独立的自检材料，没有直接拿来计算本实验指标。

测试集顶层包含 `version`、`corpus_fingerprint` 与 `queries`，每道题形如：

```json
{
  "query_id": "q002",
  "domain": "astronomy",
  "query_type": "术语查询",
  "question": "我们通常看到的太阳发光表面，在结构上叫什么层？",
  "answer": "光球层。",
  "relevant_chunk_ids": ["astronomy/01_sun.md::pnone::c0001"],
  "evidence": [{
    "chunk_id": "astronomy/01_sun.md::pnone::c0001",
    "quote": "通常所说的太阳表面指光球层，这是人们从地球观察太阳时主要看到的发光区域。"
  }],
  "review_status": "pending",
  "reviewed_by": "",
  "manual_notes": ""
}
```

主测试采用单个 Chunk 能独立支持完整答案的问题；若重叠片段也能完整回答，均应标注为相关。多个相关 ID 表示可替代的完整证据，不表示需要组合片段才能回答。原文引用的字符串校验只验证证据确实存在，不代替人工相关性判断，也不使用 `answer in text` 计算命中。

先检查数据或导出含完整 Chunk 原文的审核材料：

```powershell
.\.venv\Scripts\python.exe run_experiments.py --validate-only
.\.venv\Scripts\python.exe run_experiments.py --validate-only --export-review .cache/test_queries_review.md
```

审核时逐题核对答案与证据，检查是否漏标其他可完整作答的 Chunk。只读审核表不会更新 JSON。确认后在 JSON 中将对应题的 `review_status` 改为 `reviewed`，并填写 `reviewed_by`。全部确认后冻结测试集，通过以下命令运行正式模式；仍有 pending 时程序会在加载检索器前报错：

```powershell
.\.venv\Scripts\python.exe run_experiments.py --require-reviewed
```

### 比较口径

- BGE 仅负责将问题和 Chunk 编码为向量。实验直接复用现有文档向量，对问题调用原来的 `Encoder.encode(query=True)`，保留中文查询指令、L2 归一化和余弦点积检索，不训练模型。
- BM25 使用 Jieba 精确分词，`HMM=False`，英文转小写，删除空白与纯标点，保留中文、英文和数字。不使用额外停用词表或自定义词典。BM25Okapi 参数固定为 `k1=1.5, b=0.75, epsilon=0.25`。
- 两种方法均返回 Top-5，再取前 1、3、5 个计算指标，无额外过滤、融合或重排。并列分数按原索引顺序排序。BM25 无匹配时仍返回分数为零的并列 Top-K，便于检查，不能把零分当成有效语义匹配。
- 每题 `Recall@K = Top-K 中相关 Chunk 数 / 全部已标注相关 Chunk 数`；`Hit@K` 表示是否至少命中一个相关 Chunk。汇总对每道题等权平均。只有一个相关 Chunk 的题，其 Recall 与 Hit 相同。
- 本实验使用基于 Chunk ID 的相关性命中，并非另行人工判断的 answer-containing passage 指标。
- **主要失败口径为 `Hit@3=0`，每种方法尽量选取 3 个失败案例。** 实际少于 3 个时如实提示并导出已有案例；不保证失败数量，不依据结果删改测试题或 Ground Truth。

### 输出位置

**日常只需打开 `results/report.html`。** 指标、对比图、失败案例和全部问题都在这一份报告里，图表直接嵌入文件，可离线查看。展开问题能查看标准答案、正确证据和两种方法的 Top-3 原文；Top-4/5 可以继续展开，也可搜索或筛选失败题。

```text
results/
├── report.html       # 日常查看的实验报告
├── details.json      # 完整运行数据，需要复核或编程分析时再看
└── history/          # 上一轮结果自动归档，平时不用打开
```

|文件|内容|
|---|---|
|`report.html`|指标汇总、内嵌图表、失败候选、可搜索和展开的逐题检索结果|
|`details.json`|同一轮所有方法的元数据、指纹、配置、依赖版本、完整原文、Ground Truth、Top-5 和指标|
|`history/<上一轮编号>/`|上一轮的报告、数据及可选 CSV，保留人工填写的备注|

默认不生成 CSV 或独立 PNG，也不再向 `data/`、`BGE/`、`BM25/` 写实验结果。现有旧格式结果保留在 `results/history/legacy/`。所有生成结果均已加入 `.gitignore`。

单独执行 BGE 或 BM25 的 `evaluate` 也更新同一个报告入口，标题明确标明本轮方法，上一轮完整结果自动归档。需要两种方法的对比报告时，运行 `run_experiments.py`。新报告准备成功后才归档上一轮；统一实验中任一方法失败时保留旧报告。

需要 Excel 表格时，再加导出参数：

```powershell
.\.venv\Scripts\python.exe run_experiments.py --export-csv
```

这会额外创建 `results/exports/`，包含 `summary.csv` 和各方法的 `<方法>_queries.csv`、`<方法>_failures.csv`、`<方法>_failure_cases.csv`。分别保存完整逐题结果、全部 Top-3 失败题、最多 3 个分析候选；候选优先选择本方法失败、另一方法成功的题。CSV 使用 UTF-8 BOM，便于 Excel 识别中文。下一次运行会连同你填写的备注一起归档；未加参数时不保留上一轮 CSV 在当前目录中。

可以用 `--index`、`--queries` 指定其他索引或测试集，语料指纹必须匹配。`--output-dir` 可改变统一报告目录；`--comparison-dir` 是它的旧参数别名。两种方法的分数尺度不同，仅各自用于排序，不直接比较分值大小。

### 人工失败分析

`failure_reason` 和 `manual_notes` 导出时留空，供你在可选 CSV 中填写。建议每例记录：问题和标准答案、相关 Chunk、两种方法的 Top-3、失败原因、支持判断的原文，以及是否涉及关键词差异、语义改写、相似片段混淆或分块边界。若失败不足目标，保留真实结果；新增独立题目时应记录测试集新版本，避免只挑已知失败题回填原测试集。

验证原功能和实验公共逻辑：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
