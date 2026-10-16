# 纯 Python + DeepSeek 的简单 RAG

适合约 15 份短文档的小型知识库和课程实验。没有 LangChain、LlamaIndex 或向量数据库：读取、分块、检索和提示词都在 `rag.py`，便于理解和修改。

流程：`文档 → 分块 → 本地 BGE 向量 → JSON 索引 → 余弦 Top-K → DeepSeek → 带引用的回答`。

## 安装（Windows PowerShell）

建议使用 Python 3.11 或 3.12。普通 Python 安装后执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

不需要激活虚拟环境，也不受 PowerShell 脚本执行策略影响。如果 `python` 指向 Windows 商店占位程序，请使用实际 Python 路径执行第一条命令。

当前工作区已安装 `.venv`。`requirements-lock.txt` 记录本次验证环境的精确依赖版本；需要复现实验时可用它替代 `requirements.txt` 安装。

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

## 先用示例跑通

`examples/docs` 是两份明确标记的自编演示资料，不是正式公开语料。示例索引独立保存：

```powershell
.\.venv\Scripts\python.exe rag.py ingest --docs examples/docs --index storage/demo.json
.\.venv\Scripts\python.exe rag.py search "RAG 的基本流程是什么？" --index storage/demo.json
.\.venv\Scripts\python.exe rag.py ask "RAG 的基本流程是什么？" --index storage/demo.json --show-context
```

`search` 仅在本地检索；`ask` 才调用 DeepSeek。只有问题和检索到的片段会发送给 DeepSeek。

## 使用自己的文档

1. 将正式文档放入 `data/docs`（支持子目录）。支持 `.txt`、`.md` 和文字版 `.pdf`；TXT/Markdown 支持 UTF-8 和 GB18030。扫描 PDF 要先做 OCR。
2. 建立索引，然后提问：

```powershell
.\.venv\Scripts\python.exe rag.py ingest
.\.venv\Scripts\python.exe rag.py ask "你的问题" --show-context
.\.venv\Scripts\python.exe rag.py chat
```

`chat` 是循环独立问答，每个问题需自包含，历史记录不会作为下一次问答的上下文。输入 `/exit` 退出。修改文档、分块参数或编码器后需要重新 `ingest`。查询自动使用索引记录的编码器。

每次成功完成 `ask`、`chat` 或 `search` 查询后，会自动追加到项目的 `data/queryhistory.json`（本机为 `E:\codex_project\thesis_1\data\queryhistory.json`）。文件是 UTF-8 JSON 数组，每条记录包含 UTC 时间 `timestamp`、命令 `command`、问题 `question`、检索结果 `retrieval_results`（编号、来源、页码、片段编号、原文及相似度）、完整提示词消息 `prompt` 和答案 `answer`。`search` 不生成答案，其 `prompt` 和 `answer` 为 `null`；未检索到片段的问答仍保存构造的提示词和资料不足的答案，但不调用 API。文件不存在时自动创建，后续查询保留已有记录；生成失败的查询不写入。历史文件包含问题与资料原文，适合本地留存；同时运行多个查询进程写入同一文件暂不支持。

默认每块最多 350 个字符，重叠 60 个字符，检索 Top-3。保存文档相对路径、PDF 页码和片段编号，回答用 `[1]` 等编号对应终端中的检索来源。可以调整：

```powershell
.\.venv\Scripts\python.exe rag.py ingest --chunk-size 300 --overlap 50
.\.venv\Scripts\python.exe rag.py ask "你的问题" --top-k 4 --min-score 0.6
```

注意：分块长度按字符计算；程序会检查编码器 token 上限，过长时提示缩小片段。余弦分数不是概率，`0.6` 仅演示用法，需要在自己的语料上校准；默认不过滤。没有片段通过阈值时，程序直接返回资料不足，不请求 API。存在片段时通过提示词要求拒答和标注来源，但并不能保证模型判断或引用完全正确。做实验时应人工核对证据，并准备答案不在语料中的问题。

## 文件与验证

- `rag.py`：所有核心流程及 CLI。
- `data/docs/`：15 份公开来源的中文整理语料，按三个领域分目录。
- `data/README.md`、`data/sources.json`：文档清单、来源链接、处理说明、字数和许可。
- `data/raw/`：15 个原始网页的文字快照，不加入默认索引。
- `examples/docs/`：可运行的演示材料。
- `storage/`：生成的 JSON 索引，包含原文片段和向量。
- `tests/test_rag.py`：分块、编码格式、检索、来源、空证据与损坏索引检查。

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

依赖用 Sentence Transformers 加载现成编码器，用 OpenAI SDK 直接调用 DeepSeek 的兼容接口；它们不是 RAG 编排框架。文档在本地编码，生成部分按 DeepSeek API 计费。

## Windows DLL 加载问题

如果启动时报 `WinError 126`，并提到 `torch/lib/shm.dll`，可能是它依赖的 Visual C++ 运行库缺失，而不是 `shm.dll` 本身丢失。本机曾缺少 `VCRUNTIME140_THREADS.dll`；已将电脑上已有、微软数字签名有效的副本补入当前 `.venv/Lib/site-packages/torch/lib/`，使项目不再依赖 Codex 的额外 PATH。

如果重建 `.venv` 或换电脑，建议安装或更新 [微软官方 x64 Visual C++ 运行库](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)，然后重新运行。无需重新建立文档索引。

官方参考：[DeepSeek 接口](https://api-docs.deepseek.com/zh-cn/)、[BGE 模型及查询指令](https://huggingface.co/BAAI/bge-small-zh-v1.5)。
