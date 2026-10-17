"""BGE 入口：ingest / search / ask / chat / evaluate。"""

import argparse
import os
from pathlib import Path
import sys

import rag
from run_experiments import add_experiment_arguments, command_error, evaluate_method

DEFAULT_INDEX = rag.ROOT / "storage/index.json"


class BGEDenseRetrieval:
    """BGE 只编码问题，文档向量和余弦搜索复用公共函数。"""

    # 用途：复用已有文档向量，准备好对用户问题进行编码和向量检索的对象。
    # 输入：已加载的向量索引和可选计算设备；检查向量是否归一化，并加载索引指定的编码模型。
    # 结果：在对象中保存索引、片段 ID、编码器和检索配置；不重新编码文档、不写索引。
    def __init__(self, index: dict, device: str | None = None):
        if not index.get("embedding_model") or not index.get("vectors"):
            raise ValueError("BGE 检索需要向量索引，请先运行 python -m BGE.run ingest")
        for vector in index["vectors"]:
            if abs(sum(value * value for value in vector) - 1) > 1e-3:
                raise ValueError("Dense 索引向量未归一化，请重新 ingest")
        rag.load_config()
        self.index = index
        self.chunk_ids = list(rag.get_experiment_chunks(index))
        self.encoder = rag.Encoder(index["embedding_model"], device or os.getenv("EMBEDDING_DEVICE", "cpu"))
        self.config = {
            "model": index["embedding_model"], "device": str(self.encoder.model.device),
            "query_instruction": rag.BGE_INSTRUCTION, "normalize_embeddings": True,
            "similarity": "cosine via dot product", "min_score": -1.0,
            "document_vectors": "reused from existing index", "tie_break": "index order",
        }

    # 用途：用 BGE 编码问题，再由公共向量检索函数查找最相似的前 k 个片段。
    # 输入：问题文本和返回数量 k；问题编码会按当前中文 BGE 规则添加查询指令。
    # 返回：包含片段 ID、原文、来源、相似度和排名的列表；不生成答案、不写查询历史。
    def search(self, question: str, k: int = 5) -> list[dict]:
        vector = self.encoder.encode([question], query=True)[0]
        hits = rag.retrieve(self.index, vector, top_k=k)
        return rag.hits_to_results(hits)


# 用途：执行 ingest 命令，将原始文档编码为可供 BGE 路径检索的向量索引。
# 输入：文档目录、索引路径、分块参数，以及可选的编码模型和计算设备。
# 结果：读取并分块，调用公共编码器生成文档向量，重建 args.index（默认 storage/index.json）。
def ingest(args: argparse.Namespace) -> None:
    """文档 → 共同分块 → BGE 编码 → 向量索引。"""
    rag.split_text("", args.chunk_size, args.overlap)
    rag.load_config()
    chunks = rag.read_documents(args.docs, args.chunk_size, args.overlap)
    model = args.embedding_model or os.getenv("EMBEDDING_MODEL", rag.DEFAULT_MODEL)
    encoder = rag.Encoder(model, args.device or os.getenv("EMBEDDING_DEVICE", "cpu"))
    vectors = encoder.encode([chunk.text for chunk in chunks])
    rag.save_index(args.index, chunks, vectors, model, args.docs, args.chunk_size, args.overlap)
    print(f"BGE 已入库 {len(chunks)} 个 Chunk：{args.index.resolve()}")


# 用途：统一执行 search（仅检索）、ask（单次问答）和 chat（循环独立问答）。
# 输入：命令类型、问题、向量索引路径、计算设备、Top-K、分数阈值及原文显示选项。
# 结果：在终端显示结果，并通过公共函数追加查询历史；问答有可用证据时才调用 DeepSeek。
# 每个问题都用 BGE 编码；chat 不将上一题作为上下文，输入 /exit 等退出指令结束。
def query(args: argparse.Namespace) -> None:
    """搜索和问答共用同一个检索器；聊天逐题调用公共问答函数。"""
    if args.top_k < 1 or not -1 <= args.min_score <= 1:
        raise ValueError("要求 top_k >= 1，且 -1 <= min_score <= 1")
    retriever = BGEDenseRetrieval(rag.load_index(args.index), args.device)
    while True:
        question = input("\n你（输入 /exit 退出）> ") if args.command == "chat" else args.question
        question = question.strip()
        if args.command == "chat" and question.lower() in {"/exit", "/quit", "exit", "quit"}:
            return
        if not question and args.command == "chat":
            continue
        response = rag.run_query(
            question, retriever, top_k=args.top_k, min_score=args.min_score,
            generate_answer=args.command != "search", history_command=args.command, method="BGE")
        if response.answer is not None:
            print("\n" + response.answer + "\n\n检索来源：")
        rag.print_sources(response.hits, args.show_context or args.command == "search")
        if args.command != "chat":
            return


# 用途：定义 BGE 五个子命令各自接受的参数、默认值和 --help 说明。
# 返回：ArgumentParser 对象，供 main() 解析命令行；本函数不加载模型或执行业务操作。
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    ingest_parser = commands.add_parser("ingest", help="对文档编码并保存向量索引")
    ingest_parser.add_argument("--docs", type=Path, default=rag.ROOT / "data/docs")
    ingest_parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    ingest_parser.add_argument("--embedding-model", default=None)
    ingest_parser.add_argument("--device", default=None)
    ingest_parser.add_argument("--chunk-size", type=int, default=350)
    ingest_parser.add_argument("--overlap", type=int, default=60)

    for name, description in (("search", "只检索"), ("ask", "检索后问答"), ("chat", "循环独立问答")):
        query_parser = commands.add_parser(name, help=description)
        if name != "chat":
            query_parser.add_argument("question")
        query_parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
        query_parser.add_argument("--device", default=None)
        query_parser.add_argument("--top-k", type=int, default=3)
        query_parser.add_argument("--min-score", type=float, default=-1.0)
        query_parser.add_argument("--show-context", action="store_true")

    evaluate_parser = commands.add_parser("evaluate", help="只评估检索，不调用 DeepSeek")
    add_experiment_arguments(evaluate_parser)
    return parser


# 用途：BGE 命令行总入口，解析参数后分派到入库、查询或公共评估流程。
# 输入：可选的参数列表；省略时读取真实命令行，无子命令时默认 evaluate。
# 返回：成功或用户主动退出时为 0，已处理的运行错误为 1；参数解析错误由 argparse 处理。
def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    # 保留原来的无子命令实验用法，如 python -m BGE.run --device cpu。
    if not arguments or (arguments[0].startswith("-") and arguments[0] not in {"-h", "--help"}):
        arguments.insert(0, "evaluate")
    args = build_parser().parse_args(arguments)
    try:
        if args.command == "ingest":
            ingest(args)
        elif args.command == "evaluate":
            evaluate_method("BGE", BGEDenseRetrieval, args)
        else:
            query(args)
        return 0
    except (ValueError, OSError, ImportError, KeyError, TypeError, RuntimeError) as exc:
        return command_error(exc)
    except (KeyboardInterrupt, EOFError):
        print("\n已退出。")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
