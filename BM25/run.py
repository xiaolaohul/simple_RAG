"""BM25 入口：ingest / search / ask / chat / evaluate；检索不调用 BGE。"""

import argparse
from dataclasses import asdict
import math
from pathlib import Path
import sys

import rag
from run_experiments import add_experiment_arguments, command_error, evaluate_method

DEFAULT_INDEX = rag.ROOT / "storage/bm25.json"


class BM25Retrieval:
    """对共同 Chunk 分词并排序，也可加载入库时保存的词项。"""

    # 用途：根据索引中的片段建立可供搜索的 BM25 检索器。
    # 输入：已加载的索引数据；有已保存词项时复用，否则现场分词。device 仅为统一接口保留。
    # 结果：在对象中保存片段、词项、分词配置及词频/IDF 等统计；不写文件、不调用 BGE。
    def __init__(self, index: dict, device: str | None = None):
        import jieba
        from rank_bm25 import BM25Okapi

        self.chunks = list(rag.get_experiment_chunks(index).values())
        self.chunk_ids = [rag.chunk_id(chunk) for chunk in self.chunks]
        self.tokenizer = jieba.Tokenizer()
        self.config = {
            "algorithm": "BM25Okapi", "k1": 1.5, "b": .75, "epsilon": .25,
            "tokenizer": "jieba accurate mode", "HMM": False, "lowercase": True,
            "remove_whitespace_and_punctuation": True, "stopwords": None,
            "custom_dictionary": None, "score_threshold": None, "tie_break": "index order",
        }
        if index.get("index_type") == "bm25":
            if index.get("bm25_config") != self.config:
                raise ValueError("BM25 索引分词配置已变化，请重新 ingest")
            tokens = index.get("tokenized_corpus")
            if (not isinstance(tokens, list) or len(tokens) != len(self.chunks)
                    or any(not isinstance(row, list) or not all(isinstance(word, str) and word for word in row)
                           for row in tokens)):
                raise ValueError("BM25 词项索引损坏，请重新 ingest")
            self.document_tokens = tokens
        else:
            self.document_tokens = [self.tokenize(chunk["text"]) for chunk in self.chunks]
        if not any(self.document_tokens):
            raise ValueError("BM25 语料没有可索引的词")
        self.bm25 = BM25Okapi(self.document_tokens, k1=self.config["k1"],
                             b=self.config["b"], epsilon=self.config["epsilon"])

    # 用途：将文档片段或问题转换成 BM25 用来匹配和统计的词项。
    # 输入：一段文本；英文转小写，使用 Jieba 精确分词，关闭 HMM，过滤纯空白和纯标点。
    # 返回：保留重复词的字符串列表；不删除停用词、不生成语义向量。
    def tokenize(self, text: str) -> list[str]:
        return [token for token in self.tokenizer.cut(text.lower(), HMM=False)
                if any(character.isalnum() for character in token)]

    # 用途：对问题分词，计算所有片段的 BM25 分数，并选出分数最高的前 k 个。
    # 输入：问题文本和返回数量 k；同分时按原索引顺序排列，零分结果也可能被返回。
    # 返回：包含片段 ID、原文、来源、分数和排名的列表；不生成答案、不写查询历史。
    def search(self, question: str, k: int = 5) -> list[dict]:
        if k < 1:
            raise ValueError("k 必须 >= 1")
        scores = self.bm25.get_scores(self.tokenize(question))
        order = sorted(range(len(scores)), key=lambda position: (-float(scores[position]), position))[:k]
        return [{"chunk_id": self.chunk_ids[position], **self.chunks[position],
                 "score": float(scores[position]), "rank": rank}
                for rank, position in enumerate(order, 1)]


# 用途：执行 ingest 命令，将文档准备成后续 BM25 检索可读取的索引。
# 输入：文档目录、索引路径、分块大小和重叠长度等命令行参数。
# 结果：读取并分块、分词，重建 args.index（默认 storage/bm25.json），保存原文、词项和配置。
def ingest(args: argparse.Namespace) -> None:
    """文档 → 共同分块 → Jieba 分词 → 词项索引，全程不加载 BGE。"""
    rag.split_text("", args.chunk_size, args.overlap)
    chunks = rag.read_documents(args.docs, args.chunk_size, args.overlap)
    index = {
        "version": 1, "index_type": "bm25", "docs_dir": str(args.docs.resolve()),
        "chunk_size": args.chunk_size, "overlap": args.overlap,
        "chunks": [asdict(chunk) for chunk in chunks],
    }
    # 此时尚未写入缓存，直接从共同 Chunk 建立词项。
    retriever = BM25Retrieval({"chunks": index["chunks"]})
    index.update({"corpus_fingerprint": rag.corpus_fingerprint(index),
                  "bm25_config": retriever.config, "tokenized_corpus": retriever.document_tokens})
    rag.save_json(args.index, index)
    print(f"BM25 已入库 {len(chunks)} 个 Chunk：{args.index.resolve()}")


# 用途：统一执行 search（仅检索）、ask（单次问答）和 chat（循环独立问答）。
# 输入：命令类型、问题、索引路径、Top-K、分数阈值及是否显示原文等参数。
# 结果：在终端显示结果，并通过公共函数追加查询历史；问答有可用证据时才调用 DeepSeek。
# chat 中每题独立处理，输入 /exit 等退出指令结束；所有检索均不调用 BGE。
def query(args: argparse.Namespace) -> None:
    if args.top_k < 1 or (args.min_score is not None and not math.isfinite(args.min_score)):
        raise ValueError("要求 top_k >= 1，且 min_score 为有限数值")
    retriever = BM25Retrieval(rag.load_corpus_index(args.index))
    if args.command != "search":
        rag.load_config()  # 仅问答需要读取 DeepSeek 配置。
    while True:
        question = input("\n你（输入 /exit 退出）> ") if args.command == "chat" else args.question
        question = question.strip()
        if args.command == "chat" and question.lower() in {"/exit", "/quit", "exit", "quit"}:
            return
        if not question and args.command == "chat":
            continue
        response = rag.run_query(
            question, retriever, top_k=args.top_k, min_score=args.min_score,
            generate_answer=args.command != "search", history_command=args.command, method="BM25")
        if response.answer is not None:
            print("\n" + response.answer + "\n\n检索来源：")
        rag.print_sources(response.hits, args.show_context or args.command == "search", score_label="BM25 分数")
        if args.command != "chat":
            return


# 用途：定义 BM25 五个子命令各自接受的参数、默认值和 --help 说明。
# 返回：ArgumentParser 对象，供 main() 解析命令行；本函数不执行入库、检索或评估。
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    ingest_parser = commands.add_parser("ingest", help="分词并保存 BM25 索引，不调用 BGE")
    ingest_parser.add_argument("--docs", type=Path, default=rag.ROOT / "data/docs")
    ingest_parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    ingest_parser.add_argument("--chunk-size", type=int, default=350)
    ingest_parser.add_argument("--overlap", type=int, default=60)

    for name, description in (("search", "只检索"), ("ask", "检索后问答"), ("chat", "循环独立问答")):
        query_parser = commands.add_parser(name, help=description)
        if name != "chat":
            query_parser.add_argument("question")
        query_parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
        query_parser.add_argument("--top-k", type=int, default=3)
        query_parser.add_argument("--min-score", type=float, default=None, help="可选的 BM25 分数下限")
        query_parser.add_argument("--show-context", action="store_true")

    evaluate_parser = commands.add_parser("evaluate", help="只评估检索，不调用 DeepSeek")
    add_experiment_arguments(evaluate_parser)
    return parser


# 用途：BM25 命令行总入口，解析参数后分派到入库、查询或公共评估流程。
# 输入：可选的参数列表；省略时读取真实命令行，无子命令时默认 evaluate。
# 返回：成功或用户主动退出时为 0，已处理的运行错误为 1；参数解析错误由 argparse 处理。
def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    # 无子命令时继续运行原来的检索实验。
    if not arguments or (arguments[0].startswith("-") and arguments[0] not in {"-h", "--help"}):
        arguments.insert(0, "evaluate")
    args = build_parser().parse_args(arguments)
    try:
        if args.command == "ingest":
            ingest(args)
        elif args.command == "evaluate":
            evaluate_method("BM25", BM25Retrieval, args)
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
