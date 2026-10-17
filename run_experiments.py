"""运行 BGE/BM25，共用一份语料与测试集，导出对比结果。"""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
from typing import Callable

import rag


# 用途：为 BGE、BM25 的 evaluate 和统一实验入口注册同一套评估参数。
# 输入：已有的 ArgumentParser；向其中添加索引、测试集、输出目录、审核和导出等选项。
# 结果：修改解析器，不解析参数、不加载数据或模型，也不写文件。
def add_experiment_arguments(parser: argparse.ArgumentParser) -> None:
    """各方法的 evaluate 和总入口共用相同的实验参数。"""
    parser.add_argument("--index", type=Path, default=rag.ROOT / "storage/index.json")
    parser.add_argument("--queries", type=Path, default=rag.ROOT / "data/test_queries.json")
    parser.add_argument("--device", default=None, help="BGE 编码设备，如 cpu、cuda；默认读取原配置")
    parser.add_argument("--output-dir", "--comparison-dir", type=Path, default=None,
                        help="统一报告目录，默认 results；comparison-dir 为旧参数别名")
    parser.add_argument("--export-csv", action="store_true", help="额外导出 Excel 分析用 CSV")
    parser.add_argument("--require-reviewed", action="store_true", help="只允许人工审核通过的正式测试集")
    parser.add_argument("--validate-only", action="store_true", help="仅校验语料和标注，不加载检索模型")
    parser.add_argument("--export-review", type=Path, help="导出含原文的 Markdown 审核材料")


# 用途：在加载检索器前检查索引、测试集、语料指纹和审核状态，确保评价数据有效。
# 输入：索引与测试集路径、是否要求人工审核、可选的审核材料导出路径等参数。
# 返回：(索引数据, 测试集, 元数据)；打印数据摘要，指定 --export-review 时另写审核 Markdown。
def prepare_experiment(args: argparse.Namespace) -> tuple[dict, dict, dict]:
    """先校验共同数据，再按需导出审核表；此时还未加载检索器。"""
    context = rag.load_experiment_data(args.index, args.queries, require_reviewed=args.require_reviewed)
    index, dataset, metadata = context
    print(f"实验数据：{metadata['document_count']} 份文档，{metadata['chunk_count']} 个 Chunk，"
          f"{metadata['query_count']} 道题；指标 K=1/3/5，失败口径 Top-{rag.FAILURE_K}。", flush=True)
    if metadata["pending_review_count"]:
        print(f"提示：{metadata['pending_review_count']} 道题尚待人工审核，当前输出为预实验结果。", flush=True)
    if args.export_review:
        rag.export_testset_review(args.export_review, index, dataset)
        print(f"审核材料：{args.export_review.resolve()}", flush=True)
    return context


# 用途：创建指定方法的检索器，逐题计算指标，并打印 Top-3 指标和失败案例数量。
# 输入：方法名称、检索器构造函数 factory、命令参数、已校验数据及可选运行编号。
# 返回：(本方法完整报告, 输出目录)；这里只计算结果，报告文件由调用方统一保存。
# BGE 会编码测试问题，BM25 使用词项评分；本函数不进入 DeepSeek 问答或查询历史流程。
def run_method(method: str, factory: Callable, args: argparse.Namespace,
               context: tuple[dict, dict, dict], run_id: str | None = None) -> tuple[dict, Path]:
    """调用公共评价函数，并在终端显示该方法的摘要。"""
    if method == "BGE" and context[0].get("embedding_model") != rag.DEFAULT_MODEL:
        raise ValueError(f"当前对照实验固定使用 {rag.DEFAULT_MODEL}，请使用对应向量索引")
    report, result_dir = rag.run_retrieval_experiment(
        method, factory, *context, device=args.device, output_dir=args.output_dir,
        run_id=run_id, save_results=False)
    summary = report["summary"]
    print(f"{method}：Recall@3={summary['Recall@3']:.2%}，Hit@3={summary['Hit@3']:.2%}；"
          f"Top-3 失败 {summary['failures_at_3']} 题。", flush=True)
    if summary["failures_at_3"] < rag.FAILURE_CASE_TARGET:
        print(f"提示：{method} 当前只有 {summary['failures_at_3']} 个 Top-3 失败案例，"
              f"不足目标 {rag.FAILURE_CASE_TARGET} 个；已保留真实结果。", flush=True)
    return report, result_dir


# 用途：将运行异常转换成统一的终端错误提示，供三个命令行入口共用。
# 输入：捕获的异常；隐藏提示中与当前环境 API key 相同的文本，缺依赖时补充安装说明。
# 返回：表示失败的退出码 1；错误写到标准错误输出，不修改业务数据。
def command_error(exc: Exception) -> int:
    message = str(exc)
    key = os.getenv("DEEPSEEK_API_KEY", "")
    if key:
        message = message.replace(key, "[REDACTED]")
    if isinstance(exc, ImportError):
        message += "；请先安装 requirements-experiment.txt 中的依赖"
    print(f"运行错误：{message}", file=sys.stderr)
    return 1


# 用途：承接 BGE.run / BM25.run 的 evaluate 子命令，完成单方法的数据检查、评估和报告保存。
# 输入：方法名称、检索器构造函数及命令参数；--validate-only 时仅检查数据和按需导出审核表。
# 结果：正常评估时更新统一目录中的 report.html、details.json，归档旧结果，按需导出 CSV。
def evaluate_method(method: str, factory: Callable, args: argparse.Namespace) -> None:
    """供各方法的 evaluate 子命令调用；不进入问答流程。"""
    context = prepare_experiment(args)
    if not args.validate_only:
        report, output_dir = run_method(method, factory, args, context)
        report_path = rag.save_experiment_results({method: report}, output_dir, export_csv=args.export_csv)
        print(f"查看报告：{report_path.resolve()}", flush=True)


# 用途：统一实验入口，默认用同一份数据评估 BM25 与 BGE，也支持 --method 只评估一种方法。
# 输入：可选参数列表；省略时读取真实命令行，--validate-only 时在数据检查后结束。
# 结果：所选方法全部成功后保存一份报告和完整数据，归档旧结果，按需导出 CSV。
# 返回：成功为 0，已处理的运行错误为 1，用户中断为 130；参数解析错误由 argparse 处理。
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_experiment_arguments(parser)
    parser.add_argument("--method", choices=("both", "bge", "bm25"), default="both")
    args = parser.parse_args(argv)
    try:
        context = prepare_experiment(args)
        if args.validate_only:
            return 0
        from BGE.run import BGEDenseRetrieval
        from BM25.run import BM25Retrieval

        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        reports = {}
        output_dir = args.output_dir or rag.ROOT / "results"
        for method, factory in (("BM25", BM25Retrieval), ("BGE", BGEDenseRetrieval)):
            if args.method != "both" and args.method != method.lower():
                continue
            reports[method], _ = run_method(method, factory, args, context, run_id)
        # 两种方法全部成功后只发布一次，避免出现只有一半数据的联合报告。
        report_path = rag.save_experiment_results(reports, output_dir, export_csv=args.export_csv)
        print(f"查看报告：{report_path.resolve()}", flush=True)
        return 0
    except (ValueError, OSError, ImportError, KeyError, TypeError, RuntimeError) as exc:
        return command_error(exc)
    except (KeyboardInterrupt, EOFError):
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
