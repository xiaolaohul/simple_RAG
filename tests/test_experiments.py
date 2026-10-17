import copy
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import rag
from BGE.run import BGEDenseRetrieval
from BM25.run import BM25Retrieval
from BGE import run as bge
from BM25 import run as bm25


def example_index():
    return {"version": 1, "embedding_model": rag.DEFAULT_MODEL,
            "chunks": [dict(source=f"{i}.txt", page=None, number=1, text=text)
                       for i, text in enumerate(("月球 极区 水冰", "太阳 核心 聚变", "网页 样式 CSS",
                                                 "火星 沙尘 大气", "海水 潮差 引力"))],
            "vectors": [[float(i == j) for j in range(5)] for i in range(5)]}


def example_dataset(index):
    cid = rag.chunk_id(index["chunks"][0])
    return {"version": 1, "corpus_fingerprint": rag.corpus_fingerprint(index), "queries": [
        {"query_id": "q001", "question": "哪里有水冰？", "answer": "月球极区。", "domain": "astronomy",
         "query_type": "事实", "relevant_chunk_ids": [cid],
         "evidence": [{"chunk_id": cid, "quote": "月球 极区 水冰"}],
         "review_status": "pending", "reviewed_by": ""}]}


class FakeRetriever:
    def __init__(self, index):
        self.index = index
        self.chunk_ids = list(rag.get_experiment_chunks(index))
        self.config = {"test": True}

    def search(self, question, k=5):
        return [{"chunk_id": rag.chunk_id(chunk), **chunk, "score": float(5 - i), "rank": i + 1}
                for i, chunk in enumerate(self.index["chunks"][:k])]


class ExperimentTests(unittest.TestCase):
    def test_method_ingest_indexes_share_chunks_without_bge_in_bm25(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            docs = folder / "docs"
            docs.mkdir()
            for chunk in example_index()["chunks"]:
                (docs / chunk["source"]).write_text(chunk["text"], encoding="utf-8")
            dense_path, sparse_path = folder / "dense.json", folder / "bm25.json"
            with (patch.object(rag, "load_config"), patch.object(rag, "Encoder") as encoder,
                  patch("sys.stdout", new_callable=io.StringIO)):
                encoder.return_value.encode.return_value = example_index()["vectors"]
                self.assertEqual(bge.main(["ingest", "--docs", str(docs), "--index", str(dense_path)]), 0)
            with (patch.object(rag, "Encoder", side_effect=AssertionError("BGE loaded")),
                  patch.object(rag, "load_config", side_effect=AssertionError("configuration loaded")),
                  patch.object(rag, "generate", side_effect=AssertionError("API called")),
                  patch("sys.stdout", new_callable=io.StringIO)):
                self.assertEqual(bm25.main(["ingest", "--docs", str(docs), "--index", str(sparse_path)]), 0)
                sparse = rag.load_corpus_index(sparse_path)
                self.assertNotIn("vectors", sparse)
                self.assertEqual(len(sparse["tokenized_corpus"]), 5)
                self.assertEqual(BM25Retrieval(sparse).search("水冰")[0]["source"], "0.txt")
            dense = rag.load_index(dense_path)
            self.assertEqual(rag.corpus_fingerprint(dense), rag.corpus_fingerprint(sparse))
            sparse["chunks"][0]["text"] = "被修改的文本"
            rag.save_json(sparse_path, sparse)
            with self.assertRaises(ValueError):
                rag.load_corpus_index(sparse_path)

    def test_method_evaluation_keeps_default_command_and_never_generates(self):
        for module in (bge, bm25):
            with self.subTest(module=module.__name__):
                with (patch.object(module, "evaluate_method") as evaluate,
                      patch.object(rag, "generate", side_effect=AssertionError("API called"))):
                    self.assertEqual(module.main([]), 0)
                    self.assertEqual(module.main(["evaluate", "--validate-only"]), 0)
                    self.assertEqual(evaluate.call_count, 2)
                    self.assertEqual(evaluate.call_args.args[2].command, "evaluate")

    def test_empty_retrieval_answer_avoids_api_and_optional_history(self):
        retriever = FakeRetriever(example_index())
        with (patch.dict("os.environ", {}, clear=True),
              patch.object(rag, "save_query_history", side_effect=AssertionError("history written"))):
            response = rag.run_query("问题", retriever, min_score=100, generate_answer=True)
        self.assertEqual(response.hits, [])
        self.assertIn("无法回答", response.answer)

    def test_bm25_search_does_not_load_generation_configuration(self):
        with (patch.object(rag, "load_corpus_index", return_value=example_index()),
              patch.object(rag, "load_config", side_effect=AssertionError("configuration loaded")),
              patch.object(rag, "Encoder", side_effect=AssertionError("BGE loaded")),
              patch.object(rag, "generate", side_effect=AssertionError("API called")),
              patch.object(rag, "save_query_history") as history,
              patch("sys.stdout", new_callable=io.StringIO)):
            self.assertEqual(bm25.main(["search", "水冰"]), 0)
            self.assertEqual(history.call_args.kwargs["method"], "BM25")

    def test_public_library_has_no_cli_or_import_side_effects(self):
        code = ("import sys; sys.argv = ['rag.py', 'not-a-command']; import rag; "
                "assert not hasattr(rag, 'main'); assert not hasattr(rag, 'run_method_cli'); "
                "assert not hasattr(rag, 'argparse'); assert 'sentence_transformers' not in sys.modules")
        for command in ([sys.executable, "-B", "-c", code],
                        [sys.executable, "-B", str(rag.ROOT / "rag.py"), "--help"]):
            with self.subTest(command=command):
                result = subprocess.run(command, cwd=rag.ROOT, capture_output=True, text=True,
                                        encoding="utf-8", timeout=20)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, "")
                self.assertEqual(result.stderr, "")

    def test_public_runner_accepts_data_and_protects_existing_results(self):
        index = example_index()
        factory = Mock(side_effect=lambda shared_index, device: FakeRetriever(shared_index))
        with tempfile.TemporaryDirectory() as directory:
            with patch("sys.stderr", new_callable=io.StringIO):
                report, result_dir = rag.run_retrieval_experiment(
                    "test", factory, index, example_dataset(index), {},
                    device="cpu", output_dir=Path(directory), run_id="library-api")
            factory.assert_called_once_with(index, "cpu")
            self.assertEqual(report["summary"]["Hit@3"], 1)
            self.assertEqual(result_dir, Path(directory))
            self.assertTrue((result_dir / "details.json").is_file())
            self.assertTrue((result_dir / "report.html").is_file())
            self.assertFalse((result_dir / "exports").exists())
            with self.assertRaises(ValueError):
                rag.run_retrieval_experiment("test", factory, index, example_dataset(index), {},
                                             output_dir=Path(directory), run_id="library-api")
            factory.assert_called_once()

    def test_stable_ids_and_fingerprint_detect_content_changes(self):
        index = example_index()
        self.assertEqual(rag.chunk_id(index["chunks"][0]), "0.txt::pnone::c0001")
        changed = copy.deepcopy(index)
        changed["chunks"][0]["text"] += "新文本"
        self.assertEqual(rag.chunk_id(index["chunks"][0]), rag.chunk_id(changed["chunks"][0]))
        self.assertNotEqual(rag.corpus_fingerprint(index), rag.corpus_fingerprint(changed))
        changed["chunks"][1] = changed["chunks"][0]
        with self.assertRaises(ValueError):
            rag.get_experiment_chunks(changed)

    def test_recall_and_hit_are_distinct_with_multiple_relevant_chunks(self):
        metrics = rag.retrieval_metrics(["a", "x", "b", "z"], ["a", "b"])
        self.assertEqual(metrics["Recall@1"], .5)
        self.assertEqual(metrics["Hit@1"], 1)
        self.assertEqual(metrics["Recall@3"], 1)
        self.assertEqual(rag.retrieval_metrics(["x"], ["a"])["Hit@3"], 0)
        with self.assertRaises(ValueError):
            rag.retrieval_metrics(["a", "a"], ["a"])
        with self.assertRaises(ValueError):
            rag.retrieval_metrics([], [])

    def test_testset_validation_rejects_invalid_annotations(self):
        index = example_index()
        base = example_dataset(index)
        variants = []
        for key, value in (("question", ""), ("answer", ""), ("relevant_chunk_ids", []),
                           ("relevant_chunk_ids", ["missing"]), ("review_status", "automatic")):
            changed = copy.deepcopy(base)
            changed["queries"][0][key] = value
            variants.append(changed)
        changed = copy.deepcopy(base)
        changed["queries"].append(copy.deepcopy(changed["queries"][0]))
        variants.append(changed)
        changed = copy.deepcopy(base)
        changed["queries"][0]["evidence"][0]["quote"] = "这句话不在原文中"
        variants.append(changed)
        changed = copy.deepcopy(base)
        changed["corpus_fingerprint"] = "different"
        variants.append(changed)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "queries.json"
            for number, dataset in enumerate(variants):
                with self.subTest(number=number):
                    rag.write_experiment_json(path, dataset)
                    with self.assertRaises(ValueError):
                        rag.load_test_queries(path, index)

    def test_formal_mode_requires_explicit_human_review(self):
        index = example_index()
        dataset = example_dataset(index)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "queries.json"
            rag.write_experiment_json(path, dataset)
            self.assertEqual(len(rag.load_test_queries(path, index)["queries"]), 1)
            with self.assertRaises(ValueError):
                rag.load_test_queries(path, index, require_reviewed=True)
            dataset["queries"][0]["review_status"] = "reviewed"
            dataset["queries"][0]["reviewed_by"] = None
            rag.write_experiment_json(path, dataset)
            with self.assertRaises(ValueError):
                rag.load_test_queries(path, index)
            dataset["queries"][0]["reviewed_by"] = "人工审核者"
            rag.write_experiment_json(path, dataset)
            rag.load_test_queries(path, index, require_reviewed=True)

    def test_evaluation_does_not_call_generation_or_history(self):
        index = example_index()
        with (patch.object(rag, "generate", side_effect=AssertionError("API called")),
              patch.object(rag, "save_query_history", side_effect=AssertionError("history written")),
              patch("sys.stderr", new_callable=io.StringIO)):
            report = rag.evaluate_retrieval("test", FakeRetriever(index), index, example_dataset(index), {})
        self.assertEqual(report["summary"]["Hit@3"], 1)

    def test_evaluation_rejects_different_corpus_or_result_text(self):
        index = example_index()
        retriever = FakeRetriever(index)
        retriever.chunk_ids.reverse()
        with self.assertRaises(ValueError):
            rag.evaluate_retrieval("test", retriever, index, example_dataset(index), {})
        retriever = FakeRetriever(index)
        results = retriever.search("question")
        results[0]["text"] = "different"
        retriever.search = Mock(return_value=results)
        with self.assertRaises(ValueError):
            rag.evaluate_retrieval("test", retriever, index, example_dataset(index), {})

    def test_bm25_runs_without_loading_encoder_and_ties_use_index_order(self):
        index = example_index()
        with (patch.object(rag, "Encoder", side_effect=AssertionError("BGE loaded")),
              patch.object(rag, "generate", side_effect=AssertionError("API called")),
              patch.object(rag, "load_config", side_effect=AssertionError("generation config loaded"))):
            retriever = BM25Retrieval(index)
            self.assertEqual(retriever.search("水冰")[0]["source"], "0.txt")
            zero_scores = retriever.search("zxqvnonexistent")
            self.assertEqual([r["chunk_id"] for r in zero_scores], retriever.chunk_ids)
            self.assertTrue(all(r["score"] == 0 for r in zero_scores))

    def test_bge_reuses_query_encoding_and_existing_vector_search(self):
        index = example_index()
        with (patch.object(rag, "load_config"), patch.object(rag, "Encoder") as encoder,
              patch.object(rag, "retrieve", wraps=rag.retrieve) as retrieve,
              patch.object(rag, "generate", side_effect=AssertionError("API called"))):
            encoder.return_value.model.device = "cpu"
            encoder.return_value.encode.return_value = [[1., 0., 0., 0., 0.]]
            retriever = BGEDenseRetrieval(index, "cpu")
            result = retriever.search("问题", k=3)
            encoder.return_value.encode.assert_called_once_with(["问题"], query=True)
            self.assertIs(retrieve.call_args.args[0], index)
            self.assertEqual(len(result), 3)
            self.assertEqual(result[0]["source"], "0.txt")

    def test_exports_keep_all_failures_and_prioritize_other_method_success(self):
        index = example_index()
        with patch("sys.stderr", new_callable=io.StringIO):
            report = rag.evaluate_retrieval("test", FakeRetriever(index), index, example_dataset(index), {})
        base_query = report["queries"][0]
        report["queries"] = []
        for i in range(4):
            query = copy.deepcopy(base_query)
            query["query_id"] = f"q{i}"
            query["metrics"]["Hit@3"] = 0
            report["queries"].append(query)
        report["metadata"]["method"] = "BM25"
        other = copy.deepcopy(report)
        other["metadata"]["method"] = "BGE"
        other["queries"][3]["metrics"]["Hit@3"] = 1
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            rag.save_experiment_results({"BM25": report, "BGE": other}, folder, export_csv=True)
            with (folder / "exports/BM25_failures.csv").open(encoding="utf-8-sig", newline="") as stream:
                failures = list(csv.DictReader(stream))
            with (folder / "exports/BM25_failure_cases.csv").open(encoding="utf-8-sig", newline="") as stream:
                cases = list(csv.DictReader(stream))
            self.assertEqual(len(failures), 4)
            self.assertEqual(len(cases), 3)
            self.assertEqual(cases[0]["query_id"], "q3")
            self.assertTrue(all(r["failure_reason"] == r["manual_notes"] == "" for r in failures))
            self.assertIn("月球 极区 水冰", (folder / "details.json").read_text(encoding="utf-8"))

    def test_comparison_rejects_unpaired_runs(self):
        first = {"metadata": {"corpus_fingerprint": "one"}}
        second = {"metadata": {"corpus_fingerprint": "two"}}
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                rag.compare_experiment_results(first, second, Path(directory))
            self.assertEqual(list(Path(directory).iterdir()), [])

    def report_fixture(self, run_id="first"):
        index = example_index()
        with patch("sys.stderr", new_callable=io.StringIO):
            return rag.evaluate_retrieval("BGE", FakeRetriever(index), index, example_dataset(index),
                                          {"run_id": run_id, "chunk_count": 5})

    def test_default_report_embeds_charts_and_escapes_question_and_source(self):
        report = self.report_fixture()
        report["queries"][0]["question"] = '<script>alert("question")</script>'
        report["queries"][0]["retrieved"][0]["text"] = '<img src=x onerror="alert(1)">'
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            path = rag.save_experiment_results({"BGE": report}, folder)
            html = path.read_text(encoding="utf-8")
            self.assertEqual({p.name for p in folder.iterdir()}, {"report.html", "details.json"})
            self.assertIn("BGE 检索评估报告", html)
            self.assertEqual(html.count("data:image/png;base64,"), 2)
            self.assertIn("&lt;script&gt;", html)
            self.assertIn("&lt;img src=x", html)
            self.assertNotIn('<img src=x', html)
            self.assertNotIn('<script>alert', html)
            self.assertIn("月球 极区 水冰", html)
            self.assertEqual(json.loads((folder / "details.json").read_text(encoding="utf-8"))
                             ["reports"]["BGE"]["queries"][0]["question"], report["queries"][0]["question"])

    def test_new_report_archives_previous_outputs_and_manual_csv_notes(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(rag, "metric_chart_images", return_value={"Recall": "", "Hit": ""}):
            folder = Path(directory)
            rag.save_experiment_results({"BGE": self.report_fixture()}, folder, export_csv=True)
            old_html = (folder / "report.html").read_bytes()
            notes = folder / "exports/BGE_failure_cases.csv"
            notes.write_text("人工分析备注\n", encoding="utf-8-sig")
            rag.save_experiment_results({"BGE": self.report_fixture("second")}, folder)
            self.assertEqual({p.name for p in folder.iterdir()}, {"report.html", "details.json", "history"})
            archive = folder / "history/first"
            self.assertEqual((archive / "report.html").read_bytes(), old_html)
            self.assertEqual((archive / "exports/BGE_failure_cases.csv").read_text(encoding="utf-8-sig"), "人工分析备注\n")
            self.assertEqual(json.loads((folder / "details.json").read_text(encoding="utf-8"))["run_id"], "second")

    def test_failed_report_write_keeps_previous_outputs_and_cleans_staging(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(rag, "metric_chart_images", return_value={"Recall": "", "Hit": ""}):
            folder = Path(directory)
            rag.save_experiment_results({"BGE": self.report_fixture()}, folder)
            before = {path.name: path.read_bytes() for path in folder.iterdir()}
            with patch.object(rag, "write_experiment_json", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    rag.save_experiment_results({"BGE": self.report_fixture("second")}, folder)
            self.assertEqual({path.name: path.read_bytes() for path in folder.iterdir()}, before)

    def test_failed_publication_restores_archived_report_and_exports(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(rag, "metric_chart_images", return_value={"Recall": "", "Hit": ""}):
            folder = Path(directory)
            rag.save_experiment_results({"BGE": self.report_fixture()}, folder, export_csv=True)
            before = {str(path.relative_to(folder)): path.read_bytes() for path in folder.rglob("*") if path.is_file()}
            original_replace = Path.replace
            installed_count = 0

            def fail_second_install(source, target):
                nonlocal installed_count
                if source.parent.name.startswith(".pending-") and Path(target).parent == folder.resolve():
                    installed_count += 1
                    if installed_count == 2:
                        raise OSError("publication interrupted")
                return original_replace(source, target)

            with patch.object(Path, "replace", fail_second_install):
                with self.assertRaises(OSError):
                    rag.save_experiment_results({"BGE": self.report_fixture("second")}, folder)
            self.assertEqual({str(path.relative_to(folder)): path.read_bytes() for path in folder.rglob("*") if path.is_file()}, before)
            self.assertFalse((folder / ".publish.lock").exists())
            self.assertFalse(list(folder.glob(".pending-*")))

    def test_mismatched_complete_reports_do_not_replace_current_report(self):
        report = self.report_fixture()
        other = copy.deepcopy(report)
        other["metadata"]["method"] = "BM25"
        other["queries"][0]["relevant_chunk_ids"] = ["different"]
        with tempfile.TemporaryDirectory() as directory, patch.object(rag, "metric_chart_images", return_value={"Recall": "", "Hit": ""}):
            folder = Path(directory)
            rag.save_experiment_results({"BGE": report}, folder)
            before = {p.name: p.read_bytes() for p in folder.iterdir()}
            with self.assertRaises(ValueError):
                rag.save_experiment_results({"BGE": report, "BM25": other}, folder)
            self.assertEqual({p.name: p.read_bytes() for p in folder.iterdir()}, before)

    def test_dataset_covers_all_domains_and_chunks(self):
        index, dataset, metadata = rag.load_experiment_data(rag.ROOT / "storage/index.json",
                                                           rag.ROOT / "data/test_queries.json")
        self.assertEqual(metadata["query_count"], 48)
        for domain in ("astronomy", "environment", "computing"):
            self.assertEqual(sum(q["domain"] == domain for q in dataset["queries"]), 16)
        labeled = {cid for q in dataset["queries"] for cid in q["relevant_chunk_ids"]}
        self.assertEqual(labeled, set(rag.get_experiment_chunks(index)))


if __name__ == "__main__":
    unittest.main()
