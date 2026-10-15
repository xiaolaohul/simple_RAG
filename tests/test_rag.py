import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import rag


class RagTests(unittest.TestCase):
    def test_split_covers_text_and_keeps_overlap(self):
        text = "甲乙丙丁戊己庚辛壬癸" * 12
        chunks = rag.split_text(text, 35, 6)
        self.assertTrue(all(0 < len(c) <= 35 for c in chunks))
        rebuilt = chunks[0] + "".join(c[6:] for c in chunks[1:])
        self.assertEqual(rebuilt, text)
        self.assertEqual(rag.split_text("   \n"), [])
        with self.assertRaises(ValueError):
            rag.split_text(text, 10, 10)

    def test_short_boundary_does_not_cause_loop(self):
        chunks = rag.split_text("甲。乙。丙。丁。戊。己。庚。辛。", 8, 7)
        self.assertLess(len(chunks), 25)
        self.assertTrue(all(len(c) <= 8 for c in chunks))

    def test_document_reading_and_index_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "资料.md").write_text("\ufeff这是文档内容。", encoding="utf-8")
            (root / "编码.txt").write_bytes("中文旧编码。".encode("gb18030"))
            (root / "ignored.csv").write_text("ignore", encoding="utf-8")
            chunks = rag.read_documents(root, 350, 60)
            self.assertEqual(len(chunks), 2)
            self.assertNotIn("\ufeff", "".join(c.text for c in chunks))
            path = root / "index.json"
            rag.save_index(path, chunks, [[1., 0.], [0., 1.]], "test", root, 350, 60)
            index = rag.load_index(path)
            self.assertEqual(index["embedding_model"], "test")
            self.assertEqual(index["chunks"][0]["source"], chunks[0].source)

    def test_retrieval_ranking_threshold_and_dimension(self):
        index = {"vectors": [[0., 1.], [1., 0.], [math.sqrt(.5), math.sqrt(.5)]],
                 "chunks": [dict(source=f"{i}.txt", page=None, number=1, text=str(i)) for i in range(3)]}
        hits = rag.retrieve(index, [1., 0.], 2)
        self.assertEqual([h.chunk.source for h in hits], ["1.txt", "2.txt"])
        self.assertEqual(len(rag.retrieve(index, [1., 0.], 3, .9)), 1)
        self.assertEqual(rag.retrieve(index, [-1., 0.], 3, .5), [])
        with self.assertRaises(ValueError):
            rag.retrieve(index, [1.], 3)

    def test_empty_evidence_does_not_call_api(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertIn("无法回答", rag.generate("问题", []))

    def test_prompt_contains_traceable_pdf_page(self):
        messages = rag.make_messages("问题", [rag.Hit(rag.Chunk("paper.pdf", 3, 2, "原文"), .8)])
        evidence = json.loads(messages[1]["content"])["evidence"][0]
        self.assertEqual((evidence["id"], evidence["page"], evidence["text"]), (1, 3, "原文"))

    def test_corrupt_index_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.json"
            path.write_text(json.dumps({"version": 1, "vectors": [[1], [1, 2]], "chunks": [{}, {}]}))
            with self.assertRaises(ValueError):
                rag.load_index(path)


if __name__ == "__main__":
    unittest.main()
