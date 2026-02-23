import json
import os
import tempfile
import unittest
from pathlib import Path


class TestContracts(unittest.TestCase):
    def setUp(self):
        self._old_env = {
            "BM25_CORPUS_PATH": os.getenv("BM25_CORPUS_PATH"),
            "EMBEDDING_MODE": os.getenv("EMBEDDING_MODE"),
            "EMBEDDING_MOCK_DIM": os.getenv("EMBEDDING_MOCK_DIM"),
            "VALIDATOR_TOP_K": os.getenv("VALIDATOR_TOP_K"),
            "VALIDATOR_EVIDENCE_LIMIT": os.getenv("VALIDATOR_EVIDENCE_LIMIT"),
            "TRACE_ENABLED": os.getenv("TRACE_ENABLED"),
            "TRACE_LOG_PATH": os.getenv("TRACE_LOG_PATH"),
        }
        self.tmpdir = tempfile.TemporaryDirectory(prefix="contracts_")
        corpus_path = Path(self.tmpdir.name) / "bm25_corpus.jsonl"
        corpus_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "text": "Pizza Nebulosa contiene glutine.",
                            "source": "menu.txt",
                            "chunk_index": 0,
                        },
                        ensure_ascii=False,
                    ),
                    json.dumps(
                        {
                            "text": "Insalata Orbitale senza glutine e vegana.",
                            "source": "menu.txt",
                            "chunk_index": 1,
                        },
                        ensure_ascii=False,
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        os.environ["BM25_CORPUS_PATH"] = str(corpus_path)
        os.environ["EMBEDDING_MODE"] = "mock"
        os.environ["EMBEDDING_MOCK_DIM"] = "64"
        os.environ["VALIDATOR_TOP_K"] = "4"
        os.environ["VALIDATOR_EVIDENCE_LIMIT"] = "8"
        os.environ["TRACE_ENABLED"] = "1"
        os.environ["TRACE_LOG_PATH"] = str(Path(self.tmpdir.name) / "trace.jsonl")

        from src.retrieval import reset_runtime_state

        reset_runtime_state()

    def tearDown(self):
        self.tmpdir.cleanup()
        for key, val in self._old_env.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val

        from src.retrieval import reset_runtime_state

        reset_runtime_state()

    def test_check_constraint_contract(self):
        from src.agent import check_constraint

        out = check_constraint(item="Insalata Orbitale", constraint="senza glutine")
        obj = json.loads(out)
        required = {"item", "constraint", "status", "confidence", "reason", "queries", "evidence", "evidence_refs"}
        self.assertTrue(required.issubset(set(obj.keys())))
        self.assertIn(obj["status"], {"CONFORME", "NON CONFORME"})
        self.assertIsInstance(obj["confidence"], float)
        self.assertIsInstance(obj["queries"], list)
        self.assertIsInstance(obj["evidence"], list)
        if obj["evidence"]:
            self.assertIn("evidence_id", obj["evidence"][0])

    def test_check_constraints_batch_contract(self):
        from src.agent import check_constraints_batch

        out = check_constraints_batch(
            items=["Pizza Nebulosa", "Insalata Orbitale"],
            constraints=["senza glutine"],
            min_confidence=0.6,
        )
        obj = json.loads(out)
        self.assertIn("results", obj)
        self.assertIn("summary", obj)
        summary = obj["summary"]
        for key in ["min_confidence", "safe_items", "item_summary", "total_checks"]:
            self.assertIn(key, summary)
        self.assertIsInstance(summary["safe_items"], list)
        self.assertIsInstance(summary["item_summary"], list)
        self.assertEqual(summary["total_checks"], 2)

    def test_run_with_trace_contract(self):
        from src.agent import run_with_trace

        trace = run_with_trace("Quali piatti sono senza glutine?")
        required = {"request_id", "query", "answer", "tools_used", "usage", "raw_text", "mode"}
        self.assertTrue(required.issubset(set(trace.keys())))
        self.assertIsInstance(trace["answer"], list)

    def test_structured_orchestration_contract(self):
        from src.agent import run_structured_orchestration

        result = run_structured_orchestration(
            {
                "query": "piatti senza glutine",
                "candidates": ["Pizza Nebulosa", "Insalata Orbitale"],
                "constraints": ["senza glutine"],
                "min_confidence": 0.6,
            }
        )
        required = {"request_id", "query", "answer", "constraints", "candidates", "batch"}
        self.assertTrue(required.issubset(set(result.keys())))
        self.assertIn("summary", result["batch"])
        self.assertIn("results", result["batch"])


if __name__ == "__main__":
    unittest.main()
