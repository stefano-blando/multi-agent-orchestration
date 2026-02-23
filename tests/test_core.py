import unittest

from src.agent import (
    _build_constraint_profile,
    _evaluate_constraint_with_evidence,
    _search_queries,
    _summarize_batch_results,
    build_agent,
)
from src.eval import jaccard
from src.ingest import chunk_text
from src.retrieval import reciprocal_rank_fusion


class TestCoreLogic(unittest.TestCase):
    def test_chunk_text_with_overlap(self):
        text = " ".join(f"w{i}" for i in range(12))
        chunks = chunk_text(text, chunk_size=5, overlap=2)
        self.assertEqual(chunks[0], "w0 w1 w2 w3 w4")
        self.assertEqual(chunks[1], "w3 w4 w5 w6 w7")
        self.assertTrue(len(chunks) >= 3)

    def test_jaccard(self):
        self.assertAlmostEqual(jaccard(["a", "b"], ["b", "c"]), 1 / 3)
        self.assertEqual(jaccard([], []), 1.0)
        self.assertEqual(jaccard(["a"], []), 0.0)

    def test_rrf(self):
        fused = reciprocal_rank_fusion([[1, 2, 3], [2, 3, 4]], k=60)
        ranked_ids = [idx for idx, _ in fused]
        self.assertEqual(ranked_ids[0], 2)
        self.assertIn(1, ranked_ids)
        self.assertIn(4, ranked_ids)

    def test_constraint_evaluator(self):
        rows = [
            {"text": "Pizza X contiene glutine e latte.", "source": "menu.pdf", "index": 0, "score": 0.8},
            {"text": "Per celiaci evitare Pizza X.", "source": "note.pdf", "index": 2, "score": 0.6},
        ]
        status, _, _, _ = _evaluate_constraint_with_evidence(
            item="Pizza X",
            constraint="senza glutine",
            evidence_rows=rows,
        )
        self.assertEqual(status, "NON CONFORME")

        rows_ok = [
            {
                "text": "Insalata Y e' dichiarata senza glutine e senza tracce di wheat.",
                "source": "menu.pdf",
                "index": 1,
                "score": 0.9,
            }
        ]
        status, _, _, _ = _evaluate_constraint_with_evidence(
            item="Insalata Y",
            constraint="senza glutine",
            evidence_rows=rows_ok,
        )
        self.assertEqual(status, "CONFORME")

    def test_constraint_profile_and_queries(self):
        profile = _build_constraint_profile("senza lattosio")
        self.assertEqual(profile.mode, "absence")
        queries = _search_queries("Gelato Luna", profile)
        self.assertGreaterEqual(len(queries), 2)
        self.assertIn("Gelato Luna senza lattosio", queries)

    def test_batch_summary(self):
        results = [
            {
                "item": "Insalata Y",
                "constraint": "senza glutine",
                "status": "CONFORME",
                "confidence": 0.9,
                "reason": "ok",
            },
            {
                "item": "Pizza X",
                "constraint": "senza glutine",
                "status": "NON CONFORME",
                "confidence": 0.8,
                "reason": "contiene glutine",
            },
        ]
        summary = _summarize_batch_results(results, min_confidence=0.6)
        self.assertIn("Insalata Y", summary["safe_items"])
        self.assertNotIn("Pizza X", summary["safe_items"])
        self.assertEqual(summary["total_checks"], 2)

    def test_orchestrator_has_validator_tool(self):
        agent = build_agent()
        tool_names = [tool.name for tool in agent._tools]
        self.assertIn("validator", tool_names)


if __name__ == "__main__":
    unittest.main()
