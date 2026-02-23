import json
import os
import unittest


class TestHardNegatives(unittest.TestCase):
    def setUp(self):
        self._old_env = {
            "BM25_CORPUS_PATH": os.getenv("BM25_CORPUS_PATH"),
            "EMBEDDING_MODE": os.getenv("EMBEDDING_MODE"),
            "VALIDATOR_TOP_K": os.getenv("VALIDATOR_TOP_K"),
            "VALIDATOR_EVIDENCE_LIMIT": os.getenv("VALIDATOR_EVIDENCE_LIMIT"),
        }
        os.environ["EMBEDDING_MODE"] = "mock"
        os.environ["BM25_CORPUS_PATH"] = "data/dev/hard_negative_corpus.jsonl"
        os.environ["VALIDATOR_TOP_K"] = "4"
        os.environ["VALIDATOR_EVIDENCE_LIMIT"] = "8"

        from src.retrieval import reset_runtime_state

        reset_runtime_state()

    def tearDown(self):
        for key, val in self._old_env.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val
        from src.retrieval import reset_runtime_state

        reset_runtime_state()

    def test_no_false_positives_on_negative_cases(self):
        from src.agent import check_constraints_batch

        with open("data/dev/hard_negative_cases.json", encoding="utf-8") as f:
            cases = json.load(f)

        for case in cases:
            out = check_constraints_batch(
                items=case.get("candidates", []),
                constraints=case.get("constraints", []),
                min_confidence=0.6,
            )
            obj = json.loads(out)
            pred = sorted(obj.get("summary", {}).get("safe_items", []))
            exp = sorted(case.get("expected_safe", []))
            self.assertEqual(pred, exp, msg=f"Mismatch in {case.get('id')}: pred={pred} exp={exp}")


if __name__ == "__main__":
    unittest.main()
