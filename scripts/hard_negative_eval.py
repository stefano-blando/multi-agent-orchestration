#!/usr/bin/env python3
"""
Valuta robustezza su hard negatives (false positive control).
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="data/dev/hard_negative_cases.json")
    parser.add_argument("--corpus", default="data/dev/hard_negative_corpus.jsonl")
    parser.add_argument("--min_confidence", type=float, default=0.6)
    args = parser.parse_args()

    os.environ.setdefault("EMBEDDING_MODE", "mock")
    os.environ["BM25_CORPUS_PATH"] = args.corpus

    from src.agent import check_constraints_batch
    from src.retrieval import reset_runtime_state

    with open(args.cases, encoding="utf-8") as f:
        cases = json.load(f)

    reset_runtime_state()
    false_positives = 0
    total_negative_cases = 0
    correct = 0

    for case in cases:
        out = check_constraints_batch(
            items=case.get("candidates", []),
            constraints=case.get("constraints", []),
            min_confidence=args.min_confidence,
        )
        obj = json.loads(out)
        pred = set(obj.get("summary", {}).get("safe_items", []))
        exp = set(case.get("expected_safe", []))
        if not exp:
            total_negative_cases += 1
            if pred:
                false_positives += 1
        if pred == exp:
            correct += 1

        print(f"{case['id']}: pred={sorted(pred)} expected={sorted(exp)}")

    total = len(cases) or 1
    fp_rate = false_positives / max(1, total_negative_cases)
    acc = correct / total

    print("\nHard-negative eval")
    print(f"- accuracy: {acc:.4f}")
    print(f"- false_positive_rate_on_negative_cases: {fp_rate:.4f}")


if __name__ == "__main__":
    main()
