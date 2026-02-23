#!/usr/bin/env python3
"""
Calibra la soglia min_confidence su un benchmark locale.
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def jaccard(pred: list[str], gt: list[str]) -> float:
    p, g = set(pred), set(gt)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    return len(p & g) / len(p | g)


def precision(pred: list[str], gt: list[str]) -> float:
    p, g = set(pred), set(gt)
    if not p:
        return 0.0
    return len(p & g) / len(p)


def evaluate(cases: list[dict], threshold: float) -> dict:
    from src.agent import check_constraints_batch
    from src.retrieval import reset_runtime_state

    reset_runtime_state()
    rows = []
    for case in cases:
        out = check_constraints_batch(
            items=case.get("candidates", []),
            constraints=case.get("constraints", []),
            min_confidence=threshold,
        )
        obj = json.loads(out)
        pred = obj.get("summary", {}).get("safe_items", [])
        gt = case.get("expected", case.get("expected_safe", []))
        rows.append(
            {
                "id": case.get("id", "unknown"),
                "jaccard": jaccard(pred, gt),
                "precision": precision(pred, gt),
            }
        )
    n = len(rows) or 1
    return {
        "threshold": round(threshold, 3),
        "avg_jaccard": sum(r["jaccard"] for r in rows) / n,
        "avg_precision": sum(r["precision"] for r in rows) / n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="data/dev/benchmark_cases.json")
    parser.add_argument("--corpus", default="data/dev/bm25_corpus.jsonl")
    parser.add_argument("--start", type=float, default=0.45)
    parser.add_argument("--stop", type=float, default=0.9)
    parser.add_argument("--step", type=float, default=0.05)
    args = parser.parse_args()

    os.environ.setdefault("EMBEDDING_MODE", "mock")
    os.environ["BM25_CORPUS_PATH"] = args.corpus

    with open(args.cases, encoding="utf-8") as f:
        cases = json.load(f)

    grid = []
    cur = args.start
    while cur <= args.stop + 1e-9:
        grid.append(round(cur, 4))
        cur += args.step

    results = [evaluate(cases, t) for t in grid]
    results.sort(key=lambda r: (r["avg_jaccard"], r["avg_precision"]), reverse=True)
    best = results[0] if results else {}

    print("Confidence calibration")
    print(f"- thresholds: {len(results)}")
    print(f"- best: {best}")
    print("\nTop 5:")
    for row in results[:5]:
        print(
            f"thr={row['threshold']:.2f} "
            f"J={row['avg_jaccard']:.4f} "
            f"P={row['avg_precision']:.4f}"
        )


if __name__ == "__main__":
    main()
