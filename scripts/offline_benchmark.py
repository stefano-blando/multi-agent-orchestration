#!/usr/bin/env python3
"""
Benchmark offline di validator/retrieval senza API key.

Uso:
  ./venv/bin/python scripts/offline_benchmark.py
  ./venv/bin/python scripts/offline_benchmark.py --cases data/dev/benchmark_cases.json --min_confidence 0.65
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


def recall(pred: list[str], gt: list[str]) -> float:
    p, g = set(pred), set(gt)
    if not g:
        return 0.0
    return len(p & g) / len(g)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="data/dev/benchmark_cases.json")
    parser.add_argument("--min_confidence", type=float, default=0.6)
    parser.add_argument("--out", default="data/dev/benchmark_results.json")
    args = parser.parse_args()

    os.environ.setdefault("EMBEDDING_MODE", "mock")
    os.environ.setdefault("BM25_CORPUS_PATH", "data/dev/bm25_corpus.jsonl")

    from src.agent import check_constraints_batch
    from src.retrieval import reset_runtime_state

    cases_path = Path(args.cases)
    if not cases_path.exists():
        raise FileNotFoundError(f"Cases file non trovato: {cases_path}")

    with cases_path.open(encoding="utf-8") as f:
        cases = json.load(f)

    reset_runtime_state()

    per_case = []
    for case in cases:
        cid = case.get("id", "unknown")
        candidates = case.get("candidates", [])
        constraints = case.get("constraints", [])
        expected = case.get("expected", [])

        batch_json = check_constraints_batch(
            items=candidates,
            constraints=constraints,
            min_confidence=args.min_confidence,
        )
        batch_obj = json.loads(batch_json)
        predicted = batch_obj.get("summary", {}).get("safe_items", [])

        row = {
            "id": cid,
            "query": case.get("query", ""),
            "predicted": predicted,
            "expected": expected,
            "jaccard": round(jaccard(predicted, expected), 4),
            "precision": round(precision(predicted, expected), 4),
            "recall": round(recall(predicted, expected), 4),
        }
        per_case.append(row)

    avg_j = sum(r["jaccard"] for r in per_case) / len(per_case) if per_case else 0.0
    avg_p = sum(r["precision"] for r in per_case) / len(per_case) if per_case else 0.0
    avg_r = sum(r["recall"] for r in per_case) / len(per_case) if per_case else 0.0

    print("Offline benchmark")
    print(f"- cases: {len(per_case)}")
    print(f"- avg_jaccard:  {avg_j:.4f}")
    print(f"- avg_precision:{avg_p:.4f}")
    print(f"- avg_recall:   {avg_r:.4f}")
    print("")
    for row in per_case:
        print(
            f"{row['id']}: J={row['jaccard']:.4f} P={row['precision']:.4f} R={row['recall']:.4f} "
            f"| pred={row['predicted']}"
        )

    out = {
        "config": {
            "cases": str(cases_path),
            "min_confidence": args.min_confidence,
            "embedding_mode": os.getenv("EMBEDDING_MODE"),
            "bm25_corpus_path": os.getenv("BM25_CORPUS_PATH"),
        },
        "metrics": {
            "avg_jaccard": round(avg_j, 4),
            "avg_precision": round(avg_p, 4),
            "avg_recall": round(avg_r, 4),
        },
        "per_case": per_case,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nRisultati salvati in: {out_path}")


if __name__ == "__main__":
    main()
