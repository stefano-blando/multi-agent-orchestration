#!/usr/bin/env python3
"""
Grid search offline su chunking/top_k per retrieval+validator.

Uso:
  ./venv/bin/python scripts/stress_retrieval.py
"""

import argparse
import json
import os
import sys
import tempfile
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


def build_corpus_jsonl(docs: list[dict], out_path: Path, chunk_size: int, overlap: int):
    from src.ingest import chunk_text

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for doc in docs:
            source = doc["source"]
            text = doc["text"]
            chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
            for idx, chunk in enumerate(chunks):
                row = {"text": chunk, "source": source, "chunk_index": idx}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def evaluate_cases(cases: list[dict], min_confidence: float) -> float:
    from src.agent import check_constraints_batch
    from src.retrieval import reset_runtime_state

    reset_runtime_state()
    scores = []
    for case in cases:
        out_json = check_constraints_batch(
            items=case.get("candidates", []),
            constraints=case.get("constraints", []),
            min_confidence=min_confidence,
        )
        obj = json.loads(out_json)
        pred = obj.get("summary", {}).get("safe_items", [])
        gt = case.get("expected", [])
        scores.append(jaccard(pred, gt))
    return sum(scores) / len(scores) if scores else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", default="data/dev/docs.json")
    parser.add_argument("--cases", default="data/dev/benchmark_cases.json")
    parser.add_argument("--min_confidence", type=float, default=0.6)
    parser.add_argument("--top_n", type=int, default=5)
    args = parser.parse_args()

    with open(args.docs, encoding="utf-8") as f:
        docs = json.load(f)
    with open(args.cases, encoding="utf-8") as f:
        cases = json.load(f)

    os.environ.setdefault("EMBEDDING_MODE", "mock")

    chunk_sizes = [80, 120, 180]
    overlaps = [20, 40, 60]
    top_ks = [3, 4, 6]
    evidence_limits = [6, 8]

    results = []
    with tempfile.TemporaryDirectory(prefix="retrieval_stress_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        for chunk_size in chunk_sizes:
            for overlap in overlaps:
                if overlap >= chunk_size:
                    continue
                for top_k in top_ks:
                    for ev_limit in evidence_limits:
                        corpus_path = tmpdir_path / f"bm25_{chunk_size}_{overlap}_{top_k}_{ev_limit}.jsonl"
                        build_corpus_jsonl(
                            docs=docs,
                            out_path=corpus_path,
                            chunk_size=chunk_size,
                            overlap=overlap,
                        )
                        os.environ["BM25_CORPUS_PATH"] = str(corpus_path)
                        os.environ["VALIDATOR_TOP_K"] = str(top_k)
                        os.environ["VALIDATOR_EVIDENCE_LIMIT"] = str(ev_limit)
                        score = evaluate_cases(cases=cases, min_confidence=args.min_confidence)
                        results.append(
                            {
                                "chunk_size": chunk_size,
                                "overlap": overlap,
                                "validator_top_k": top_k,
                                "evidence_limit": ev_limit,
                                "avg_jaccard": round(score, 4),
                            }
                        )

    results.sort(key=lambda r: r["avg_jaccard"], reverse=True)
    best = results[0] if results else {}
    print("Stress retrieval completed.")
    print(f"Grid points: {len(results)}")
    print(f"Best: {best}")
    print("\nTop configs:")
    for row in results[: max(1, args.top_n)]:
        print(row)


if __name__ == "__main__":
    main()
