"""
Evaluation locale con Jaccard Similarity.
Usare questo script ad ogni iterazione per misurare il miglioramento.

Uso:
    python src/eval.py --predictions predictions.json --ground_truth gt.json
"""

import json
import argparse
from pathlib import Path


def jaccard(pred: list, gt: list) -> float:
    """Jaccard similarity tra due liste (trattate come insiemi)."""
    p, g = set(pred), set(gt)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    return len(p & g) / len(p | g)


def evaluate(predictions: dict, ground_truth: dict) -> dict:
    """
    Calcola Jaccard per ogni query e ritorna score medio.

    Args:
        predictions: {query_id: [item1, item2, ...]}
        ground_truth: {query_id: [item1, item2, ...]}
    """
    scores = {}
    for qid, gt_items in ground_truth.items():
        pred_items = predictions.get(qid, [])
        scores[qid] = jaccard(pred_items, gt_items)

    avg = sum(scores.values()) / len(scores) if scores else 0.0
    return {"per_query": scores, "average": avg}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--ground_truth", required=True)
    args = parser.parse_args()

    with open(args.predictions) as f:
        predictions = json.load(f)
    with open(args.ground_truth) as f:
        ground_truth = json.load(f)

    results = evaluate(predictions, ground_truth)
    print(f"\nAverage Jaccard Score: {results['average']:.4f}")
    print("\nPer-query scores:")
    for qid, score in sorted(results["per_query"].items()):
        print(f"  {qid}: {score:.4f}")


if __name__ == "__main__":
    main()
