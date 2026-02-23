#!/usr/bin/env python3
"""
Smoke check locale pre-hackathon (senza API key).
Valida: ingest -> retrieval -> validator batch -> run_with_trace.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def write_sample_raw_data(data_dir: Path):
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "menu_galattico.txt").write_text(
        "Pizza Nebulosa contiene glutine. Insalata Orbitale senza glutine e vegana. Chips Solari senza glutine.",
        encoding="utf-8",
    )
    (data_dir / "menu_alieno.txt").write_text(
        "Zuppa Cosmica contiene latticini. Tofu Meteorico vegano e senza glutine.",
        encoding="utf-8",
    )


def main():
    os.environ["EMBEDDING_MODE"] = "mock"
    os.environ["EMBEDDING_MOCK_DIM"] = "64"
    os.environ["VALIDATOR_TOP_K"] = "4"
    os.environ["VALIDATOR_EVIDENCE_LIMIT"] = "8"

    with tempfile.TemporaryDirectory(prefix="hackapizza_smoke_") as tmpdir:
        tmp_path = Path(tmpdir)
        raw_dir = tmp_path / "raw"
        corpus_path = tmp_path / "bm25_corpus.jsonl"
        os.environ["BM25_CORPUS_PATH"] = str(corpus_path)

        from src.agent import check_constraints_batch, run_with_trace
        from src.ingest import ingest
        from src.retrieval import bm25_search, reset_runtime_state

        write_sample_raw_data(raw_dir)
        ingest(data_dir=str(raw_dir), recreate=True, batch_size=32)

        reset_runtime_state()
        retr = bm25_search("senza glutine", top_k=3)
        if not retr:
            raise RuntimeError("Smoke fail: bm25_search non ha restituito risultati.")

        batch_out = check_constraints_batch(
            items=["Pizza Nebulosa", "Insalata Orbitale", "Tofu Meteorico"],
            constraints=["senza glutine"],
            min_confidence=0.6,
        )
        batch_obj = json.loads(batch_out)
        safe_items = batch_obj.get("summary", {}).get("safe_items", [])
        if "Insalata Orbitale" not in safe_items:
            raise RuntimeError("Smoke fail: validator batch non ha marcato Insalata Orbitale come safe.")

        trace = run_with_trace("Quali piatti sono senza glutine?")
        expected_trace_keys = {"answer", "tools_used", "usage", "raw_text"}
        if not expected_trace_keys.issubset(set(trace.keys())):
            raise RuntimeError("Smoke fail: run_with_trace non ha il contratto atteso.")

    print("SMOKE_OK")
    print("- ingest: OK")
    print("- retrieval: OK")
    print("- validator_batch: OK")
    print("- run_with_trace: OK")


if __name__ == "__main__":
    main()
