#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT_DIR/venv/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "Errore: python venv non trovato in $PY"
  exit 1
fi

cd "$ROOT_DIR"

echo "[1/6] Unit tests"
"$PY" -m unittest discover -s tests -v

echo "[2/6] Compile check"
"$PY" -m py_compile src/agent.py src/ingest.py src/retrieval.py src/eval.py src/utils.py scripts/offline_benchmark.py scripts/stress_retrieval.py scripts/smoke_check.py scripts/calibrate_confidence.py scripts/hard_negative_eval.py

echo "[3/6] Smoke check end-to-end offline"
"$PY" scripts/smoke_check.py

echo "[4/6] Offline benchmark quick run"
EMBEDDING_MODE=mock BM25_CORPUS_PATH=data/dev/bm25_corpus.jsonl "$PY" scripts/offline_benchmark.py --cases data/dev/benchmark_cases.json --min_confidence 0.6 --out data/dev/benchmark_results.json

echo "[5/6] Hard negative eval"
EMBEDDING_MODE=mock BM25_CORPUS_PATH=data/dev/hard_negative_corpus.jsonl "$PY" scripts/hard_negative_eval.py --cases data/dev/hard_negative_cases.json --min_confidence 0.6

echo "[6/6] Confidence calibration snapshot"
EMBEDDING_MODE=mock BM25_CORPUS_PATH=data/dev/bm25_corpus.jsonl "$PY" scripts/calibrate_confidence.py --cases data/dev/benchmark_cases.json --corpus data/dev/bm25_corpus.jsonl --start 0.5 --stop 0.8 --step 0.1

echo "PRE_HACKATHON_CHECK_OK"
