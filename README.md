# Hackapizza 2.0 — Agent Edition

Soluzione per Hackapizza 2.0, competizione di Agentic AI organizzata da DataPizza.

## Setup rapido

```bash
# 1. Clone & venv
git clone https://github.com/StefanoBlando/hackapizza-2026.git
cd hackapizza-2026
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Config
cp .env.example .env
# → Inserisci le chiavi API nel .env

# 3. Ingestion (prima run consigliata con recreate)
python src/ingest.py --data_dir data/raw/ --recreate

# 4. Test pipeline
python src/agent.py

# 5. Eval
python src/eval.py --predictions predictions.json --ground_truth gt.json

# 5b. Pre-check completo offline (senza API key)
bash scripts/pre_hackathon_check.sh

# 6. Demo
streamlit run src/app.py

# 6b. Demo chat-first (Chainlit)
chainlit run src/app_chainlit.py
```

## Architettura

Sistema multi-agente basato su [datapizza-ai](https://github.com/datapizza-labs/datapizza-ai).
L'agente pianifica autonomamente, usa tool in sequenza e si auto-corregge fino a produrre una risposta soddisfacente.

```
Query
  └── OrchestratorAgent          # pianifica e coordina
        ├── RetrieverAgent       # cerca nei documenti (BM25 + semantic)
        ├── ValidatorAgent       # verifica vincoli (anche in batch)
        └── SynthesizerAgent     # produce la risposta finale
```

Ogni agente opera in un loop ReAct (Reason → Act → Observe) e può richiamare
gli altri agenti come tool. L'orchestratore decide autonomamente quante
iterazioni eseguire prima di rispondere.

Note retrieval:
- Il corpus BM25 viene persistito in `data/index/bm25_corpus.jsonl` (configurabile con `BM25_CORPUS_PATH`).
- `src/retrieval.py` usa quel file come fallback anche se ingestion e agent girano in processi separati.
- `EMBEDDING_MODE=mock` abilita test offline senza API/GPU.
- `RERANK_MODE` abilita reranking pluggable (`heuristic`, `cross_encoder`, `none`).

## Quality Loop Offline

Per migliorare prima del kickoff senza API key:

```bash
# Smoke end-to-end locale
./venv/bin/python scripts/smoke_check.py

# Benchmark validator/retrieval su casi dev
./venv/bin/python scripts/offline_benchmark.py \
  --cases data/dev/benchmark_cases.json \
  --min_confidence 0.6

# Stress test chunking/top_k
./venv/bin/python scripts/stress_retrieval.py

# Calibrazione soglia confidence
./venv/bin/python scripts/calibrate_confidence.py \
  --cases data/dev/benchmark_cases.json \
  --corpus data/dev/bm25_corpus.jsonl

# Hard-negative evaluation (controllo falsi positivi)
./venv/bin/python scripts/hard_negative_eval.py \
  --cases data/dev/hard_negative_cases.json \
  --corpus data/dev/hard_negative_corpus.jsonl
```

Policy fallback conservativa:
- `src/agent.py::run()` restituisce `[]` se l'output agente non e' JSON valido con `answer` lista.
- Questo evita output rumorosi in assenza di verifiche affidabili.
- Con `ENABLE_STRUCTURED_FALLBACK=1`, se l'agente LLM non produce output valido, viene usata la pipeline schema-first offline.

## Demo / PoC

La demo Streamlit supporta due modalita':
- `Structured PoC`: orchestrazione schema-first (funziona anche senza API key).
- `Agent`: usa l'orchestratore LLM con trace runtime.

Output demo:
- risposta finale
- riepilogo validator batch (`safe_items`, vincoli passati/falliti)
- payload JSON completo per debug rapido in hackathon

Demo Chainlit:
- UX chat-first per pitch agentico live
- stessi backend/tool (`run_with_trace`, `run_structured_orchestration`, `check_constraints_batch`)
- comandi rapidi: `/help`, `/mode`, `/constraints`, `/candidates`, `/topk`, `/evidence_limit`, `/timeout`, `/settings`
- quick actions in chat (bottoni mode/demo/settings/help)
- timeline step-by-step (`Step`) per orchestration, tool usage e validator summary

Runtime tuning utili demo:
- `APP_TIMEOUT_SECONDS`: timeout hard per chiamate runtime UI
- `HYBRID_CACHE_TTL_SECONDS`: cache retrieval query ripetute
- `BATCH_CACHE_TTL_SECONDS`: cache validator batch ripetuti
