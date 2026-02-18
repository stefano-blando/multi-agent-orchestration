# Hackapizza 2.0 — Agent Edition

Multi-agent RAG system per Hackapizza 2.0.

## Setup rapido

```bash
# 1. Clone & venv
git clone <repo_url>
cd hackapizza-team
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Config
cp .env.example .env
# → Modifica .env con le chiavi API

# 3. Avvia Qdrant
docker compose up -d

# 4. Ingestion (appena ricevete il dataset)
python src/ingest.py --data_dir data/raw/

# 5. Test pipeline
python src/agent.py

# 6. Eval
python src/eval.py --predictions predictions.json --ground_truth gt.json

# 7. Demo (solo alla fine)
streamlit run src/app.py
```

## Architettura

```
Query → OrchestratorAgent
           ├── RetrieverAgent  (semantic + BM25)
           ├── ValidatorAgent  (constraint checking)
           └── SynthesizerAgent (risposta finale)
```

## Team
- Stefano — Infra / Tech Lead
- Piergiuseppe Pezzoli — Agent Logic / Eval
- Andrea Chizzola — Ingestion Pipeline
- Tomaso Castellani — Retrieval / Tools
