# CLAUDE.md — Contesto progetto Hackapizza 2.0

## Cos'è questo progetto
Soluzione per **Hackapizza 2.0 — Agent Edition** (DataPizza, fine febbraio 2026).
Challenge: sistema multi-agente che risponde a query in linguaggio naturale su un dataset
di documenti eterogenei e rumorosi (PDF, HTML, DOCX, CSV con typo e info fuorvianti).
Scoring: **Jaccard Similarity** tra risposta e ground truth.

## Team
- **Stefano** — Tech Lead, infra, GPU (Linux Mint 22.3, RTX A1000)
- **Piergiuseppe Pezzoli** — Agent logic, prompt engineering, eval (AIRIC Polimi)
- **Andrea Chizzola** — Ingestion pipeline, RAG (DEIB Polimi)
- **Tomaso Castellani** — Retrieval, tools, hybrid search (DEIB Polimi)

## Stack
- **Framework agenti**: `datapizza-ai` (obbligatorio per la challenge)
- **Vector store**: Qdrant (Docker in prod, in-memory come fallback)
- **Embedding**: `nomic-ai/nomic-embed-text-v1` via sentence-transformers (locale/GPU)
- **Hybrid search**: BM25 (rank-bm25) + Semantic con Reciprocal Rank Fusion
- **Document parsing**: Docling (PDF/DOCX), BeautifulSoup (HTML), pandas (CSV)
- **Demo**: Streamlit
- **Eval**: Jaccard Similarity locale (src/eval.py)

## Architettura agenti (datapizza-ai)
```
OrchestratorAgent
  ├── RetrieverAgent   → tools: [semantic_search]     → src/retrieval.py
  ├── ValidatorAgent   → tools: [check_constraint]    → da implementare col dataset
  └── SynthesizerAgent → tools: []
```
Gli agenti si chiamano tramite `orchestrator.can_call([retriever, validator, synthesizer])`.
`agent.run(task)` ritorna `StepResult` con `.text`, `.tools_used`, `.usage`.

## API datapizza-ai (note chiave)
- `Agent(name, client, system_prompt, tools)` — tutti obbligatori
- `orchestrator.can_call(agent)` — registra un agente come tool
- `agent.as_tool()` — converte agente in Tool (description = None se non specificata)
- Client: `OpenAIClient(api_key, model)` oppure `MockClient()` per test senza chiavi
- **Non esistono extras**: usare `pip install datapizza-ai openai` separati
- Qdrant: usare `client.query_points(collection, query, limit)` (non `.search()`)

## Comandi utili
```bash
# Setup
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Infrastruttura
docker compose up -d          # Qdrant su porta 6333 (richiede Docker installato)

# Ingestion (dopo kickoff con dataset)
python src/ingest.py --data_dir data/raw/

# Test pipeline
python src/agent.py

# Evaluation
python src/eval.py --predictions predictions.json --ground_truth gt.json

# Hello world datapizza-ai
python hello_datapizza.py

# BM25 test
python -m src.retrieval

# Demo (ultima priorità)
streamlit run src/app.py
```

## Note operative
- Le API keys vengono fornite da DataPizza il giorno dell'evento
- Mettere le chiavi in `.env` (non committare mai `.env`)
- Docker non è ancora installato sulla macchina — installare con:
  `sudo apt install -y docker.io docker-compose-v2 nvidia-container-toolkit`
- `einops` è richiesto da nomic-embed-text-v1
- Il corpus BM25 (`src/retrieval.corpus`) è in-memory: va ripopolato ad ogni restart
- Kaggle notebook pronto in `notebooks/kaggle_ingest.ipynb` per batch embedding su T4

## Priorità al kickoff
1. Copiare il dataset in `data/raw/`
2. Aggiornare `check_constraint` in `src/agent.py` con i vincoli reali
3. Lanciare `python src/ingest.py`
4. Verificare score iniziale con `src/eval.py`
5. Iterare su prompt engineering e retrieval
