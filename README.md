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

# 3. Ingestion (dopo aver ricevuto il dataset, Qdrant gira in-memory)
python src/ingest.py --data_dir data/raw/

# 4. Test pipeline
python src/agent.py

# 5. Eval
python src/eval.py --predictions predictions.json --ground_truth gt.json

# 6. Demo
streamlit run src/app.py
```

## Architettura

Sistema multi-agente basato su [datapizza-ai](https://github.com/datapizza-labs/datapizza-ai).
L'agente pianifica autonomamente, usa tool in sequenza e si auto-corregge fino a produrre una risposta soddisfacente.

```
Query
  └── OrchestratorAgent          # pianifica e coordina
        ├── RetrieverAgent       # cerca nei documenti (BM25 + semantic)
        ├── ValidatorAgent       # verifica vincoli e normative
        └── SynthesizerAgent     # produce la risposta finale
```

Ogni agente opera in un loop ReAct (Reason → Act → Observe) e può richiamare
gli altri agenti come tool. L'orchestratore decide autonomamente quante
iterazioni eseguire prima di rispondere.
