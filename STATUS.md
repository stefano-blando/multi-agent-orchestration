# STATUS.md — Stato del progetto

Ultimo aggiornamento: 2026-02-18

## Stato generale: PRE-KICKOFF ✅

---

## Cosa è pronto

| Componente | File | Stato | Note |
|---|---|---|---|
| Struttura repo | - | ✅ | Git inizializzato |
| Docker Compose | `docker-compose.yml` | ✅ | Qdrant su porta 6333 |
| Requirements | `requirements.txt` | ✅ | Fixato (no extras inesistenti) |
| Hello world datapizza-ai | `hello_datapizza.py` | ✅ | Testato con MockClient |
| Ingestion pipeline | `src/ingest.py` | ✅ | PDF/DOCX/HTML/CSV, fallback in-memory |
| Agenti multi-agent | `src/agent.py` | ✅ | API reale verificata |
| Hybrid search | `src/retrieval.py` | ✅ | BM25 testato, RRF implementato |
| Eval Jaccard | `src/eval.py` | ✅ | Pronto, testare con dati reali |
| Embedding locale | sentence-transformers | ✅ | nomic-embed-text-v1 funziona |
| Kaggle notebook | `notebooks/kaggle_ingest.ipynb` | ✅ | Pronto da caricare |
| Demo Streamlit | `src/app.py` | ⏳ | Placeholder, da completare |
| CLAUDE.md | `CLAUDE.md` | ✅ | Contesto completo |

---

## Da fare prima dell'evento

- [ ] Installare Docker: `sudo apt install -y docker.io docker-compose-v2 nvidia-container-toolkit`
- [ ] Testare Docker + Qdrant: `docker compose up -d`
- [ ] Creare repository remoto (GitHub) e invitare il team
- [ ] Creare account Kaggle per tutti i membri (verifica GPU quota)
- [ ] Testare `hello_datapizza.py` con API key reale quando disponibile

---

## Da fare al kickoff (ordine di priorità)

1. **[0-30min]** Copiare dataset in `data/raw/`, esplorare formati e qualità
2. **[30-90min]** Lanciare `python src/ingest.py` — verificare chunk e parsing
3. **[90-120min]** Lanciare Kaggle notebook per embedding batch in parallelo
4. **[2-3h]** Implementare `check_constraint` con vincoli reali del challenge
5. **[3h+]** Prima run eval Jaccard — stabilire baseline score
6. **[ongoing]** Iterare su prompt, retrieval, chunking per migliorare score
7. **[ultimi 2h]** Completare demo Streamlit + README per presentazione

---

## Decisioni architetturali prese

| Decisione | Scelta | Motivazione |
|---|---|---|
| Framework agenti | datapizza-ai | Obbligatorio per challenge |
| Orchestrazione | can_call nativo datapizza-ai | Sufficiente, no LangGraph necessario |
| Vector store | Qdrant | Integrato in datapizza-ai |
| Embedding | nomic-embed-text-v1 locale | Gratis, GPU, 768-dim |
| Search | BM25 + Semantic (RRF) | Robusto su doc rumorosi |
| LLM agent | gpt-4o-mini (default) | Economico, veloce |
| Compute extra | Kaggle T4 | Batch embedding gratis |

---

## Score tracker (da aggiornare durante l'evento)

| Ora | Versione | Jaccard Score | Note |
|---|---|---|---|
| - | baseline | - | Da misurare al kickoff |

---

## Problemi noti

- Docker non installato → richiede `sudo` (fare prima dell'evento)
- `check_constraint` è un placeholder → da implementare col dataset reale
- `bm25_search` in `agent.py` dipende da `src/retrieval.corpus` popolato da ingest
- Corpus BM25 è in-memory → si perde al restart (da rilanciare ingest)
