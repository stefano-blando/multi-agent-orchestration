# Multi-Agent Orchestration (Hackapizza 2.0)

Repository rifattorizzata sul codice runtime avanzato sviluppato in `ventre-a-terra`.

Il focus e' il bot competitivo event-driven per Hackapizza 2.0:
- loop SSE di gioco
- orchestrazione per fase (`speaking`, `closed_bid`, `waiting`, `serving`)
- tools MCP/HTTP + strategy engine + persistence locale

## Struttura

- `main.py`: entrypoint SSE, dispatch fasi, retry, KPI e failover
- `agent.py`: factory agent e sub-agent market analyst
- `api_get.py`, `api_mcp.py`: client API
- `game_state.py`: stato runtime del ristorante
- `phases/`: prompt + logica per ogni fase
- `utils/`: bidding/menu/serving/market/state persistence
- `tools.py`: tool datapizza-ai usati dagli agenti
- `tests/test_logic.py`: test unit logica strategica
- `smoke_test.py`, `test_agent.py`: test operativi

## Setup rapido

```bash
git clone git@github.com:stefano-blando/multi-agent-orchestration.git
cd multi-agent-orchestration
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Configura `.env` con:
- `TEAM_API_KEY`
- `TEAM_ID`
- `REGOLO`

## Esecuzione

```bash
python main.py
```

## Test

```bash
pytest -q tests/test_logic.py
python smoke_test.py
```

## Note

- Il materiale pre-kickoff (demo Streamlit/Chainlit, pipeline retrieval sperimentale) e' stato rimosso dalla root.
- La repo ora e' costruita sul codice operativo usato in `ventre-a-terra`.
