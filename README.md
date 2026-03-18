# 🍕 Multi-Agent Orchestration

Bot competitivo per **Hackapizza 2.0**, basato su orchestrazione multi-agente event-driven.

## ✨ Cosa fa

- Gestisce il flusso di gioco via SSE (`game_started`, `phase_changed`, `client_spawned`, ...)
- Esegue strategie diverse per ogni fase: `speaking`, `closed_bid`, `waiting`, `serving`
- Usa tool MCP/HTTP per menu, bid, market, preparazione e servizio
- Tiene stato runtime, metriche e persistenza locale per decisioni più robuste

## 🗂️ Struttura Progetto

```text
app/
  main.py                # entrypoint runtime
  agent.py               # factory agent e sub-agent
  api_get.py             # client REST
  api_mcp.py             # client MCP
  game_state.py          # stato runtime ristorante
  tools.py               # tool chiamabili dagli agenti
  phases/                # logica e prompt per fase
  utils/                 # strategia, pricing, serving, persistence
  replay_metrics.py      # analisi replay

scripts/
  smoke_test.py          # check rapido API/SSE/MCP
  test_agent.py          # simulazione fasi senza loop SSE completo

tests/
  test_logic.py          # unit test logica strategica
```

## ⚙️ Setup

```bash
git clone git@github.com:stefano-blando/multi-agent-orchestration.git
cd multi-agent-orchestration
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Variabili minime in `.env`:
- `TEAM_API_KEY`
- `TEAM_ID`
- `REGOLO`

## ▶️ Run

```bash
python app/main.py
```

## 🧪 Test

```bash
pytest -q tests/test_logic.py
python scripts/smoke_test.py
python scripts/test_agent.py speaking
```
