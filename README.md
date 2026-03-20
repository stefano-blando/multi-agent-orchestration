# Multi-Agent Orchestration

Competitive bot for **Hackapizza 2.0**, built around event-driven multi-agent orchestration.

## What It Does

- Handles game flow through SSE events (`game_started`, `phase_changed`, `client_spawned`, ...)
- Executes different strategies for each phase: `speaking`, `closed_bid`, `waiting`, `serving`
- Uses MCP/HTTP tools for menu, bidding, market, preparation, and serving actions
- Maintains runtime state, metrics, and local persistence for more robust decision-making

## Project Structure

```text
app/
  main.py                # runtime entrypoint
  agent.py               # agent and sub-agent factory
  api_get.py             # REST client
  api_mcp.py             # MCP client
  game_state.py          # restaurant runtime state
  tools.py               # callable tools exposed to agents
  phases/                # phase logic and prompts
  utils/                 # strategy, pricing, serving, persistence
  replay_metrics.py      # replay analysis

scripts/
  smoke_test.py          # quick API/SSE/MCP check
  test_agent.py          # phase simulation without full SSE loop

tests/
  test_logic.py          # unit tests for strategic logic
```

## Setup

```bash
git clone git@github.com:stefano-blando/multi-agent-orchestration.git
cd multi-agent-orchestration
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Minimum variables in `.env`:
- `TEAM_API_KEY`
- `TEAM_ID`
- `REGOLO`

## Run

```bash
python app/main.py
```

## Test

```bash
pytest -q tests/test_logic.py
python scripts/smoke_test.py
python scripts/test_agent.py speaking
```
