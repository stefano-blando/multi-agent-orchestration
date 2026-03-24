# Multi-Agent Orchestration

Event-driven multi-agent coordination system for auction-based procurement, inventory allocation, and real-time fulfillment under demand uncertainty.

## Overview

This project implements a real-time coordination loop in a competitive market-like environment.

At a high level, the system solves four linked problems:

- policy update: choose the active supply set for the next operating cycle
- procurement: place bids on missing inputs under budget constraints
- inventory reconciliation: update feasible choices after market outcomes and liquidate low-value surplus
- fulfillment: serve queued demand under timing constraints while tracking operational KPIs

The core point is not generic prompting. The interesting part is the orchestration layer: live events update a shared state ledger, phase-specific agents act through a restricted capability surface, selected memory is preserved across turns, and replay metrics make the runtime inspectable afterwards.

## Coordination Cycle

The runtime is organized around four phases:

- `speaking`: policy update
- `closed_bid`: procurement auction
- `waiting`: inventory reconciliation
- `serving`: fulfillment

Each phase has its own prompt, action surface, and execution contract. This keeps the system readable and reduces failure modes that would appear in a single monolithic agent.

## Architecture

```text
app/
  main.py                # event loop and phase dispatch
  agent.py               # phase-specific agent factory and memory handling
  api_get.py             # state and market reads
  api_mcp.py             # action interface
  game_state.py          # shared runtime state ledger
  tools.py               # callable capabilities exposed to agents
  phases/                # phase prompts and phase logic
  utils/                 # strategy, pricing, serving, persistence
  replay_metrics.py      # KPI replay analysis

scripts/
  smoke_test.py          # quick connectivity and action-surface check
  test_agent.py          # local phase simulation without full event loop

tests/
  test_logic.py          # unit tests for strategic logic
```

## Runtime Features

- event-driven coordination through live phase and demand updates
- shared state tracking for balance, inventory, market context, queue state, and KPIs
- bounded memory for procurement decisions across recent turns
- auxiliary analyst path for price and competition checks
- replay-oriented logging for conversion, revenue, spend, latency, and failures

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

These credentials target the original competitive environment used by the project.

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
