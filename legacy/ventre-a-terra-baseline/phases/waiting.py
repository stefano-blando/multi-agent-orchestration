"""Waiting phase: check auction results, update menu, browse market."""

from __future__ import annotations

import json

from game_state import GameState
from tools import get_restaurant_info, set_menu, market_sell, get_market, market_execute, end_phase
from utils.decision_engine import build_waiting_menu_decision
from utils.market_utils import build_liquidation_sales
from utils.recipe_utils import compute_ingredient_competition
from utils.strategy_flags import menu_max_items

TOOLS = [get_restaurant_info, set_menu, market_sell, get_market, market_execute, end_phase]

SYSTEM_PROMPT = """\
You are the AI manager of restaurant "ventre a terra" in Hackapizza 2.0.

## Your task: WAITING phase
1. get_restaurant_info() to see what we won from auction
2. set_menu() with the EXACT JSON provided below — copy it character by character, do NOT modify names or prices
3. If surplus sales are listed below, execute each market_sell() call exactly as shown
4. OPTIONAL: If missing ingredients are listed below, you MAY call get_market() to check if cheap ingredients are available, and buy them with market_execute(entry_id). Only buy if price < typical bid price.
5. You MAY call `market_analyst` (sub-agent) to evaluate if a market purchase is worth it.
6. end_phase()

## Rules
- Copy the set_menu JSON EXACTLY as given. Do NOT retype recipe names. Do NOT change prices.
- Execute surplus sales EXACTLY as listed. Do NOT invent new ones.
- Market buying is optional — only buy if clearly profitable.
- If a tool returns an error, move on.
- ALWAYS call end_phase() at the end.
"""

def build_prompt(state: GameState) -> str:
    max_items = menu_max_items()
    decision = build_waiting_menu_decision(state, max_items=max_items)
    selected = decision.selected
    menu_items = decision.menu_items
    lines = [
        f"## State: Turn {state.turn_id} | Balance: {state.balance}",
        f"Served last turn: {state.served_last_turn}",
        f"Inventory: {json.dumps(state.inventory, ensure_ascii=False) if state.inventory else '(empty)'}",
    ]

    if menu_items:
        lines.append(f"\n## Action: Update menu to cookable recipes only:")
        lines.append(f"set_menu({json.dumps(menu_items, ensure_ascii=False)})")
    else:
        lines.append("\n## WARNING: No selected recipes are cookable!")
        lines.append("No menu update needed.")

    lines.append("\n## Recipe status:")
    for r, missing in selected:
        status = "COOKABLE" if not missing else f"MISSING: {', '.join(missing)}"
        lines.append(f"- P={r.get('prestige',0)} {r['name']}... → {status}")

    # Surplus liquidation sales
    competition = compute_ingredient_competition(state.recipes or [])
    sales = build_liquidation_sales(state.inventory, selected, competition)
    if sales:
        lines.append("\n## Surplus sales (ingredients expire at turn end):")
        for s in sales:
            lines.append(
                f'market_sell("{s["ingredient"]}", {s["quantity"]}, {s["price"]})'
            )
    else:
        lines.append("\n## No surplus to sell.")

    # Missing ingredients context for opportunistic market buying
    all_missing = set()
    for r, missing in selected:
        all_missing.update(missing)
    if all_missing:
        lines.append("\n## Missing ingredients (consider buying from market if cheap):")
        for ing in sorted(all_missing):
            avg_price = ""
            if hasattr(state, 'bid_archive'):
                s = state.bid_archive.summary_for_ingredient(ing)
                if s.get("n_observations", 0) > 0:
                    avg_price = f" (typical bid price: {s['avg_price']})"
            lines.append(f"- {ing}{avg_price}")

    lines.append("\n→ Execute the actions above, then call end_phase().")
    return "\n".join(lines)
