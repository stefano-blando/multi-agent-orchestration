"""Speaking phase: set menu and open restaurant."""

from __future__ import annotations

import json

from game_state import GameState
from tools import get_restaurant_info, set_menu, open_close_restaurant, end_phase
from utils.strategy_flags import menu_max_items, strat_value_pricing_enabled
from utils.decision_engine import build_speaking_menu_decision

TOOLS = [get_restaurant_info, set_menu, open_close_restaurant, end_phase]

SYSTEM_PROMPT = """\
You are the AI manager of restaurant "ventre a terra" in Hackapizza 2.0.

## Your task: SPEAKING phase
1. set_menu() with the EXACT JSON provided below — copy it character by character, do NOT modify names or prices
2. open_close_restaurant(true)
3. end_phase()

## Rules
- Copy the set_menu JSON EXACTLY as given. Do NOT retype recipe names (typos cause errors).
- Do NOT change prices.
- If a tool returns an error, move on.
- ALWAYS call end_phase() at the end.
"""


def build_prompt(state: GameState) -> str:
    max_items = menu_max_items()
    decision = build_speaking_menu_decision(state, max_items=max_items)
    selected = decision.selected
    menu_items = decision.menu_items

    lines = [
        f"## State: Turn {state.turn_id} | Balance: {state.balance}",
        f"Served last turn: {state.served_last_turn}",
        f"Inventory: {json.dumps(state.inventory, ensure_ascii=False) if state.inventory else '(empty)'}",
        f"\n## Action: Set this menu and open restaurant:",
        f"set_menu({json.dumps(menu_items, ensure_ascii=False)})",
        "open_close_restaurant(true)",
        "\n## Selected recipes (top candidates by feasibility):",
    ]
    for r, missing in selected:
        status = "READY" if not missing else f"need {len(missing)}: {', '.join(missing)}"
        lines.append(f"- P={r.get('prestige',0)} | {r['name']} | {status}")

    if strat_value_pricing_enabled():
        lines.append("\nValue-pricing ON: COGS floor + recent realized prices + scarcity/demand weighting.")

    lines.append("\n→ Execute the actions above, then call end_phase().")
    return "\n".join(lines)
