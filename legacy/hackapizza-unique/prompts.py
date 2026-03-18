"""System prompt and phase prompt builder for the restaurant agent."""

from __future__ import annotations

import json

from game_state import GameState
from utils.recipe_utils import (
    compute_cookable_recipes,
    compute_ingredient_competition,
    select_best_recipes,
)
from utils.bid_utils import compute_smart_bids
from utils.menu_utils import build_cookable_menu, build_menu_items, recipe_price
from utils.serving_utils import rank_dishes_for_serving


# ── System prompt ────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are the AI manager of restaurant "ventre a terra" in Hackapizza 2.0.
Goal: MAXIMIZE balance. Be concise and efficient.

## Phase actions

### SPEAKING
1. get_restaurant_info() to check balance and inventory
2. set_menu() with the menu provided in the phase prompt
3. open_close_restaurant(true)
4. end_phase()

### CLOSED_BID
1. The phase prompt gives you the EXACT bids to place. Execute them with place_bid().
2. end_phase()

### WAITING
1. get_restaurant_info() to see what we won from auction
2. The phase prompt tells you which recipes are cookable. Update menu with set_menu().
3. Optionally check get_market() for cheap ingredients we still need.
4. end_phase()

### SERVING
For EACH pending client (one at a time):
1. Pick the best matching dish from "Available dishes" in the phase prompt
2. prepare_dish(dish_name) — use the EXACT full recipe name
3. wait_for_preparation(dish_name) — blocks until ready (max 15s)
4. serve_dish(dish_name, client_id)
If no dishes available or no match, skip that client.
Serve highest-price dishes first to maximize revenue.
After all clients handled: end_phase()

## Rules
- ALWAYS end with end_phase()
- Be extremely concise
- If a tool returns an error, do NOT retry. Move on.
- Execute the instructions in the phase prompt precisely.
"""


# ── Phase prompt builder ─────────────────────────────────────────────────

def build_phase_prompt(state: GameState) -> str:
    """Build contextual prompt with pre-computed strategy for the current phase."""
    lines = [
        f"## State: Turn {state.turn_id} | Phase: {state.phase} | Balance: {state.balance}",
        f"Inventory: {json.dumps(state.inventory, ensure_ascii=False) if state.inventory else '(empty)'}",
        f"Open: {state.is_open}",
    ]

    recipes = state.recipes or []
    competition = compute_ingredient_competition(recipes)
    selected = select_best_recipes(recipes, state.inventory, n=5)

    if state.phase == "speaking":
        menu_items = build_menu_items([r for r, _ in selected])
        lines.append(f"\n## Action: Set this menu and open restaurant:")
        lines.append(f"set_menu({json.dumps(menu_items, ensure_ascii=False)})")
        lines.append("open_close_restaurant(true)")

        lines.append("\n## Selected recipes (top 5 by feasibility):")
        for r, missing in selected:
            status = "READY" if not missing else f"need {len(missing)}: {', '.join(missing)}"
            lines.append(f"- P={r.get('prestige',0)} | {r['name'][:60]} | {status}")

    elif state.phase == "closed_bid":
        bids = compute_smart_bids(selected, state.inventory, state.balance, competition)
        if bids:
            lines.append(f"\n## Action: Place these exact bids:")
            lines.append(f"place_bid({json.dumps(bids, ensure_ascii=False)})")

            lines.append("\n## Reasoning (selected recipes, sorted by priority):")
            for r, missing in selected:
                status = "COMPLETE" if not missing else f"missing {len(missing)}: {', '.join(missing)}"
                lines.append(f"- P={r.get('prestige',0)} {r['name'][:50]}... → {status}")

            total_cost = sum(b["bid"] * b["quantity"] for b in bids)
            lines.append(f"\nTotal bid cost: {total_cost} (budget cap: {int(state.balance * 0.4)})")
        else:
            lines.append("\n## No bids needed — we have all ingredients or budget too low.")

    elif state.phase == "waiting":
        cookable = compute_cookable_recipes(recipes, state.inventory)
        selected_names = {r["name"] for r, _ in selected}
        menu_items = build_cookable_menu(selected_names, cookable)

        if menu_items:
            lines.append(f"\n## Action: Update menu to cookable recipes only:")
            lines.append(f"set_menu({json.dumps(menu_items, ensure_ascii=False)})")
        else:
            lines.append("\n## WARNING: No selected recipes are cookable!")
            lines.append("Check get_market() for missing ingredients, or close restaurant.")

        lines.append("\n## Recipe status:")
        for r, missing in selected:
            status = "COOKABLE" if not missing else f"MISSING: {', '.join(missing)}"
            lines.append(f"- P={r.get('prestige',0)} {r['name'][:50]}... → {status}")

    elif state.phase == "serving":
        unserved = state.unserved_clients()
        if unserved:
            lines.append("\n## Pending clients:")
            for c in unserved:
                lines.append(f"- client_id={c.client_id}, name={c.name}: \"{c.order_text}\"")

            cookable = compute_cookable_recipes(recipes, state.inventory)
            ranked = rank_dishes_for_serving(cookable)
            if ranked:
                lines.append("\n## Available dishes to serve:")
                for r in ranked:
                    lines.append(f"- {r['name']} (P={r.get('prestige',0)}, price={recipe_price(r)})")
            else:
                lines.append("\n## No dishes available! Close restaurant.")
        else:
            lines.append("\n## No pending clients.")

        if state.prepared_dishes:
            lines.append(f"\n## Prepared (ready to serve): {state.prepared_dishes}")

    lines.append(f"\n→ Execute the actions above, then call end_phase().")

    return "\n".join(lines)
