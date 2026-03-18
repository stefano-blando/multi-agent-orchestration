"""Serving phase: match clients to dishes, prepare and serve."""

from __future__ import annotations

import json

from game_state import GameState
from tools import (
    get_meals,
    prepare_dish,
    wait_for_preparation,
    serve_dish,
    get_market,
    market_execute,
    end_phase,
)
from utils.recipe_utils import compute_cookable_recipes
from utils.serving_utils import extract_intolerances, rank_dishes_for_serving

TOOLS = [
    get_meals,
    prepare_dish,
    wait_for_preparation,
    serve_dish,
    get_market,
    market_execute,
    end_phase,
]

_MAX_PENDING_CLIENTS_IN_PROMPT = 12
_MAX_AVAILABLE_DISHES_IN_PROMPT = 18

SYSTEM_PROMPT = """\
You are the AI manager of restaurant "ventre a terra" in Hackapizza 2.0.

## Your task: SERVING phase
1. Call get_meals(turn_id) using the turn_id provided below.
2. From get_meals, keep only executed=false and status=waiting.
3. Build the full FIFO queue of ALL waiting meal IDs (oldest startTime first).
4. For EACH waiting meal ID in that queue (one at a time, do not skip IDs silently):
   - serve ONLY the exact requested dish name from the meal request
   - if request is ingredient-based ("something with X, Y"), choose a cookable menu dish that contains ALL requested ingredients
   - if the exact requested dish is not available/preparable, skip that client
   - prepare_dish(dish_name) — copy the name EXACTLY from the list
   - wait_for_preparation(dish_name)
   - serve_dish(dish_name, client_id) where client_id is the meal "id" field (as string)
5. After processing all current waiting clients, ALWAYS call end_phase() with handled/skipped IDs.
   - The orchestrator will trigger serving again on new clients or keepalive polls.

## Rules
- Use meal "id" from get_meals as client_id for serve_dish.
- Call get_meals only once per phase run.
- Copy dish names EXACTLY from the Available dishes list. Do NOT retype them.
- Note: the Available dishes section may be truncated for brevity; if needed, use exact dish name from get_meals request text.
- Ingredient-based requests are valid: when no exact dish name is present, serve a dish that includes all requested ingredients.
- If a client mentions intolerance (e.g. "I'm intolerant to X"), do NOT serve a dish containing ingredient X. Skip that client.
- For each client, do at most one serve attempt via serve_dish.
- Do NOT serve substitute dishes; only exact request.
- If a tool returns an error, skip that client and move to the next.
- You MUST process all waiting IDs returned by get_meals in this run.
- ALWAYS call end_phase() after this run's queue is processed, even if cookable capacity remains.
"""


def build_prompt(state: GameState) -> str:
    recipes = state.recipes or []

    lines = [
        f"## State: Turn {state.turn_id} | Balance: {state.balance}",
        f"turn_id to use for get_meals: {state.turn_id}",
        f"Inventory: {json.dumps(state.inventory, ensure_ascii=False) if state.inventory else '(empty)'}",
    ]

    unserved = state.unserved_clients()
    if unserved:
        lines.append("\n## Pending clients from SSE (IDs may be missing until get_meals):")
        shown_clients = unserved[:_MAX_PENDING_CLIENTS_IN_PROMPT]
        for c in shown_clients:
            cid = c.client_id if c.client_id else "unknown_from_sse"
            intol = extract_intolerances(c.order_text)
            intol_warn = f" ⚠ INTOLERANT TO: {', '.join(intol)}" if intol else ""
            lines.append(f"- client_id={cid}, name={c.name}: \"{c.order_text}\"{intol_warn}")
        if len(unserved) > len(shown_clients):
            lines.append(f"- ... and {len(unserved) - len(shown_clients)} more pending clients")

        cookable = compute_cookable_recipes(recipes, state.inventory)
        ranked = rank_dishes_for_serving(cookable)
        if ranked:
            lines.append("\n## Available dishes to serve:")
            shown_dishes = ranked[:_MAX_AVAILABLE_DISHES_IN_PROMPT]
            for r in shown_dishes:
                ings = ", ".join(r.get("ingredients", {}).keys())
                prep_ms = int(r.get("preparationTimeMs", 0))
                lines.append(f"- {r['name']} (P={r.get('prestige',0)}, prepMs={prep_ms}, ingredients: {ings})")
            if len(ranked) > len(shown_dishes):
                lines.append(f"- ... and {len(ranked) - len(shown_dishes)} more cookable dishes")
        else:
            lines.append("\n## No dishes available. Call end_phase().")
    else:
        lines.append("\n## No pending clients.")

    if state.prepared_dishes:
        lines.append(f"\n## Prepared (ready to serve): {state.prepared_dishes}")

    # Cookable capacity summary — drives end_phase decision
    can_cook_more = state.can_cook_any_menu_recipe()
    if can_cook_more:
        cookable_menu = compute_cookable_recipes(recipes, state.inventory)
        menu_names = {
            item if isinstance(item, str) else str(item.get("name", ""))
            for item in state.menu
            if item
        }
        cookable_names = [r["name"] for r in cookable_menu if r.get("name") in menu_names]
        lines.append(f"\n## Remaining cookable capacity: YES — {len(cookable_names)} menu dish(es) still cookable: {cookable_names}")
        lines.append("→ Cookable capacity remains, but still call end_phase() after processing current waiting IDs.")
    else:
        lines.append("\n## Remaining cookable capacity: NONE — inventory depleted for all menu items.")
        lines.append("→ Before calling end_phase(), you MAY call get_market() to check if missing ingredients are available cheaply. If you find a good deal, buy with market_execute(entry_id) and continue serving.")
        lines.append("→ Otherwise, call end_phase() after serving all current clients.")

    lines.append(
        "\n## Execution policy: get_meals once -> build full waiting ID queue -> FIFO by oldest startTime -> exact dish OR ingredient-compatible dish -> one serve attempt per client."
    )
    lines.append("\n## Mandatory: process every waiting ID from get_meals and report handled/skipped IDs in end_phase summary (if calling end_phase).")
    return "\n".join(lines)
