"""Closed bid phase: place bids on missing ingredients."""

from __future__ import annotations

import json

from game_state import GameState
from tools import place_bid, end_phase
from utils.decision_engine import build_bidding_decision
from utils.strategy_flags import strat_value_bidding_enabled

TOOLS = [place_bid, end_phase]

SYSTEM_PROMPT = """\
You are the AI manager of restaurant "ventre a terra" in Hackapizza 2.0.
Goal: MAXIMIZE balance by winning bids at the lowest possible price.

## Your task: CLOSED_BID phase
The prompt gives you RECOMMENDED bids based on deterministic analysis.
These are a good default — but you have autonomy to adjust them.

## Decision process
1. Review the recommended bids and the historical price context provided.
2. You MAY call `market_analyst` (sub-agent) to verify historical prices, trends, or competition for specific ingredients.
3. You MAY adjust bid prices within ±30% of the recommended value if you have evidence (e.g. market_analyst shows lower clearing prices).
4. Execute final bids with place_bid().
5. end_phase()

## Rules
- If bids list is empty, go directly to end_phase().
- If market_analyst is slow or errors, just use the recommended bids as-is.
- If a tool returns an error, do NOT retry. Move on.
- ALWAYS call end_phase() at the end.
- You have memory of previous bidding turns — use patterns you've seen before.
"""


def build_prompt(state: GameState) -> str:
    decision = build_bidding_decision(state)
    selected = decision.selected
    bids = decision.bids
    drought_mode = decision.drought_mode
    num_cookable = decision.num_cookable
    budget_cap_override = decision.budget_cap_override
    use_v2_bid = decision.use_v2_bid
    json_bids = json.dumps(bids, ensure_ascii=False) if bids else "[]"
    # Keep latest bids in-memory for next menu pricing decisions.
    state._latest_bids = bids
    state._latest_bid_prices = {b["ingredient"]: int(b["bid"]) for b in bids}

    lines = [
        f"## State: Turn {state.turn_id} | Balance: {state.balance}",
        f"Served last turn: {state.served_last_turn}",
        f"Inventory: {json.dumps(state.inventory, ensure_ascii=False) if state.inventory else '(empty)'}",
    ]

    if bids:
        lines.append(f"\n## Action: Place these exact bids:")
        lines.append(f"place_bid({json_bids})")

        lines.append("\n## Reasoning (selected recipes, sorted by priority):")
        for r, missing in selected:
            status = "COMPLETE" if not missing else f"missing {len(missing)}: {', '.join(missing)}"
            lines.append(f"- P={r.get('prestige',0)} {r['name']}... → {status}")

        total_cost = sum(b["bid"] * b["quantity"] for b in bids)
        if drought_mode:
            if num_cookable >= 3:
                cap_pct = 0.12
            elif num_cookable >= 1:
                cap_pct = 0.18
            else:
                cap_pct = 0.24
        else:
            if num_cookable >= 3:
                cap_pct = 0.20
            elif num_cookable >= 1:
                cap_pct = 0.28
            else:
                cap_pct = 0.42
        budget_cap = min(int(state.balance * cap_pct), 700 if drought_mode else 1100)
        if budget_cap_override is not None:
            budget_cap = min(budget_cap, int(budget_cap_override))
        lines.append(f"\nTotal bid cost: {total_cost} (budget cap: {budget_cap})")
        if use_v2_bid:
            adjusted_count = len(state.bid_adjustments() or [])
            lines.append(
                f"V2 bid strategy ON: rolling_rev_avg={state.rolling_revenue_avg(3):.1f}, "
                f"adjusted_ingredients={adjusted_count}"
            )
        if strat_value_bidding_enabled():
            lines.append("Value-bidding ON: bids shaded by 2-round market avg/trend with margin safety cap.")
    else:
        lines.append("\n## No bids needed — we have all ingredients or budget too low.")

    # Append historical price context from bid archive if available.
    if bids and hasattr(state, 'bid_archive'):
        bid_ingredients = [b["ingredient"] for b in bids]
        archive_lines = []
        for ing in bid_ingredients:
            s = state.bid_archive.summary_for_ingredient(ing)
            if s.get("n_observations", 0) > 0:
                archive_lines.append(
                    f"  {ing}: avg={s['avg_price']}, range=[{s['min_price']}-{s['max_price']}], "
                    f"trend={s['trend']}, competition={s['competition']}, n={s['n_observations']}"
                )
        if archive_lines:
            lines.append("\n## Historical price context (from bid archive):")
            lines.extend(archive_lines)

    lines.append("\n→ Execute the actions above, then call end_phase().")
    return "\n".join(lines)
