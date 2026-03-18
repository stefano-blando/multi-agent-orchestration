"""Bid calculation logic — adaptive based on inventory gaps and budget."""

from __future__ import annotations


def bid_price(freq: int) -> int:
    """Bid per unit based on how many recipes use this ingredient."""
    if freq < 20:
        return 10
    elif freq < 35:
        return 15
    elif freq < 50:
        return 20
    return 25


def _market_ref_price(
    ingredient: str,
    competition: dict[str, int],
    market_signal_fn=None,
    bid_archive=None,
) -> int:
    """Best bid reference price for an ingredient.

    Uses market signal (short-horizon avg_win, then long-horizon EMA from price_db)
    when available.  Falls back to bid_archive, then competition-based heuristic.
    """
    if not callable(market_signal_fn):
        # Try bid archive before competition heuristic.
        if bid_archive is not None:
            avg = bid_archive.avg_clearing_price(ingredient)
            if avg > 0:
                trend_label = bid_archive.price_trend(ingredient)
                trend_mult = {"rising": 1.08, "falling": 0.93, "stable": 1.0}.get(trend_label, 1.0)
                return max(1, int(round(avg * trend_mult)))
        return bid_price(competition.get(ingredient, 0))

    sig: dict = market_signal_fn(ingredient) or {}

    # Short-horizon: last 2-turn average clearing price.
    avg_win_2 = float(sig.get("avg_win_2", 0.0))
    if avg_win_2 > 0:
        trend = max(0.8, min(1.5, float(sig.get("trend", 1.0))))
        return max(1, int(round(avg_win_2 * trend)))

    # Mid-horizon: bid archive average clearing price.
    if bid_archive is not None:
        avg = bid_archive.avg_clearing_price(ingredient)
        if avg > 0:
            trend_label = bid_archive.price_trend(ingredient)
            trend_mult = {"rising": 1.08, "falling": 0.93, "stable": 1.0}.get(trend_label, 1.0)
            return max(1, int(round(avg * trend_mult)))

    # Long-horizon: EMA from price_db (requires at least 3 observations).
    ema_short = float(sig.get("ema_short", 0.0))
    n_obs = float(sig.get("n_obs", 0.0))
    if ema_short > 0 and n_obs >= 3:
        long_trend = max(0.85, min(1.3, float(sig.get("long_trend", 1.0))))
        return max(1, int(round(ema_short * long_trend)))

    # Competition-based heuristic fallback.
    return bid_price(competition.get(ingredient, 0))


def compute_smart_bids(
    selected: list[tuple[dict, list[str]]],
    inventory: dict,
    balance: float,
    competition: dict[str, int],
    num_cookable: int = 0,
    demand_by_dish: dict[str, int] | None = None,
    low_demand_mode: bool = False,
    bid_adjustments: dict[str, int] | None = None,
    budget_cap_override: int | float | None = None,
    market_signal_fn=None,
    bid_archive=None,
) -> list[dict]:
    """Calculate bids for missing ingredients of selected recipes.

    Priority: selected order (core first, then premium/ROI).
    Budget cap: dynamic by existing cookable coverage.
    Quantity: boosts high-demand ingredients to support multi-client service.
    """
    # If we already have cookable dishes, bid less aggressively.
    if low_demand_mode:
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

    budget_cap = min(balance * cap_pct, 700 if low_demand_mode else 1100)
    if budget_cap_override is not None:
        budget_cap = min(float(budget_cap), float(max(0, budget_cap_override)))
    total_cost = 0
    bids: dict[str, dict] = {}
    adjustments = bid_adjustments or {}

    # Demand of each missing ingredient across selected recipes.
    demand: dict[str, int] = {}
    selected_missing = [(r, m) for r, m in selected if m]
    for _, missing in selected_missing:
        for ing in missing:
            demand[ing] = demand.get(ing, 0) + 1

    for idx, (recipe, missing) in enumerate(selected_missing):
        # Favor recipes that can be completed with few ingredients (faster to unlock).
        if idx == 0:
            recipe_boost_qty = 2
        else:
            recipe_boost_qty = 1

        def desired_qty(ing: str, boost_qty: int) -> int:
            ing_demand = demand.get(ing, 0)
            if ing_demand >= 3:
                shared_qty = 3
            elif ing_demand >= 2:
                shared_qty = 2
            else:
                shared_qty = 1
            qty_cap = 2 if low_demand_mode else 3
            return min(qty_cap, max(shared_qty, boost_qty))

        new_ings = [ing for ing in missing if ing not in bids]
        recipe_cost = sum(
            max(1, _market_ref_price(ing, competition, market_signal_fn, bid_archive) + int(adjustments.get(ing, 0)))
            * desired_qty(ing, recipe_boost_qty)
            for ing in new_ings
        )

        if total_cost + recipe_cost > budget_cap:
            # Fallback with quantity 1 for non-shared ingredients.
            recipe_cost = sum(
                max(1, _market_ref_price(ing, competition, market_signal_fn, bid_archive) + int(adjustments.get(ing, 0)))
                * (1 if low_demand_mode else (2 if demand.get(ing, 0) >= 2 else 1))
                for ing in new_ings
            )
            if total_cost + recipe_cost > budget_cap:
                continue

        for ing in new_ings:
            if ing in bids:
                continue
            bp = max(1, _market_ref_price(ing, competition, market_signal_fn, bid_archive) + int(adjustments.get(ing, 0)))
            qty = desired_qty(ing, recipe_boost_qty)
            if total_cost + (bp * qty) > budget_cap:
                qty = 1 if low_demand_mode else (2 if demand.get(ing, 0) >= 2 else 1)
            if total_cost + (bp * qty) > budget_cap:
                continue
            bids[ing] = {"ingredient": ing, "bid": bp, "quantity": qty}
            total_cost += bp * qty

    # If budget remains, top up ingredients for already-cookable high-demand dishes.
    if demand_by_dish and total_cost < (budget_cap * 0.85) and not low_demand_mode:
        cookable_selected = [(r, m) for r, m in selected if not m]
        prioritized = sorted(
            cookable_selected,
            key=lambda x: (
                -int(demand_by_dish.get(str(x[0].get("name", "")), 0)),
                int(x[0].get("preparationTimeMs", 999999)),
                -int(x[0].get("prestige", 0)),
            ),
        )

        for recipe, _ in prioritized:
            dish_name = str(recipe.get("name", ""))
            demand_score = int(demand_by_dish.get(dish_name, 0))
            if demand_score < 2:
                continue

            # Keep a modest target stock per demanded recipe to serve bursts.
            target_servings = 1 + (1 if demand_score >= 3 else 0) + (1 if demand_score >= 6 else 0)
            for ing, qty in recipe.get("ingredients", {}).items():
                qty_i = int(qty)
                if qty_i <= 0:
                    continue

                desired_total = qty_i * target_servings
                current_stock = int(inventory.get(ing, 0))
                already_bid = int(bids.get(ing, {}).get("quantity", 0))
                shortage = max(0, desired_total - (current_stock + already_bid))
                if shortage <= 0:
                    continue

                bp = max(1, _market_ref_price(ing, competition, market_signal_fn, bid_archive) + int(adjustments.get(ing, 0)))
                affordable = int((budget_cap - total_cost) // bp)
                if affordable <= 0:
                    return list(bids.values())

                add_qty = min(shortage, affordable, 2)
                if add_qty <= 0:
                    continue

                if ing in bids:
                    bids[ing]["quantity"] = int(bids[ing]["quantity"]) + add_qty
                else:
                    bids[ing] = {"ingredient": ing, "bid": bp, "quantity": add_qty}
                total_cost += bp * add_qty

                if total_cost >= budget_cap:
                    return list(bids.values())

    return list(bids.values())
