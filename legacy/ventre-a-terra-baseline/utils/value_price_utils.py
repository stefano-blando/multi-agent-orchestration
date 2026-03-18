"""Value-based menu pricing helpers."""

from __future__ import annotations

from utils.bid_utils import bid_price


def _safe_signal(market_signal_fn, ingredient: str) -> dict[str, float]:
    if not callable(market_signal_fn):
        return {"avg_win_2": 0.0, "competition_2": 0.0, "trend": 1.0, "rarity": 1.0}
    out = market_signal_fn(ingredient) or {}
    return {
        "avg_win_2": float(out.get("avg_win_2", 0.0)),
        "competition_2": float(out.get("competition_2", 0.0)),
        "trend": float(out.get("trend", 1.0) or 1.0),
        "rarity": float(out.get("rarity", 1.0) or 1.0),
    }


def estimate_ingredient_unit_cost(
    ingredient: str,
    ingredient_bid_prices: dict[str, int] | None = None,
    competition: dict[str, int] | None = None,
    market_signal_fn=None,
) -> float:
    latest = ingredient_bid_prices or {}
    if ingredient in latest:
        try:
            return float(max(1, int(latest[ingredient])))
        except (TypeError, ValueError):
            pass
    sig = _safe_signal(market_signal_fn, ingredient)
    if sig["avg_win_2"] > 0:
        return max(1.0, sig["avg_win_2"])
    comp = competition or {}
    return float(max(1, bid_price(int(comp.get(ingredient, 0)))))


def estimate_recipe_cogs(
    recipe: dict,
    ingredient_bid_prices: dict[str, int] | None = None,
    competition: dict[str, int] | None = None,
    market_signal_fn=None,
) -> float:
    total = 0.0
    for ingredient, qty in recipe.get("ingredients", {}).items():
        try:
            qty_i = max(0, int(qty))
        except (TypeError, ValueError):
            qty_i = 0
        if qty_i <= 0:
            continue
        unit = estimate_ingredient_unit_cost(
            str(ingredient),
            ingredient_bid_prices=ingredient_bid_prices,
            competition=competition,
            market_signal_fn=market_signal_fn,
        )
        total += unit * float(qty_i)
    return max(1.0, total)


def scarcity_factor_for_recipe(recipe: dict, market_signal_fn=None) -> float:
    comps: list[float] = []
    for ingredient in recipe.get("ingredients", {}):
        sig = _safe_signal(market_signal_fn, str(ingredient))
        c = sig["competition_2"]
        if c > 0:
            comps.append(c)
    if not comps:
        return 1.0
    avg_comp = sum(comps) / float(len(comps))
    if avg_comp <= 1.5:
        return 1.20
    if avg_comp <= 3.0:
        return 1.08
    if avg_comp <= 5.0:
        return 1.00
    if avg_comp <= 7.0:
        return 0.92
    return 0.86


def value_based_recipe_price(
    recipe: dict,
    fallback_price: int,
    dish_name: str,
    demand_score: int,
    served_score: int,
    dish_price_history: dict[str, list[int]] | None = None,
    ingredient_bid_prices: dict[str, int] | None = None,
    competition: dict[str, int] | None = None,
    market_signal_fn=None,
    inventory: dict[str, int] | None = None,
) -> int:
    """Compute dish price using COGS floor + scarcity-adjusted market reference."""
    cogs = estimate_recipe_cogs(
        recipe,
        ingredient_bid_prices=ingredient_bid_prices,
        competition=competition,
        market_signal_fn=market_signal_fn,
    )
    floor_price = cogs * 1.20

    history = (dish_price_history or {}).get(dish_name, [])
    hist_recent = [int(x) for x in history[-2:] if int(x) > 0]
    hist_max = max(hist_recent) if hist_recent else 0
    hist_avg = (sum(hist_recent) / float(len(hist_recent))) if hist_recent else 0.0

    reference = hist_avg if hist_avg > 0 else float(max(50, int(fallback_price)))
    scarcity = scarcity_factor_for_recipe(recipe, market_signal_fn=market_signal_fn)

    # Demand pressure and conversion signal.
    if demand_score >= 3 and served_score >= max(1, demand_score - 1):
        demand_mult = 1.08
    elif demand_score >= 2 and served_score == 0:
        demand_mult = 0.92
    elif served_score >= 2 and demand_score == 0:
        demand_mult = 1.04
    else:
        demand_mult = 1.0

    market_target = reference * scarcity * demand_mult

    # Ceil from recent realized sales; allow breakout only when very scarce.
    if hist_max > 0:
        if scarcity >= 1.15:
            ceiling = float(hist_max) * 1.20
        else:
            ceiling = float(hist_max) * 1.04
    else:
        ceiling = max(float(fallback_price) * 1.15, floor_price * 1.7)

    # Inventory flush mode on common dishes with high local stock and weak demand.
    flush_mult = 1.0
    if inventory and demand_score <= 1:
        capacities = []
        for ingredient, qty in recipe.get("ingredients", {}).items():
            try:
                need = int(qty)
            except (TypeError, ValueError):
                need = 0
            if need <= 0:
                continue
            capacities.append(int(inventory.get(ingredient, 0)) // need)
        capacity = min(capacities) if capacities else 0
        if capacity >= 4 and scarcity <= 0.95:
            flush_mult = 0.95

    raw = market_target * flush_mult
    final = max(floor_price, min(raw, ceiling))
    final_i = int(round(final))
    return min(1000, max(50, final_i))
