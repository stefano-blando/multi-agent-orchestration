"""Value-based bidding strategy — simple profit-margin approach."""

from __future__ import annotations

import logging

from utils.bid_utils import bid_price
from utils.menu_utils import recipe_price

log = logging.getLogger(__name__)

_MAX_TARGETS = 7


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


def _menu_price_lookup(menu: list[dict] | dict | None) -> dict[str, int]:
    if isinstance(menu, dict):
        items = menu.get("items", []) or []
    elif isinstance(menu, list):
        items = menu
    else:
        items = []
    out: dict[str, int] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        try:
            out[name] = int(item.get("price", 0))
        except (TypeError, ValueError):
            continue
    return out


def _ingredient_ref_price(
    ingredient: str,
    competition: dict[str, int],
    market_signal_fn=None,
    bid_archive=None,
) -> float:
    """Best estimate of what we need to bid per unit for this ingredient.

    Priority:
    1. Recent short-horizon market average (avg_win_2), adjusted by recent trend.
    2. Mid-horizon bid archive average clearing price, adjusted by archive trend.
    3. Long-horizon EMA from price_db (ema_short), adjusted by long-term trend.
    4. Competition-based heuristic (bid_price) as last resort.
    """
    sig = _safe_signal(market_signal_fn, ingredient)

    if sig["avg_win_2"] > 0:
        # Short-horizon: use recent market clearing price, nudged by trend.
        trend = max(0.8, min(1.5, sig["trend"]))
        return sig["avg_win_2"] * trend

    # Mid-horizon: bid archive average clearing price.
    if bid_archive is not None:
        avg = bid_archive.avg_clearing_price(ingredient)
        if avg > 0:
            trend_label = bid_archive.price_trend(ingredient)
            trend_mult = {"rising": 1.08, "falling": 0.93, "stable": 1.0}.get(trend_label, 1.0)
            return avg * trend_mult

    # Long-horizon: use EMA from price_db if we have enough observations.
    ema_short = sig.get("ema_short", 0.0)
    n_obs = sig.get("n_obs", 0.0)
    if ema_short > 0 and n_obs >= 3:
        long_trend = max(0.85, min(1.3, sig.get("long_trend", 1.0)))
        return ema_short * long_trend

    # No market data: use competition-based heuristic.
    return float(bid_price(competition.get(ingredient, 0)))


def compute_value_based_bids(
    selected: list[tuple[dict, list[str]]],
    inventory: dict[str, int],
    balance: float,
    competition: dict[str, int],
    menu: list[dict] | dict | None = None,
    demand_by_dish: dict[str, int] | None = None,
    served_by_dish: dict[str, int] | None = None,
    market_signal_fn=None,
    low_demand_mode: bool = False,
    budget_cap_override: int | float | None = None,
    num_cookable: int = 0,
    bid_adjustments: dict[str, int] | None = None,
    bid_archive=None,
) -> list[dict]:
    """Simple profit-margin bidding.

    For each target recipe:
      menu_price = what we sell it for
      owned_cost = estimated cost of ingredients we already have
      profit_budget = menu_price - owned_cost
      max_bid_per_missing = profit_budget * 0.70 / num_missing

    Prioritize recipes with fewer missing ingredients (easier to complete).
    Target up to 7 recipes. Budget-capped.
    """
    demand = demand_by_dish or {}
    menu_prices = _menu_price_lookup(menu)
    adjustments = bid_adjustments or {}

    # -- Budget cap --
    if low_demand_mode:
        if num_cookable >= 2:
            cap_pct = 0.16
        elif num_cookable >= 1:
            cap_pct = 0.22
        else:
            cap_pct = 0.30
        cap_abs = 900.0
    else:
        if num_cookable >= 2:
            cap_pct = 0.26
        elif num_cookable >= 1:
            cap_pct = 0.34
        else:
            cap_pct = 0.46
        cap_abs = 1800.0

    budget_cap = min(float(balance) * cap_pct, cap_abs)
    if budget_cap_override is not None:
        budget_cap = min(budget_cap, float(max(0, budget_cap_override)))
    if budget_cap <= 0:
        return []

    # -- Rank target recipes --
    # Sort by: fewest missing ingredients first, then by prestige desc, then fast prep.
    targets: list[tuple[dict, list[str]]] = []
    for recipe, missing in selected:
        if not missing:
            continue
        targets.append((recipe, missing))

    targets.sort(key=lambda x: (
        len(x[1]),                                           # fewer missing = easier to complete
        -int(demand.get(str(x[0].get("name", "")), 0)),     # higher demand
        -int(x[0].get("prestige", 0)),                       # higher prestige
        int(x[0].get("preparationTimeMs", 999999)),          # faster prep
    ))
    targets = targets[:_MAX_TARGETS]

    if not targets:
        return []

    # -- Calculate max bid per ingredient per recipe --
    # For each recipe: how much can we afford to pay per missing ingredient?
    ingredient_max_bid: dict[str, float] = {}
    ingredient_qty: dict[str, int] = {}

    for recipe, missing in targets:
        name = str(recipe.get("name", ""))
        menu_price = float(menu_prices.get(name, recipe_price(recipe)))
        num_missing = len(missing)

        # Estimate cost of ingredients we already own (sunk cost — not relevant to bid,
        # but the profit margin on selling the dish is what matters).
        # profit_budget = what we can afford to spend on missing ingredients
        # and still make a profit when we sell the dish.
        profit_budget = menu_price * 0.70  # keep 30% margin

        # Spread budget across missing ingredients.
        max_per_missing = profit_budget / max(1, num_missing)

        # Boost for recipes with only 1-2 missing ingredients (high completion probability).
        if num_missing == 1:
            max_per_missing *= 1.3
        elif num_missing == 2:
            max_per_missing *= 1.1

        for ing in missing:
            current = ingredient_max_bid.get(ing, 0.0)
            # Take the highest willingness-to-pay across all recipes that need this.
            ingredient_max_bid[ing] = max(current, max_per_missing)
            ingredient_qty[ing] = ingredient_qty.get(ing, 0) + 1

    # -- Build actual bids --
    total = 0.0
    bids: list[dict] = []

    # Sort ingredients: most-needed first (shared across recipes), then highest value.
    sorted_ingredients = sorted(
        ingredient_max_bid.keys(),
        key=lambda ing: (-ingredient_qty.get(ing, 0), -ingredient_max_bid.get(ing, 0.0)),
    )

    for ing in sorted_ingredients:
        max_affordable = ingredient_max_bid[ing]

        # Reference price: what the market typically clears at.
        ref = _ingredient_ref_price(ing, competition, market_signal_fn, bid_archive)

        # Bid = min(what we can afford, market reference * 1.1).
        # The 1.1 multiplier gives us a slight edge over average bidders.
        bid_unit = int(round(min(max_affordable, ref * 1.10)))

        # Apply historical win-rate adjustments.
        bid_unit += int(adjustments.get(ing, 0))

        # Floor: at least 5 per unit to be competitive.
        bid_unit = max(5, bid_unit)

        # Ceiling: never bid more than what we can profit from.
        bid_unit = min(bid_unit, int(max_affordable))
        if bid_unit <= 0:
            continue

        # Quantity: 1 normally, 2 if shared across 3+ recipes or high demand.
        qty = 1
        if not low_demand_mode and ingredient_qty.get(ing, 0) >= 3:
            qty = 2
        elif not low_demand_mode and ingredient_qty.get(ing, 0) >= 2 and bid_unit <= int(max_affordable * 0.6):
            qty = 2

        # Budget check.
        cost = float(bid_unit * qty)
        if total + cost > budget_cap:
            if qty > 1 and total + float(bid_unit) <= budget_cap:
                qty = 1
                cost = float(bid_unit)
            else:
                continue

        bids.append({"ingredient": ing, "bid": bid_unit, "quantity": qty})
        total += cost

    log.info(
        "Value bids: %d ingredients, total=%d, budget_cap=%d, targets=%d recipes",
        len(bids), int(total), int(budget_cap), len(targets),
    )
    return bids
