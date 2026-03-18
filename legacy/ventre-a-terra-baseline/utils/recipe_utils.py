"""Recipe selection logic — dynamic, based on inventory and available recipes."""

from __future__ import annotations

from utils.bid_utils import bid_price


def compute_ingredient_competition(recipes: list[dict]) -> dict[str, int]:
    """Count how many recipes use each ingredient."""
    freq: dict[str, int] = {}
    for r in recipes:
        for ing in r.get("ingredients", {}):
            freq[ing] = freq.get(ing, 0) + 1
    return freq


def _estimate_bid_cost(
    missing: list[str],
    competition: dict[str, int],
) -> int:
    """Estimate total bid cost for a list of missing ingredients."""
    return sum(bid_price(competition.get(ing, 0)) for ing in missing)


def select_best_recipes(
    recipes: list[dict],
    inventory: dict,
    n: int = 5,
    competition: dict[str, int] | None = None,
    demand: dict[str, int] | None = None,
) -> list[tuple[dict, list[str]]]:
    """Pick a balanced recipe set for service throughput + value.

    Strategy:
    - Prefer cookable "core" recipes (fast/easy) for throughput.
    - Add cookable premium recipes for margin/prestige.
    - Fill remaining slots with aspirational recipes by ROI.
    - Return up to n recipes total

    Returns list of (recipe, [missing_ingredients]).
    """
    comp = competition or {}
    demand_scores = demand or {}

    cookable = []
    need_bids = []

    for r in recipes:
        ings = r.get("ingredients", {})
        missing = [ing for ing, qty in ings.items() if inventory.get(ing, 0) < qty]
        if not missing:
            cookable.append((r, missing))
        else:
            need_bids.append((r, missing))

    def core_key(item: tuple[dict, list[str]]) -> tuple[int, int, int, int]:
        r, _ = item
        dish_demand = int(demand_scores.get(str(r.get("name", "")), 0))
        prep = int(r.get("preparationTimeMs", 999999))
        ing_count = len(r.get("ingredients", {}))
        prestige = int(r.get("prestige", 0))
        return (-dish_demand, prep, ing_count, -prestige)

    def premium_key(item: tuple[dict, list[str]]) -> tuple[int, int, int, int]:
        r, _ = item
        dish_demand = int(demand_scores.get(str(r.get("name", "")), 0))
        prep = int(r.get("preparationTimeMs", 999999))
        ing_count = len(r.get("ingredients", {}))
        prestige = int(r.get("prestige", 0))
        return (-dish_demand, -prestige, prep, ing_count)

    # Sort aspirational recipes by ROI descending.
    def aspirational_key(item: tuple[dict, list[str]]) -> tuple[float, int, int]:
        recipe, missing = item
        prestige = int(recipe.get("prestige", 0))
        dish_demand = int(demand_scores.get(str(recipe.get("name", "")), 0))
        demand_factor = 1.0 + (0.35 * dish_demand)
        roi = (prestige * demand_factor) / (1 + _estimate_bid_cost(missing, comp))
        return (-roi, -dish_demand, -prestige)

    need_bids.sort(key=aspirational_key)

    result: list[tuple[dict, list[str]]] = []
    chosen_names: set[str] = set()

    # 3 "core" slots for fast, reliable dishes.
    core_slots = min(3, n)
    for item in sorted(cookable, key=core_key):
        if len(result) >= core_slots:
            break
        r, _ = item
        name = r.get("name")
        if not name or name in chosen_names:
            continue
        result.append(item)
        chosen_names.add(name)

    # Fill remaining cookable slots with premium dishes.
    if len(result) < n:
        for item in sorted(cookable, key=premium_key):
            if len(result) >= n:
                break
            r, _ = item
            name = r.get("name")
            if not name or name in chosen_names:
                continue
            result.append(item)
            chosen_names.add(name)

    # If still not enough, include aspirational recipes to guide bidding.
    if len(result) < n:
        for item in need_bids:
            if len(result) >= n:
                break
            r, _ = item
            name = r.get("name")
            if not name or name in chosen_names:
                continue
            result.append(item)
            chosen_names.add(name)

    return result


def compute_cookable_recipes(
    recipes: list[dict],
    inventory: dict,
) -> list[dict]:
    """Return recipes we can cook right now (have all ingredients)."""
    return [
        r for r in recipes
        if all(inventory.get(ing, 0) >= qty for ing, qty in r.get("ingredients", {}).items())
    ]
