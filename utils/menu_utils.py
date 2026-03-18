"""Menu building logic — pricing, per-dish adaptation, and lineup mix."""

from __future__ import annotations

from utils.value_price_utils import value_based_recipe_price


def recipe_price(recipe: dict, price_multiplier: float = 1.0) -> int:
    """Calculate menu price based on prestige with optional multiplier."""
    base = int(recipe.get("prestige", 50))
    scaled = int(round(base * max(0.5, price_multiplier)))
    return min(1000, max(50, scaled))


def _dish_dynamic_multiplier(
    dish_name: str,
    demand: dict[str, int] | None = None,
    served: dict[str, int] | None = None,
) -> float:
    """Per-dish pricing factor based on recent demand vs serving performance."""
    demand_scores = demand or {}
    served_scores = served or {}
    d = int(demand_scores.get(dish_name, 0))
    s = int(served_scores.get(dish_name, 0))

    if d == 0 and s == 0:
        return 1.00
    if d >= 4 and s >= max(1, d - 1):
        return 1.12
    if d >= 3 and s == 0:
        return 0.85
    if d >= 2 and s * 2 < d:
        return 0.92
    if s >= 3 and d >= 2:
        return 1.08
    if s >= 2 and d == 0:
        return 1.04
    return 1.0


def _prep_bucket(recipe: dict) -> str:
    prep_ms = int(recipe.get("preparationTimeMs", 999999))
    if prep_ms <= 7000:
        return "fast"
    if prep_ms <= 15000:
        return "medium"
    return "slow"


def _pick_menu_mix(
    candidates: list[dict],
    demand: dict[str, int] | None = None,
    max_items: int = 5,
) -> list[dict]:
    """Pick a diversified menu: 2 fast + 2 medium + 1 premium (best effort)."""
    if max_items <= 0 or not candidates:
        return []

    demand_scores = demand or {}
    used: set[str] = set()
    picked: list[dict] = []

    def dish_name(recipe: dict) -> str:
        return str(recipe.get("name", ""))

    def premium_key(recipe: dict) -> tuple[int, int, int]:
        name = dish_name(recipe)
        return (
            -int(demand_scores.get(name, 0)),
            -int(recipe.get("prestige", 0)),
            int(recipe.get("preparationTimeMs", 999999)),
        )

    def speed_key(recipe: dict) -> tuple[int, int, int]:
        name = dish_name(recipe)
        return (
            -int(demand_scores.get(name, 0)),
            int(recipe.get("preparationTimeMs", 999999)),
            -int(recipe.get("prestige", 0)),
        )

    # 1 premium anchor dish.
    for recipe in sorted(candidates, key=premium_key):
        name = dish_name(recipe)
        if not name or name in used:
            continue
        picked.append(recipe)
        used.add(name)
        break

    # Throughput dishes.
    for bucket, target in (("fast", 2), ("medium", 2)):
        count = 0
        for recipe in sorted(candidates, key=speed_key):
            name = dish_name(recipe)
            if not name or name in used or _prep_bucket(recipe) != bucket:
                continue
            picked.append(recipe)
            used.add(name)
            count += 1
            if count >= target or len(picked) >= max_items:
                break
        if len(picked) >= max_items:
            return picked[:max_items]

    # Fill remaining slots by demand then speed.
    for recipe in sorted(candidates, key=speed_key):
        name = dish_name(recipe)
        if not name or name in used:
            continue
        picked.append(recipe)
        used.add(name)
        if len(picked) >= max_items:
            break

    return picked[:max_items]


def build_menu_items(
    recipes: list[dict],
    ingredient_bid_prices: dict[str, int] | None = None,
    price_multiplier: float = 1.0,
    demand: dict[str, int] | None = None,
    served: dict[str, int] | None = None,
    max_items: int | None = None,
    enforce_mix: bool = False,
    competition: dict[str, int] | None = None,
    market_signal_fn=None,
    dish_price_history: dict[str, list[int]] | None = None,
    inventory: dict[str, int] | None = None,
    value_pricing: bool = False,
) -> list[dict]:
    """Build menu items from recipe candidates."""
    _ = ingredient_bid_prices or {}
    chosen = recipes
    if max_items is not None:
        if enforce_mix:
            chosen = _pick_menu_mix(recipes, demand=demand, max_items=max_items)
        else:
            demand_scores = demand or {}
            chosen = sorted(
                recipes,
                key=lambda r: (
                    -int(demand_scores.get(str(r.get("name", "")), 0)),
                    int(r.get("preparationTimeMs", 999999)),
                    -int(r.get("prestige", 0)),
                ),
            )[:max_items]

    items: list[dict] = []
    for recipe in chosen:
        name = str(recipe.get("name", ""))
        if not name:
            continue
        dish_mult = _dish_dynamic_multiplier(name, demand=demand, served=served)
        fallback_price = recipe_price(recipe, price_multiplier=price_multiplier * dish_mult)
        if value_pricing:
            price = value_based_recipe_price(
                recipe,
                fallback_price=fallback_price,
                dish_name=name,
                demand_score=int((demand or {}).get(name, 0)),
                served_score=int((served or {}).get(name, 0)),
                dish_price_history=dish_price_history,
                ingredient_bid_prices=ingredient_bid_prices,
                competition=competition,
                market_signal_fn=market_signal_fn,
                inventory=inventory,
            )
        else:
            price = fallback_price
        items.append({"name": name, "price": int(price)})
    return items


def build_cookable_menu(
    selected_names: set[str],
    cookable: list[dict],
    demand: dict[str, int] | None = None,
    served: dict[str, int] | None = None,
    inventory: dict[str, int] | None = None,
    max_items: int = 5,
    price_multiplier: float = 1.0,
    ingredient_bid_prices: dict[str, int] | None = None,
    competition: dict[str, int] | None = None,
    market_signal_fn=None,
    dish_price_history: dict[str, list[int]] | None = None,
    value_pricing: bool = False,
) -> list[dict]:
    """Build menu from cookable recipes, with diversified lineup and dynamic prices."""
    demand_scores = demand or {}

    def dish_capacity(recipe: dict) -> int:
        if not inventory:
            return 1
        ings = recipe.get("ingredients", {})
        if not ings:
            return 1
        caps = []
        for ing, qty in ings.items():
            if int(qty) <= 0:
                continue
            caps.append(int(inventory.get(ing, 0)) // int(qty))
        return min(caps) if caps else 1

    ranked = sorted(
        cookable,
        key=lambda r: (
            -int(demand_scores.get(str(r.get("name", "")), 0)),
            -dish_capacity(r),
            int(r.get("preparationTimeMs", 999999)),
            -int(r.get("prestige", 0)),
        ),
    )

    preferred = [r for r in ranked if str(r.get("name", "")) in selected_names]
    fallback = [r for r in ranked if str(r.get("name", "")) not in selected_names]

    chosen = _pick_menu_mix(preferred, demand=demand, max_items=max_items)
    if len(chosen) < max_items:
        used = {str(r.get("name", "")) for r in chosen}
        for r in _pick_menu_mix(fallback, demand=demand, max_items=max_items):
            name = str(r.get("name", ""))
            if not name or name in used:
                continue
            chosen.append(r)
            used.add(name)
            if len(chosen) >= max_items:
                break

    items: list[dict] = []
    for recipe in chosen[:max_items]:
        name = str(recipe.get("name", ""))
        if not name:
            continue
        dish_mult = _dish_dynamic_multiplier(name, demand=demand, served=served)
        fallback_price = recipe_price(recipe, price_multiplier=price_multiplier * dish_mult)
        if value_pricing:
            price = value_based_recipe_price(
                recipe,
                fallback_price=fallback_price,
                dish_name=name,
                demand_score=int((demand or {}).get(name, 0)),
                served_score=int((served or {}).get(name, 0)),
                dish_price_history=dish_price_history,
                ingredient_bid_prices=ingredient_bid_prices,
                competition=competition,
                market_signal_fn=market_signal_fn,
                inventory=inventory,
            )
        else:
            price = fallback_price
        items.append(
            {
                "name": name,
                "price": price,
            }
        )
    return items
