"""Market and drought-mode helpers."""

from __future__ import annotations

from utils.bid_utils import bid_price


def is_drought_turn(turn_id: int, served_last_turn: int) -> bool:
    """True when the previous full turn produced zero served clients."""
    return turn_id > 1 and served_last_turn == 0


def drought_price_multiplier(turn_id: int, served_last_turn: int) -> float:
    """Turn-level pricing factor: conservative after weak turns, aggressive after strong turns."""
    if turn_id <= 1:
        return 1.0
    if served_last_turn == 0:
        return 0.80
    if served_last_turn <= 2:
        return 0.90
    if served_last_turn >= 6:
        return 1.12
    if served_last_turn >= 3:
        return 1.08
    return 1.0


def build_liquidation_sales(
    inventory: dict[str, int],
    selected: list[tuple[dict, list[str]]],
    competition: dict[str, int],
    max_entries: int = 8,
) -> list[dict]:
    """Build conservative SELL listings for surplus inventory."""
    if not inventory:
        return []

    reserve: dict[str, int] = {}
    for recipe, _ in selected[:3]:
        for ingredient, qty in recipe.get("ingredients", {}).items():
            reserve[ingredient] = max(reserve.get(ingredient, 0), int(qty) * 2)

    sales: list[dict] = []
    for ingredient, qty in sorted(inventory.items(), key=lambda kv: int(kv[1]), reverse=True):
        stock = int(qty)
        keep = int(reserve.get(ingredient, 0))
        surplus = max(0, stock - keep)
        if surplus <= 0:
            continue

        if stock >= 6:
            sell_qty = max(1, int(round(surplus * 0.6)))
        else:
            sell_qty = 1 if surplus >= 2 else 0
        if sell_qty <= 0:
            continue

        unit_price = max(1, int(round(bid_price(competition.get(ingredient, 0)) * 0.75)))
        sales.append(
            {
                "ingredient": ingredient,
                "quantity": sell_qty,
                "price": unit_price * sell_qty,
            }
        )
        if len(sales) >= max_entries:
            break

    return sales
