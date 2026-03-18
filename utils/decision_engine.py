"""Shared decision layer for menu and bidding.

This module centralizes strategy decisions so both agent prompts and
deterministic fallbacks use the same logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from utils.bid_utils import compute_smart_bids
from utils.market_utils import drought_price_multiplier, is_drought_turn
from utils.menu_utils import build_cookable_menu, build_menu_items
from utils.recipe_utils import (
    compute_cookable_recipes,
    compute_ingredient_competition,
    select_best_recipes,
)
from utils.strategy_flags import (
    rolling_revenue_budget_cap,
    strat_v2_bid_enabled,
    strat_value_bidding_enabled,
    strat_value_pricing_enabled,
)
from utils.value_bid_utils import compute_value_based_bids

if TYPE_CHECKING:
    from game_state import GameState
    from utils.contracts import BidEntry, MenuItem


@dataclass
class MenuDecision:
    selected: list[tuple[dict, list[str]]]
    competition: dict[str, int]
    menu_items: list["MenuItem"]
    cookable: list[dict]
    price_multiplier: float


@dataclass
class BiddingDecision:
    selected: list[tuple[dict, list[str]]]
    competition: dict[str, int]
    bids: list["BidEntry"]
    drought_mode: bool
    num_cookable: int
    budget_cap_override: int | None
    use_v2_bid: bool


def _common_selection(state: "GameState", max_items: int) -> tuple[list[dict], dict[str, int], list[tuple[dict, list[str]]]]:
    recipes = state.recipes or []
    competition = compute_ingredient_competition(recipes)
    selected = select_best_recipes(
        recipes,
        state.inventory,
        n=max(10, max_items * 2),
        competition=competition,
        demand=state.dish_demand,
    )
    return recipes, competition, selected


def build_speaking_menu_decision(state: "GameState", max_items: int) -> MenuDecision:
    recipes, competition, selected = _common_selection(state, max_items=max_items)
    latest_bid_prices = getattr(state, "_latest_bid_prices", None)
    price_multiplier = drought_price_multiplier(state.turn_id, state.served_last_turn)
    menu_items = build_menu_items(
        [r for r, _ in selected],
        latest_bid_prices,
        price_multiplier=price_multiplier,
        demand=state.dish_demand,
        served=state.dish_served_score,
        max_items=max_items,
        enforce_mix=True,
        competition=competition,
        market_signal_fn=state.ingredient_market_signal,
        dish_price_history=state.dish_realized_prices,
        inventory=state.inventory,
        value_pricing=strat_value_pricing_enabled(),
    )
    cookable = compute_cookable_recipes(recipes, state.inventory)
    return MenuDecision(
        selected=selected,
        competition=competition,
        menu_items=menu_items,
        cookable=cookable,
        price_multiplier=price_multiplier,
    )


def build_waiting_menu_decision(state: "GameState", max_items: int) -> MenuDecision:
    recipes, competition, selected = _common_selection(state, max_items=max_items)
    price_multiplier = drought_price_multiplier(state.turn_id, state.served_last_turn)
    cookable = compute_cookable_recipes(recipes, state.inventory)
    selected_names = {r["name"] for r, _ in selected}
    menu_items = build_cookable_menu(
        selected_names,
        cookable,
        demand=state.dish_demand,
        served=state.dish_served_score,
        inventory=state.inventory,
        max_items=max_items,
        price_multiplier=price_multiplier,
        ingredient_bid_prices=getattr(state, "_latest_bid_prices", None),
        competition=competition,
        market_signal_fn=state.ingredient_market_signal,
        dish_price_history=state.dish_realized_prices,
        value_pricing=strat_value_pricing_enabled(),
    )
    return MenuDecision(
        selected=selected,
        competition=competition,
        menu_items=menu_items,
        cookable=cookable,
        price_multiplier=price_multiplier,
    )


def build_bidding_decision(state: "GameState") -> BiddingDecision:
    recipes = state.recipes or []
    competition = compute_ingredient_competition(recipes)
    drought_mode = is_drought_turn(state.turn_id, state.served_last_turn)
    selected = select_best_recipes(
        recipes,
        state.inventory,
        n=14,
        competition=competition,
        demand=state.dish_demand,
    )
    num_cookable = sum(1 for _, missing in selected if not missing)
    use_v2_bid = strat_v2_bid_enabled()
    bid_adjustments = state.bid_adjustments() if use_v2_bid else None
    budget_cap_override = None
    if use_v2_bid:
        rolling_avg = state.rolling_revenue_avg(window=3)
        budget_cap_override = rolling_revenue_budget_cap(
            balance=state.balance,
            rolling_revenue_avg=rolling_avg,
            num_cookable=num_cookable,
            low_demand_mode=drought_mode,
        )
    archive = getattr(state, "bid_archive", None)
    if strat_value_bidding_enabled():
        bids = compute_value_based_bids(
            selected,
            state.inventory,
            state.balance,
            competition,
            menu=state.menu,
            demand_by_dish=state.dish_demand,
            served_by_dish=state.dish_served_score,
            market_signal_fn=state.ingredient_market_signal,
            low_demand_mode=drought_mode,
            budget_cap_override=budget_cap_override,
            num_cookable=num_cookable,
            bid_adjustments=bid_adjustments,
            bid_archive=archive,
        )
    else:
        bids = compute_smart_bids(
            selected,
            state.inventory,
            state.balance,
            competition,
            num_cookable,
            demand_by_dish=state.dish_demand,
            low_demand_mode=drought_mode,
            bid_adjustments=bid_adjustments,
            budget_cap_override=budget_cap_override,
            market_signal_fn=state.ingredient_market_signal,
            bid_archive=archive,
        )
    return BiddingDecision(
        selected=selected,
        competition=competition,
        bids=bids,
        drought_mode=drought_mode,
        num_cookable=num_cookable,
        budget_cap_override=budget_cap_override,
        use_v2_bid=use_v2_bid,
    )
