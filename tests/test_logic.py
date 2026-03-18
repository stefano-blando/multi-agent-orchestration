"""Unit tests for game logic — no server, no LLM."""

import json
import sys
import os

# Allow imports from app/
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "app"))

from game_state import GameState
from phases import serving
from utils.bid_utils import bid_price, compute_smart_bids
from utils.menu_utils import recipe_price, build_menu_items, build_cookable_menu
from utils.recipe_utils import (
    compute_cookable_recipes,
    compute_ingredient_competition,
    select_best_recipes,
)
from utils.market_utils import build_liquidation_sales
from utils.serving_utils import extract_intolerances, dish_has_intolerance
from utils.decision_engine import build_speaking_menu_decision, build_bidding_decision
from utils.serving_engine import build_serving_plan
from utils.value_bid_utils import compute_value_based_bids
from utils.bid_archive import BidArchive
import tools as tools_mod

# ── Sample data ────────────────────────────────────────────────────────

RECIPES = [
    {"name": "Pasta Stellare", "prestige": 80, "ingredients": {"grano": 2, "sale": 1}, "preparationTimeMs": 3000},
    {"name": "Zuppa Cosmica", "prestige": 60, "ingredients": {"acqua": 3, "sale": 1}, "preparationTimeMs": 2000},
    {"name": "Risotto Galattico", "prestige": 120, "ingredients": {"riso": 2, "burro": 1, "zafferano": 1}, "preparationTimeMs": 5000},
    {"name": "Insalata Nebulosa", "prestige": 40, "ingredients": {"lattuga": 1}, "preparationTimeMs": 1000},
]

FULL_INVENTORY = {"grano": 5, "sale": 5, "acqua": 5, "riso": 5, "burro": 5, "zafferano": 5, "lattuga": 5}
PARTIAL_INVENTORY = {"grano": 2, "sale": 1, "lattuga": 1}


# ── turn_id detection ──────────────────────────────────────────────────

class TestTurnIdDetection:
    def test_turn_id_from_restaurant_data_snake(self):
        state = GameState()
        assert state.turn_id == 0
        state.update_restaurant({"balance": 100, "inventory": {}, "turn_id": 3})
        assert state.turn_id == 3

    def test_turn_id_from_restaurant_data_camel(self):
        state = GameState()
        state.update_restaurant({"balance": 100, "inventory": {}, "turnId": 5})
        assert state.turn_id == 5

    def test_turn_id_not_overwritten_if_already_set(self):
        state = GameState()
        state.turn_id = 7
        state.update_restaurant({"balance": 100, "inventory": {}, "turn_id": 2})
        assert state.turn_id == 7

    def test_turn_id_zero_stays_if_no_data(self):
        state = GameState()
        state.update_restaurant({"balance": 100, "inventory": {}})
        assert state.turn_id == 0

    def test_turn_id_from_game_started(self):
        state = GameState()
        state.on_game_started({"turn_id": 4})
        assert state.turn_id == 4

    def test_turn_id_increments_on_missing(self):
        state = GameState()
        state.turn_id = 3
        state.on_game_started({})
        assert state.turn_id == 4


# ── Serving build_prompt (indentation bug) ─────────────────────────────

class TestServingPrompt:
    def test_all_dishes_listed(self):
        state = GameState()
        state.turn_id = 1
        state.balance = 500
        state.inventory = FULL_INVENTORY.copy()
        state.recipes = RECIPES
        state.on_client_spawned({"clientId": "1", "clientName": "Test", "orderText": "anything"})

        prompt = serving.build_prompt(state)
        # All 4 recipes should be listed (we have all ingredients)
        for r in RECIPES:
            assert r["name"] in prompt, f"{r['name']} not found in serving prompt"

    def test_only_cookable_listed(self):
        state = GameState()
        state.turn_id = 1
        state.balance = 500
        state.inventory = PARTIAL_INVENTORY.copy()
        state.recipes = RECIPES
        state.on_client_spawned({"clientId": "1", "clientName": "Test", "orderText": "anything"})

        prompt = serving.build_prompt(state)
        assert "Pasta Stellare" in prompt
        assert "Insalata Nebulosa" in prompt
        # Can't cook these
        assert "Risotto Galattico" not in prompt
        assert "Zuppa Cosmica" not in prompt

    def test_turn_id_zero_returns_error(self):
        state = GameState()
        state.turn_id = 0
        state.balance = 500
        state.inventory = FULL_INVENTORY.copy()
        state.recipes = RECIPES
        state.on_client_spawned({"clientId": "1", "clientName": "Test", "orderText": "anything"})

        prompt = serving.build_prompt(state)
        assert "end_phase" in prompt.lower() or "error" in prompt.lower()

    def test_available_dishes_section_is_truncated(self):
        state = GameState()
        state.turn_id = 1
        state.balance = 500
        state.inventory = {"base": 200}
        state.recipes = [
            {
                "name": f"Dish {i}",
                "prestige": 60 + i,
                "ingredients": {"base": 1},
                "preparationTimeMs": 1000 + i,
            }
            for i in range(30)
        ]
        state.on_client_spawned({"clientId": "1", "clientName": "Test", "orderText": "anything"})
        prompt = serving.build_prompt(state)
        assert "more cookable dishes" in prompt

    def test_pending_clients_section_is_truncated(self):
        state = GameState()
        state.turn_id = 1
        state.balance = 500
        state.inventory = FULL_INVENTORY.copy()
        state.recipes = RECIPES
        for i in range(20):
            state.on_client_spawned(
                {"clientId": str(i), "clientName": f"Test{i}", "orderText": "anything"}
            )
        prompt = serving.build_prompt(state)
        assert "more pending clients" in prompt


# ── Error handling (prepare_dish / serve_dish JSON parse) ──────────────

class TestErrorParsing:
    def test_error_json_detected(self):
        error_resp = json.dumps({"error": "dish not found"})
        parsed = json.loads(error_resp)
        assert parsed.get("error")

    def test_success_json_not_error(self):
        ok_resp = json.dumps({"status": "ok", "dish": "Error Handler Soup"})
        parsed = json.loads(ok_resp)
        assert not parsed.get("error")


class TestIngredientBasedRequestMatching:
    def test_extract_requested_ingredients_from_text(self):
        state = GameState()
        state.recipes = [
            {"name": "Dish A", "ingredients": {"Plasma Vitale": 1, "Uova di Fenice": 1}},
            {"name": "Dish B", "ingredients": {"Essenza di Tachioni": 1}},
        ]
        tools_mod._state = state
        req = "I want something with Plasma Vitale and Uova di Fenice"
        found = tools_mod._requested_ingredients_from_text(req)
        assert "plasma vitale" in found
        assert "uova di fenice" in found

    def test_dish_matches_ingredient_based_request(self):
        state = GameState()
        state.recipes = [
            {
                "name": "Portale Cosmico",
                "ingredients": {
                    "Shard di Materia Oscura": 1,
                    "Uova di Fenice": 1,
                    "Gnocchi del Crepuscolo": 1,
                    "Plasma Vitale": 1,
                    "Essenza di Tachioni": 1,
                },
            },
        ]
        tools_mod._state = state
        req = (
            "I want something with Shard di Materia Oscura, Uova di Fenice, "
            "Gnocchi del Crepuscolo, Plasma Vitale, and Essenza di Tachioni"
        )
        assert tools_mod._dish_matches_request("Portale Cosmico", req)

    def test_dish_name_with_error_substring(self):
        """Old code did `'error' not in resp` — would false-positive on dish names containing 'error'."""
        ok_resp = json.dumps({"status": "ok", "dish": "Terrorific Pasta"})
        # Old check would fail:
        assert "error" in ok_resp  # substring match finds it
        # New check works:
        parsed = json.loads(ok_resp)
        assert not parsed.get("error")


# ── Bid strategy ───────────────────────────────────────────────────────

class TestBidStrategy:
    def test_bid_price_tiers(self):
        assert bid_price(5) == 10
        assert bid_price(19) == 10
        assert bid_price(20) == 15
        assert bid_price(34) == 15
        assert bid_price(35) == 20
        assert bid_price(49) == 20
        assert bid_price(50) == 25
        assert bid_price(100) == 25

    def test_no_bids_when_all_in_inventory(self):
        competition = compute_ingredient_competition(RECIPES)
        selected = select_best_recipes(RECIPES, FULL_INVENTORY, n=3, competition=competition)
        bids = compute_smart_bids(selected, FULL_INVENTORY, 1000, competition)
        assert bids == []

    def test_bids_for_missing_ingredients(self):
        competition = compute_ingredient_competition(RECIPES)
        selected = select_best_recipes(RECIPES, PARTIAL_INVENTORY, n=3, competition=competition)
        bids = compute_smart_bids(selected, PARTIAL_INVENTORY, 1000, competition)
        for b in bids:
            assert b["ingredient"] not in PARTIAL_INVENTORY or PARTIAL_INVENTORY[b["ingredient"]] < 1

    def test_budget_cap_respected(self):
        competition = compute_ingredient_competition(RECIPES)
        selected = select_best_recipes(RECIPES, {}, n=4, competition=competition)
        bids = compute_smart_bids(selected, {}, 100, competition)
        total = sum(b["bid"] * b["quantity"] for b in bids)
        cap = min(100 * 0.55, 1500)  # 0 cookable → 55% cap
        assert total <= cap

    def test_top_recipe_gets_quantity_2(self):
        competition = compute_ingredient_competition(RECIPES)
        selected = select_best_recipes(RECIPES, {}, n=4, competition=competition)
        bids = compute_smart_bids(selected, {}, 5000, competition, num_cookable=0)
        # At least one bid should have quantity 2 (top prestige recipe)
        quantities = [b["quantity"] for b in bids]
        assert 2 in quantities or not bids  # if budget allows

    def test_less_aggressive_when_cookable(self):
        competition = compute_ingredient_competition(RECIPES)
        selected = select_best_recipes(RECIPES, {}, n=4, competition=competition)
        bids_aggressive = compute_smart_bids(selected, {}, 500, competition, num_cookable=0)
        bids_conservative = compute_smart_bids(selected, {}, 500, competition, num_cookable=3)
        total_agg = sum(b["bid"] * b["quantity"] for b in bids_aggressive)
        total_con = sum(b["bid"] * b["quantity"] for b in bids_conservative)
        assert total_con <= total_agg

    def test_low_demand_mode_is_more_conservative(self):
        competition = compute_ingredient_competition(RECIPES)
        selected = select_best_recipes(RECIPES, {}, n=4, competition=competition)
        normal = compute_smart_bids(selected, {}, 1000, competition, num_cookable=0, low_demand_mode=False)
        drought = compute_smart_bids(selected, {}, 1000, competition, num_cookable=0, low_demand_mode=True)
        total_normal = sum(b["bid"] * b["quantity"] for b in normal)
        total_drought = sum(b["bid"] * b["quantity"] for b in drought)
        assert total_drought <= total_normal

    def test_bid_adjustment_changes_price(self):
        competition = compute_ingredient_competition(RECIPES)
        selected = select_best_recipes(RECIPES, {}, n=4, competition=competition)
        normal = compute_smart_bids(selected, {}, 1000, competition, num_cookable=0)
        adjusted = compute_smart_bids(
            selected,
            {},
            1000,
            competition,
            num_cookable=0,
            bid_adjustments={"riso": 3},
        )
        normal_map = {b["ingredient"]: b["bid"] for b in normal}
        adjusted_map = {b["ingredient"]: b["bid"] for b in adjusted}
        if "riso" in normal_map and "riso" in adjusted_map:
            assert adjusted_map["riso"] >= normal_map["riso"] + 1

    def test_budget_cap_override_respected(self):
        competition = compute_ingredient_competition(RECIPES)
        selected = select_best_recipes(RECIPES, {}, n=4, competition=competition)
        bids = compute_smart_bids(
            selected,
            {},
            1000,
            competition,
            num_cookable=0,
            budget_cap_override=90,
        )
        total = sum(b["bid"] * b["quantity"] for b in bids)
        assert total <= 90


# ── Menu strategy ──────────────────────────────────────────────────────

class TestMenuStrategy:
    def test_price_clamp_lower(self):
        assert recipe_price({"prestige": 10}) == 50

    def test_price_clamp_upper(self):
        assert recipe_price({"prestige": 2000}) == 1000

    def test_price_normal(self):
        assert recipe_price({"prestige": 80}) == 80

    def test_price_missing_prestige(self):
        assert recipe_price({}) == 50

    def test_price_multiplier_lowers_price(self):
        assert recipe_price({"prestige": 100}, price_multiplier=0.88) == 88

    def test_build_menu_items(self):
        items = build_menu_items(RECIPES)
        assert len(items) == len(RECIPES)
        for item in items:
            assert 50 <= item["price"] <= 1000
            assert "name" in item

    def test_build_menu_items_enforce_mix_max_items(self):
        items = build_menu_items(RECIPES, max_items=3, enforce_mix=True)
        assert len(items) <= 3
        assert len({i["name"] for i in items}) == len(items)

    def test_demand_served_adjusts_price_per_dish(self):
        base = build_menu_items([RECIPES[0]], price_multiplier=1.0)[0]["price"]
        # High demand with poor serving should push price down.
        adjusted = build_menu_items(
            [RECIPES[0]],
            price_multiplier=1.0,
            demand={RECIPES[0]["name"]: 4},
            served={RECIPES[0]["name"]: 0},
        )[0]["price"]
        assert adjusted < base

    def test_build_cookable_menu(self):
        cookable = compute_cookable_recipes(RECIPES, PARTIAL_INVENTORY)
        selected_names = {r["name"] for r in RECIPES}
        menu = build_cookable_menu(selected_names, cookable)
        # Only cookable ones
        names = {m["name"] for m in menu}
        assert "Pasta Stellare" in names
        assert "Insalata Nebulosa" in names
        assert "Risotto Galattico" not in names


# ── Recipe utils ───────────────────────────────────────────────────────

class TestRecipeUtils:
    def test_cookable_full_inventory(self):
        cookable = compute_cookable_recipes(RECIPES, FULL_INVENTORY)
        assert len(cookable) == len(RECIPES)

    def test_cookable_empty_inventory(self):
        cookable = compute_cookable_recipes(RECIPES, {})
        assert len(cookable) == 0

    def test_select_best_cookable_first(self):
        competition = compute_ingredient_competition(RECIPES)
        selected = select_best_recipes(RECIPES, PARTIAL_INVENTORY, n=4, competition=competition)
        # Cookable recipes (0 missing) should come first
        found_non_cookable = False
        for r, missing in selected:
            if missing:
                found_non_cookable = True
            elif found_non_cookable:
                assert False, "Cookable recipe after non-cookable"

    def test_select_best_roi_ordering(self):
        competition = compute_ingredient_competition(RECIPES)
        selected = select_best_recipes(RECIPES, {}, n=4, competition=competition)
        # With empty inventory, all need bids. Should be ordered by ROI.
        assert len(selected) == 4
        # Highest prestige recipe should be first (Risotto Galattico P=120)
        assert selected[0][0]["name"] == "Risotto Galattico"

    def test_select_best_prefers_only_cookable_when_enough(self):
        competition = compute_ingredient_competition(RECIPES)
        selected = select_best_recipes(RECIPES, FULL_INVENTORY, n=3, competition=competition)
        assert len(selected) == 3
        assert all(not missing for _, missing in selected)

    def test_competition_counts(self):
        comp = compute_ingredient_competition(RECIPES)
        # "sale" is in Pasta Stellare and Zuppa Cosmica
        assert comp["sale"] == 2
        assert comp["lattuga"] == 1


# ── Intolerance handling ───────────────────────────────────────────────

class TestIntolerances:
    def test_extract_english(self):
        assert "sale" in extract_intolerances("I'm intolerant to sale")

    def test_extract_italian(self):
        assert "grano" in extract_intolerances("Sono intollerante al grano")

    def test_extract_allergic(self):
        assert "burro" in extract_intolerances("I'm allergic to burro")

    def test_extract_multiple(self):
        result = extract_intolerances("I can't eat grano and sale")
        assert "grano" in result
        assert "sale" in result

    def test_extract_none(self):
        assert extract_intolerances("I'd like something cosmic") == set()

    def test_extract_no_false_positive_generic_no(self):
        assert extract_intolerances("No onions please") == set()

    def test_dish_has_intolerance_match(self):
        recipe = {"name": "Pasta", "ingredients": {"grano": 2, "sale": 1}}
        assert dish_has_intolerance(recipe, {"grano"})

    def test_dish_has_intolerance_no_match(self):
        recipe = {"name": "Pasta", "ingredients": {"grano": 2, "sale": 1}}
        assert not dish_has_intolerance(recipe, {"burro"})

    def test_dish_has_intolerance_partial_match(self):
        """Intolerance 'grano' should match ingredient 'grano cosmico'."""
        recipe = {"name": "X", "ingredients": {"grano cosmico": 2}}
        assert dish_has_intolerance(recipe, {"grano"})

    def test_dish_has_intolerance_avoids_substring_false_positive(self):
        recipe = {"name": "X", "ingredients": {"salsa piccante": 1}}
        assert not dish_has_intolerance(recipe, {"sale"})

    def test_dish_has_intolerance_empty(self):
        recipe = {"name": "X", "ingredients": {"grano": 1}}
        assert not dish_has_intolerance(recipe, set())


# ── Architecture helpers ──────────────────────────────────────────────

class TestDecisionLayer:
    def test_speaking_decision_builds_menu(self):
        state = GameState()
        state.turn_id = 2
        state.recipes = RECIPES
        state.inventory = FULL_INVENTORY.copy()
        decision = build_speaking_menu_decision(state, max_items=5)
        assert decision.menu_items
        assert len(decision.menu_items) <= 5

    def test_bidding_decision_returns_list(self):
        state = GameState()
        state.turn_id = 2
        state.recipes = RECIPES
        state.inventory = {}
        state.balance = 500
        decision = build_bidding_decision(state)
        assert isinstance(decision.bids, list)

    def test_value_bids_more_aggressive_when_no_cookable(self):
        competition = compute_ingredient_competition(RECIPES)
        selected = select_best_recipes(RECIPES, {}, n=4, competition=competition)
        menu = [{"name": r["name"], "price": int(r.get("prestige", 50))} for r in RECIPES]

        no_cookable = compute_value_based_bids(
            selected,
            {},
            1500,
            competition,
            menu=menu,
            num_cookable=0,
            low_demand_mode=False,
        )
        some_cookable = compute_value_based_bids(
            selected,
            {},
            1500,
            competition,
            menu=menu,
            num_cookable=2,
            low_demand_mode=False,
        )
        total_no = sum(int(b["bid"]) * int(b["quantity"]) for b in no_cookable)
        total_some = sum(int(b["bid"]) * int(b["quantity"]) for b in some_cookable)
        assert total_no >= total_some


class TestServingPlanner:
    def test_plan_marks_uncookable_as_skip(self):
        state = GameState()
        state.recipes = RECIPES
        state.inventory = {"lattuga": 1}  # only Insalata is cookable
        meals = [
            {
                "id": 10,
                "customerId": 99,
                "status": "waiting",
                "executed": False,
                "request": "Vorrei Risotto Galattico",
                "startTime": "2026-03-01T00:00:00Z",
            }
        ]
        plan = build_serving_plan(state, meals, RECIPES)
        assert len(plan) == 1
        assert plan[0].action == "skip"

    def test_plan_respects_intolerance(self):
        state = GameState()
        state.recipes = RECIPES
        state.inventory = FULL_INVENTORY.copy()
        meals = [
            {
                "id": 11,
                "customerId": 101,
                "status": "waiting",
                "executed": False,
                "request": "I want Pasta Stellare. I'm intolerant to sale",
                "startTime": "2026-03-01T00:00:00Z",
            }
        ]
        plan = build_serving_plan(state, meals, RECIPES)
        assert len(plan) == 1
        assert plan[0].action == "skip"

    def test_plan_deducts_inventory_between_dishes(self):
        """After serving one Pasta Stellare (needs grano=2, sale=1),
        a second one should be skipped if inventory is exhausted."""
        state = GameState()
        state.recipes = RECIPES
        state.inventory = {"grano": 2, "sale": 1}  # enough for exactly 1 Pasta Stellare
        meals = [
            {
                "id": 20,
                "customerId": 200,
                "status": "waiting",
                "executed": False,
                "request": "Vorrei Pasta Stellare",
                "startTime": "2026-03-01T00:00:01Z",
            },
            {
                "id": 21,
                "customerId": 201,
                "status": "waiting",
                "executed": False,
                "request": "Vorrei Pasta Stellare",
                "startTime": "2026-03-01T00:00:02Z",
            },
        ]
        plan = build_serving_plan(state, meals, RECIPES)
        serve_count = sum(1 for t in plan if t.action == "serve")
        skip_count = sum(1 for t in plan if t.action == "skip")
        assert serve_count == 1, f"Expected 1 serve, got {serve_count}"
        assert skip_count == 1, f"Expected 1 skip, got {skip_count}"

    def test_plan_prioritizes_high_prestige(self):
        """With limited inventory, high prestige dishes should be served first."""
        state = GameState()
        state.recipes = RECIPES
        # Enough for 1 of any recipe needing sale (Pasta P=80 or Zuppa P=60)
        state.inventory = {"grano": 2, "sale": 1, "acqua": 3}
        meals = [
            {
                "id": 30,
                "customerId": 300,
                "status": "waiting",
                "executed": False,
                "request": "Vorrei Zuppa Cosmica",  # P=60
                "startTime": "2026-03-01T00:00:01Z",  # arrived first
            },
            {
                "id": 31,
                "customerId": 301,
                "status": "waiting",
                "executed": False,
                "request": "Vorrei Pasta Stellare",  # P=80
                "startTime": "2026-03-01T00:00:02Z",  # arrived second
            },
        ]
        plan = build_serving_plan(state, meals, RECIPES)
        served = [t for t in plan if t.action == "serve"]
        assert len(served) >= 1
        # Pasta Stellare (P=80) should be served despite arriving second
        assert served[0].dish_to_serve == "Pasta Stellare"


class TestLiquidationSales:
    def test_surplus_generates_sales(self):
        inventory = {"grano": 10, "sale": 8, "acqua": 2}
        competition = compute_ingredient_competition(RECIPES)
        selected = select_best_recipes(RECIPES, inventory, n=3, competition=competition)
        sales = build_liquidation_sales(inventory, selected, competition)
        # Should sell surplus grano and sale (high stock), not acqua (low stock)
        sold_ingredients = {s["ingredient"] for s in sales}
        assert "grano" in sold_ingredients or "sale" in sold_ingredients
        for s in sales:
            assert s["quantity"] > 0
            assert s["price"] > 0

    def test_no_sales_when_low_stock(self):
        inventory = {"grano": 2, "sale": 1}
        competition = compute_ingredient_competition(RECIPES)
        selected = select_best_recipes(RECIPES, inventory, n=3, competition=competition)
        sales = build_liquidation_sales(inventory, selected, competition)
        assert sales == []

    def test_reserves_ingredients_for_selected(self):
        inventory = {"grano": 5, "sale": 5, "lattuga": 1}
        competition = compute_ingredient_competition(RECIPES)
        selected = select_best_recipes(RECIPES, inventory, n=3, competition=competition)
        sales = build_liquidation_sales(inventory, selected, competition)
        # Should not sell all of any ingredient needed for selected recipes
        for s in sales:
            remaining = inventory[s["ingredient"]] - s["quantity"]
            assert remaining >= 0


class TestStateSnapshot:
    def test_snapshot_roundtrip(self):
        state = GameState()
        state.dish_demand = {"A": 3}
        state.dish_served_score = {"A": 2}
        state.dish_realized_prices = {"A": [70, 80]}
        snap = state.to_snapshot()
        fresh = GameState()
        fresh.apply_snapshot(snap)
        assert fresh.dish_demand.get("A") == 3
        assert fresh.dish_served_score.get("A") == 2
        assert fresh.dish_realized_prices.get("A") == [70, 80]


class TestBidHistoryIngestion:
    def test_bid_history_updates_market_signal(self):
        state = GameState()
        entries_t1 = [
            {"ingredient": "Sale Temporale", "bid": 15, "quantity": 2, "restaurantId": 19, "wonQuantity": 2},
            {"ingredient": "Sale Temporale", "bid": 10, "quantity": 3, "restaurantId": 5, "wonQuantity": 3},
        ]
        entries_t2 = [
            {"ingredient": "Sale Temporale", "bid": 20, "quantity": 1, "restaurantId": 19, "wonQuantity": 1},
            {"ingredient": "Sale Temporale", "bid": 12, "quantity": 2, "restaurantId": 6, "wonQuantity": 2},
        ]
        state.record_bid_history_turn(10, entries_t1, "19")
        state.record_bid_history_turn(11, entries_t2, "19")
        sig = state.ingredient_market_signal("Sale Temporale")
        assert sig["avg_win_2"] > 0
        assert sig["competition_2"] >= 1

    def test_bid_history_updates_team_win_stats(self):
        state = GameState()
        entries = [
            {"ingredient": "Plasma Vitale", "bid": 15, "quantity": 4, "restaurantId": 19, "wonQuantity": 3},
            {"ingredient": "Plasma Vitale", "bid": 10, "quantity": 3, "restaurantId": 5, "wonQuantity": 3},
        ]
        state.record_bid_history_turn(12, entries, "19")
        stats = state.bid_win_stats.get("Plasma Vitale")
        assert stats is not None
        assert int(stats["attempts"]) >= 4
        assert int(stats["wins"]) >= 3

    def test_bid_history_does_not_double_count_local_attempts(self):
        state = GameState()
        state.turn_id = 12
        state.record_bids_submitted(
            [{"ingredient": "Plasma Vitale", "bid": 15, "quantity": 4}],
            {},
        )
        before = dict(state.bid_win_stats.get("Plasma Vitale", {}))
        entries = [
            {"ingredient": "Plasma Vitale", "bid": 15, "quantity": 4, "restaurantId": 19, "wonQuantity": 3},
            {"ingredient": "Plasma Vitale", "bid": 10, "quantity": 3, "restaurantId": 5, "wonQuantity": 3},
        ]
        state.record_bid_history_turn(12, entries, "19")
        after = state.bid_win_stats.get("Plasma Vitale", {})
        assert int(after.get("attempts", 0)) == int(before.get("attempts", 0))

    def test_bid_history_also_records_in_archive(self):
        state = GameState()
        entries = [
            {"ingredient": "Sale Temporale", "bid": 15, "quantity": 2, "restaurantId": 19, "wonQuantity": 2},
        ]
        state.record_bid_history_turn(10, entries, "19")
        assert state.bid_archive.has_turn(10)


class TestBidArchive:
    def _make_entries(self, ingredient, prices_and_wins):
        """Helper: create bid entries from list of (price, qty_won) tuples."""
        return [
            {"ingredient": ingredient, "bid": p, "quantity": 2, "restaurantId": str(i), "wonQuantity": w}
            for i, (p, w) in enumerate(prices_and_wins)
        ]

    def test_avg_clearing_price(self):
        archive = BidArchive()
        entries = self._make_entries("grano", [(10, 2), (20, 1)])
        archive.record_turn(1, entries, "0")
        avg = archive.avg_clearing_price("grano")
        # (10*2 + 20*1) / (2+1) = 40/3 ≈ 13.3
        assert 13.0 <= avg <= 14.0

    def test_price_range(self):
        archive = BidArchive()
        entries = self._make_entries("sale", [(10, 1), (20, 1), (15, 1)])
        archive.record_turn(1, entries, "0")
        p_min, p_max, p_med = archive.price_range("sale")
        assert p_min == 10.0
        assert p_max == 20.0
        assert p_med == 15.0

    def test_price_trend_stable(self):
        archive = BidArchive()
        for t in range(1, 8):
            entries = self._make_entries("riso", [(15, 1)])
            archive.record_turn(t, entries, "0")
        assert archive.price_trend("riso") == "stable"

    def test_price_trend_rising(self):
        archive = BidArchive()
        for t in range(1, 8):
            price = 10 + t * 5  # rising prices
            entries = self._make_entries("burro", [(price, 1)])
            archive.record_turn(t, entries, "0")
        assert archive.price_trend("burro") == "rising"

    def test_price_trend_falling(self):
        archive = BidArchive()
        for t in range(1, 8):
            price = 50 - t * 5  # falling prices
            entries = self._make_entries("zafferano", [(price, 1)])
            archive.record_turn(t, entries, "0")
        assert archive.price_trend("zafferano") == "falling"

    def test_competition_level(self):
        archive = BidArchive()
        entries = [
            {"ingredient": "grano", "bid": 10, "quantity": 2, "restaurantId": "1", "wonQuantity": 1},
            {"ingredient": "grano", "bid": 12, "quantity": 1, "restaurantId": "2", "wonQuantity": 1},
            {"ingredient": "grano", "bid": 8, "quantity": 3, "restaurantId": "3", "wonQuantity": 0},
        ]
        archive.record_turn(1, entries, "0")
        assert archive.competition_level("grano") == 3.0

    def test_cheapest_ingredients(self):
        archive = BidArchive()
        entries = (
            self._make_entries("grano", [(30, 1)])
            + self._make_entries("sale", [(5, 1)])
            + self._make_entries("burro", [(15, 1)])
        )
        archive.record_turn(1, entries, "0")
        cheapest = archive.cheapest_ingredients(top_n=2)
        assert len(cheapest) == 2
        assert cheapest[0][0] == "sale"  # cheapest first

    def test_summary_for_ingredient(self):
        archive = BidArchive()
        entries = self._make_entries("grano", [(10, 2), (20, 1)])
        archive.record_turn(1, entries, "0")
        summary = archive.summary_for_ingredient("grano")
        assert summary["ingredient"] == "grano"
        assert summary["avg_price"] > 0
        assert summary["n_observations"] > 0

    def test_no_duplicate_turns(self):
        archive = BidArchive()
        entries = self._make_entries("grano", [(10, 1)])
        archive.record_turn(1, entries, "0")
        archive.record_turn(1, entries, "0")  # duplicate
        assert len(archive._turns) == 1

    def test_trim_old_turns(self):
        archive = BidArchive()
        for t in range(1, 60):
            entries = self._make_entries("grano", [(10, 1)])
            archive.record_turn(t, entries, "0")
        assert len(archive._turns) <= 50

    def test_save_load_roundtrip(self, tmp_path):
        archive = BidArchive()
        entries = self._make_entries("grano", [(10, 2)])
        archive.record_turn(5, entries, "0")
        path = str(tmp_path / "test_archive.json")
        assert archive.save(path)
        fresh = BidArchive()
        assert fresh.load(path)
        assert fresh.has_turn(5)
        assert fresh.avg_clearing_price("grano") == archive.avg_clearing_price("grano")


class TestCanCookMenuShape:
    def test_can_cook_with_menu_as_dict_items(self):
        state = GameState()
        state.recipes = [{"name": "Dish A", "ingredients": {"A": 1}}]
        state.inventory = {"A": 1}
        state.menu = {"items": [{"name": "Dish A", "price": 80}]}
        assert state.can_cook_any_menu_recipe()

    def test_can_cook_with_menu_as_list(self):
        state = GameState()
        state.recipes = [{"name": "Dish A", "ingredients": {"A": 1}}]
        state.inventory = {"A": 1}
        state.menu = [{"name": "Dish A", "price": 80}]
        assert state.can_cook_any_menu_recipe()
