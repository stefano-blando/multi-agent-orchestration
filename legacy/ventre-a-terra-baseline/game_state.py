"""In-memory game state tracker, updated from SSE events and HTTP responses."""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field

from utils.price_db import PriceDatabase
from utils.bid_archive import BidArchive

log = logging.getLogger(__name__)
_MARKET_BUY_LINE_RE = re.compile(
    r"Restaurant\s+\d+\s+try to buy:(?P<qty>\d+)\s+(?P<ingredient>.+?)\s+at single price of:\s*(?P<price>\d+)\s+result:(?P<result>.+)$",
    re.IGNORECASE,
)
_MARKET_BOUGHT_QTY_RE = re.compile(r"Bought\s+(?P<qty>\d+)\s+", re.IGNORECASE)
_INT_RE = re.compile(r"-?\d+")


@dataclass
class ClientOrder:
    client_id: str
    name: str
    order_text: str
    requested_dish: str = ""
    served: bool = False


@dataclass
class GameState:
    # Turn / phase
    phase: str = "unknown"
    turn_id: int = 0

    # Restaurant state (refreshed via GET /restaurant/:id)
    balance: float = 0.0
    inventory: dict[str, int] = field(default_factory=dict)
    menu: list[dict] = field(default_factory=list)
    is_open: bool = True

    # Recipes (refreshed via GET /recipes)
    recipes: list[dict] = field(default_factory=list)

    # Serving phase helpers
    pending_clients: list[ClientOrder] = field(default_factory=list)
    prepared_dishes: list[str] = field(default_factory=list)
    dish_demand: dict[str, int] = field(default_factory=dict)
    dish_served_score: dict[str, int] = field(default_factory=dict)
    served_this_turn: int = 0
    served_last_turn: int = 0
    clients_seen_this_turn: int = 0
    clients_seen_last_turn: int = 0
    dish_requests_this_turn: dict[str, int] = field(default_factory=dict)
    dish_requests_last_turn: dict[str, int] = field(default_factory=dict)
    dish_served_this_turn: dict[str, int] = field(default_factory=dict)
    dish_served_last_turn: dict[str, int] = field(default_factory=dict)
    _seen_meal_ids_this_turn: set[str] = field(default_factory=set, repr=False)
    revenue_this_turn: int = 0
    revenue_last_turn: int = 0
    revenue_history: list[int] = field(default_factory=list)
    bid_submitted_spend_this_turn: int = 0
    bid_submitted_spend_last_turn: int = 0
    bid_submitted_by_ingredient_this_turn: dict[str, int] = field(default_factory=dict)
    bid_submitted_by_ingredient_last_turn: dict[str, int] = field(default_factory=dict)
    _pending_bid_quantities: dict[str, int] = field(default_factory=dict, repr=False)
    _pending_bid_inventory_snapshot: dict[str, int] = field(default_factory=dict, repr=False)
    bid_win_stats: dict[str, dict[str, int]] = field(default_factory=dict)
    dish_realized_prices: dict[str, list[int]] = field(default_factory=dict)
    meal_runtime_state: dict[str, str] = field(default_factory=dict)
    meal_runtime_dish: dict[str, str] = field(default_factory=dict)
    meal_runtime_reason: dict[str, str] = field(default_factory=dict)
    meal_runtime_operation_id: dict[str, str] = field(default_factory=dict)
    operation_journal: dict[str, dict] = field(default_factory=dict)
    _meal_started_at_epoch: dict[str, float] = field(default_factory=dict, repr=False)
    prepare_attempts_this_turn: int = 0
    prepare_attempts_last_turn: int = 0
    prepare_failures_this_turn: int = 0
    prepare_failures_last_turn: int = 0
    serve_attempts_this_turn: int = 0
    serve_attempts_last_turn: int = 0
    serve_failures_this_turn: int = 0
    serve_failures_last_turn: int = 0
    serve_not_ready_errors_this_turn: int = 0
    serve_not_ready_errors_last_turn: int = 0
    serve_latency_ms_sum_this_turn: int = 0
    serve_latency_ms_sum_last_turn: int = 0
    serve_latency_samples_this_turn: int = 0
    serve_latency_samples_last_turn: int = 0

    # Market snapshot
    market_entries: list[dict] = field(default_factory=list)
    ingredient_market_history: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    _market_turn_stats: dict[str, dict[str, float]] = field(default_factory=dict, repr=False)
    bid_history_processed_turns: set[int] = field(default_factory=set, repr=False)
    bid_win_stats_accounted_turns: set[int] = field(default_factory=set, repr=False)
    bid_attempts_recorded_turns: set[int] = field(default_factory=set, repr=False)

    # Long-horizon price database (persisted between restarts)
    price_db: PriceDatabase = field(default_factory=PriceDatabase, repr=False, compare=False)

    # Historical bid archive (persisted between restarts and game resets)
    bid_archive: BidArchive = field(default_factory=BidArchive, repr=False, compare=False)

    # ── SSE event handlers ────────────────────────────────────────────

    @staticmethod
    def _norm(text: str) -> str:
        return " ".join(re.sub(r"\s+", " ", str(text)).strip().lower().split())

    def _extract_requested_dish(self, order_text: str) -> str:
        norm_order = self._norm(order_text)
        if not norm_order:
            return ""

        # Longest recipe names first to avoid partial matches.
        for recipe in sorted(self.recipes or [], key=lambda r: len(str(r.get("name", ""))), reverse=True):
            name = str(recipe.get("name", "")).strip()
            if not name:
                continue
            if self._norm(name) in norm_order:
                return name
        return ""

    def on_game_started(self, data: dict) -> None:
        self._finalize_market_turn()
        # turn_id may be missing on reconnect; keep monotonic progression.
        self.turn_id = data.get("turn_id", self.turn_id + 1)
        self.served_last_turn = self.served_this_turn
        self.served_this_turn = 0
        self.clients_seen_last_turn = self.clients_seen_this_turn
        self.clients_seen_this_turn = 0
        self.dish_requests_last_turn = dict(self.dish_requests_this_turn)
        self.dish_requests_this_turn.clear()
        self.dish_served_last_turn = dict(self.dish_served_this_turn)
        self.dish_served_this_turn.clear()
        self._seen_meal_ids_this_turn.clear()
        self.revenue_last_turn = int(self.revenue_this_turn)
        self.revenue_history.append(self.revenue_last_turn)
        if len(self.revenue_history) > 12:
            self.revenue_history = self.revenue_history[-12:]
        self.revenue_this_turn = 0
        self.bid_submitted_spend_last_turn = int(self.bid_submitted_spend_this_turn)
        self.bid_submitted_spend_this_turn = 0
        self.bid_submitted_by_ingredient_last_turn = dict(self.bid_submitted_by_ingredient_this_turn)
        self.bid_submitted_by_ingredient_this_turn.clear()
        self._pending_bid_quantities.clear()
        self._pending_bid_inventory_snapshot.clear()
        self.prepare_attempts_last_turn = int(self.prepare_attempts_this_turn)
        self.prepare_attempts_this_turn = 0
        self.prepare_failures_last_turn = int(self.prepare_failures_this_turn)
        self.prepare_failures_this_turn = 0
        self.serve_attempts_last_turn = int(self.serve_attempts_this_turn)
        self.serve_attempts_this_turn = 0
        self.serve_failures_last_turn = int(self.serve_failures_this_turn)
        self.serve_failures_this_turn = 0
        self.serve_not_ready_errors_last_turn = int(self.serve_not_ready_errors_this_turn)
        self.serve_not_ready_errors_this_turn = 0
        self.serve_latency_ms_sum_last_turn = int(self.serve_latency_ms_sum_this_turn)
        self.serve_latency_ms_sum_this_turn = 0
        self.serve_latency_samples_last_turn = int(self.serve_latency_samples_this_turn)
        self.serve_latency_samples_this_turn = 0
        # Reset per-turn state
        self.pending_clients.clear()
        self.prepared_dishes.clear()
        self.meal_runtime_state.clear()
        self.meal_runtime_dish.clear()
        self.meal_runtime_reason.clear()
        self.meal_runtime_operation_id.clear()
        self.operation_journal.clear()
        self._meal_started_at_epoch.clear()
        self._selected_recipes = None
        # Keep a decayed memory of demand across turns.
        if self.dish_demand:
            self.dish_demand = {
                dish: int(score * 0.7)
                for dish, score in self.dish_demand.items()
                if int(score * 0.7) > 0
            }
        if self.dish_served_score:
            self.dish_served_score = {
                dish: int(score * 0.7)
                for dish, score in self.dish_served_score.items()
                if int(score * 0.7) > 0
            }
        log.info("Game started, turn %d", self.turn_id)

    def on_phase_changed(self, phase: str) -> None:
        self.phase = phase
        log.info("Phase → %s", phase)

    def on_client_spawned(self, data: dict) -> None:
        raw_client_id = data.get("clientId")
        order_text = str(data.get("orderText", ""))
        requested_dish = self._extract_requested_dish(order_text)
        client = ClientOrder(
            client_id=str(raw_client_id) if raw_client_id is not None else "",
            name=data.get("clientName", "unknown"),
            order_text=order_text,
            requested_dish=requested_dish,
        )
        self.pending_clients.append(client)
        shown_id = client.client_id or "pending_from_meals"
        log.info("Client %s (id=%s) wants: %s", client.name, shown_id, client.order_text)

    def on_preparation_complete(self, data: dict) -> None:
        dish = data.get("dish", "unknown")
        self.prepared_dishes.append(dish)
        log.info("Dish ready: %s", dish)

    def on_game_reset(self) -> None:
        self.phase = "unknown"
        self.turn_id = 0
        self.served_this_turn = 0
        self.served_last_turn = 0
        self.clients_seen_this_turn = 0
        self.clients_seen_last_turn = 0
        self.dish_requests_this_turn.clear()
        self.dish_requests_last_turn.clear()
        self.dish_served_this_turn.clear()
        self.dish_served_last_turn.clear()
        self._seen_meal_ids_this_turn.clear()
        self.revenue_this_turn = 0
        self.revenue_last_turn = 0
        self.revenue_history.clear()
        self.bid_submitted_spend_this_turn = 0
        self.bid_submitted_spend_last_turn = 0
        self.bid_submitted_by_ingredient_this_turn.clear()
        self.bid_submitted_by_ingredient_last_turn.clear()
        self._pending_bid_quantities.clear()
        self._pending_bid_inventory_snapshot.clear()
        self.bid_win_stats.clear()
        self.dish_realized_prices.clear()
        self.meal_runtime_state.clear()
        self.meal_runtime_dish.clear()
        self.meal_runtime_reason.clear()
        self.meal_runtime_operation_id.clear()
        self.operation_journal.clear()
        self._meal_started_at_epoch.clear()
        self.prepare_attempts_this_turn = 0
        self.prepare_attempts_last_turn = 0
        self.prepare_failures_this_turn = 0
        self.prepare_failures_last_turn = 0
        self.serve_attempts_this_turn = 0
        self.serve_attempts_last_turn = 0
        self.serve_failures_this_turn = 0
        self.serve_failures_last_turn = 0
        self.serve_not_ready_errors_this_turn = 0
        self.serve_not_ready_errors_last_turn = 0
        self.serve_latency_ms_sum_this_turn = 0
        self.serve_latency_ms_sum_last_turn = 0
        self.serve_latency_samples_this_turn = 0
        self.serve_latency_samples_last_turn = 0
        self.pending_clients.clear()
        self.prepared_dishes.clear()
        self._selected_recipes = None
        self._latest_bids = None
        self._latest_bid_prices = None
        self.ingredient_market_history.clear()
        self._market_turn_stats.clear()
        self.bid_history_processed_turns.clear()
        self.bid_win_stats_accounted_turns.clear()
        self.bid_attempts_recorded_turns.clear()
        log.info("Game reset")

    @staticmethod
    def _as_int(value, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            if isinstance(value, str):
                m = _INT_RE.search(value)
                if m:
                    try:
                        return int(m.group(0))
                    except (TypeError, ValueError):
                        return default
            return default

    def has_bid_history_turn(self, turn_id: int) -> bool:
        return int(turn_id) in self.bid_history_processed_turns

    def record_bid_history_turn(self, turn_id: int, entries: list[dict], my_team_id: str) -> None:
        """Ingest one turn of /bid_history to derive market and team win signals.

        This is used as short-horizon memory (last 2 turns) instead of disk snapshots.
        """
        t = int(turn_id)
        if t <= 0 or t in self.bid_history_processed_turns:
            return
        if not isinstance(entries, list):
            return

        turn_stats: dict[str, dict[str, float]] = {}
        my_team = str(my_team_id).strip()
        my_team_i = self._as_int(my_team_id, -1)
        update_team_stats = (
            t not in self.bid_win_stats_accounted_turns
            and t not in self.bid_attempts_recorded_turns
        )

        def _pick(entry: dict, keys: tuple[str, ...], default=None):
            for k in keys:
                if k in entry:
                    return entry.get(k)
            return default

        for raw in entries:
            if not isinstance(raw, dict):
                continue
            ingredient = str(
                _pick(raw, ("ingredient", "ingredientName", "item", "name"), "")
            ).strip()
            if not ingredient:
                continue

            qty_req = max(
                0,
                self._as_int(
                    _pick(raw, ("quantity", "qty", "requestedQuantity", "requested_qty", "amount"), 0),
                    0,
                ),
            )
            bid_unit = self._as_int(
                _pick(raw, ("bid", "price", "unitPrice", "unit_price", "singlePrice", "single_price"), 0),
                0,
            )
            if bid_unit <= 0:
                total_price = self._as_int(_pick(raw, ("totalPrice", "total_price", "cost"), 0), 0)
                if qty_req > 0 and total_price > 0:
                    bid_unit = max(1, int(total_price // qty_req))

            stats = turn_stats.setdefault(
                ingredient,
                {"bid_events": 0.0, "unit_bid_sum": 0.0, "winning_qty": 0.0, "winning_spend": 0.0},
            )
            stats["bid_events"] += 1.0
            if bid_unit > 0:
                stats["unit_bid_sum"] += float(bid_unit)

            won_qty = max(
                0,
                self._as_int(
                    _pick(raw, ("wonQuantity", "won_qty", "boughtQuantity", "executedQuantity", "filledQuantity"), 0),
                    0,
                ),
            )
            if won_qty <= 0:
                result_text = str(_pick(raw, ("result", "status", "outcome", "message"), "") or "")
                bought_m = _MARKET_BOUGHT_QTY_RE.search(result_text)
                if bought_m:
                    won_qty = max(0, self._as_int(bought_m.group("qty"), 0))

            if won_qty > 0 and bid_unit > 0:
                stats["winning_qty"] += float(won_qty)
                stats["winning_spend"] += float(won_qty * bid_unit)

            rid_raw = _pick(raw, ("restaurantId", "restaurant_id", "teamId", "team_id", "restaurant"), "")
            rid = str(
                rid_raw
            ).strip()
            rid_i = self._as_int(rid_raw, -1)
            is_mine = (rid and rid == my_team) or (rid_i >= 0 and my_team_i >= 0 and rid_i == my_team_i)
            if update_team_stats and is_mine and qty_req > 0:
                my_stats = self.bid_win_stats.setdefault(ingredient, {"wins": 0, "attempts": 0})
                my_stats["attempts"] = int(my_stats.get("attempts", 0)) + int(qty_req)
                my_stats["wins"] = int(my_stats.get("wins", 0)) + int(min(qty_req, won_qty))

        for ingredient, stats in turn_stats.items():
            bid_events = int(stats.get("bid_events", 0.0))
            if bid_events <= 0:
                continue
            winning_qty = float(stats.get("winning_qty", 0.0))
            winning_spend = float(stats.get("winning_spend", 0.0))
            unit_bid_sum = float(stats.get("unit_bid_sum", 0.0))

            if winning_qty > 0:
                avg_win = winning_spend / winning_qty
            else:
                avg_win = unit_bid_sum / float(bid_events)

            entry = self.ingredient_market_history.setdefault(
                ingredient,
                {"avg_win": [], "competition": []},
            )
            entry["avg_win"].append(float(avg_win))
            entry["competition"].append(float(bid_events))
            if len(entry["avg_win"]) > 8:
                entry["avg_win"] = entry["avg_win"][-8:]
            if len(entry["competition"]) > 8:
                entry["competition"] = entry["competition"][-8:]

            # Also record in long-horizon price database.
            self.price_db.update(ingredient, avg_win)

        self.bid_history_processed_turns.add(t)
        if update_team_stats:
            self.bid_win_stats_accounted_turns.add(t)

        # Also record in persistent bid archive.
        self.bid_archive.record_turn(t, entries, my_team_id)

    def record_market_activity_message(self, payload: str) -> None:
        """Parse auction summary lines to infer winning prices and competition."""
        if not payload:
            return
        for raw in str(payload).splitlines():
            line = raw.strip()
            if not line:
                continue
            m = _MARKET_BUY_LINE_RE.match(line)
            if not m:
                continue
            ingredient = str(m.group("ingredient")).strip()
            try:
                unit_price = int(m.group("price"))
            except (TypeError, ValueError):
                continue
            result = str(m.group("result")).strip()
            bought_qty = 0
            bought_m = _MARKET_BOUGHT_QTY_RE.search(result)
            if bought_m:
                try:
                    bought_qty = int(bought_m.group("qty"))
                except (TypeError, ValueError):
                    bought_qty = 0

            stats = self._market_turn_stats.setdefault(
                ingredient,
                {
                    "bid_events": 0.0,
                    "unit_bid_sum": 0.0,
                    "winning_qty": 0.0,
                    "winning_spend": 0.0,
                },
            )
            stats["bid_events"] += 1.0
            stats["unit_bid_sum"] += float(unit_price)
            if bought_qty > 0:
                stats["winning_qty"] += float(bought_qty)
                stats["winning_spend"] += float(unit_price * bought_qty)

    def _finalize_market_turn(self) -> None:
        """Roll per-turn market parsed stats into short history."""
        if not self._market_turn_stats:
            return
        for ingredient, stats in self._market_turn_stats.items():
            bid_events = int(stats.get("bid_events", 0.0))
            winning_qty = float(stats.get("winning_qty", 0.0))
            winning_spend = float(stats.get("winning_spend", 0.0))
            unit_bid_sum = float(stats.get("unit_bid_sum", 0.0))
            if bid_events <= 0:
                continue

            if winning_qty > 0:
                avg_win = winning_spend / winning_qty
            else:
                avg_win = unit_bid_sum / float(bid_events)

            entry = self.ingredient_market_history.setdefault(
                ingredient,
                {"avg_win": [], "competition": []},
            )
            entry["avg_win"].append(float(avg_win))
            entry["competition"].append(float(bid_events))
            if len(entry["avg_win"]) > 8:
                entry["avg_win"] = entry["avg_win"][-8:]
            if len(entry["competition"]) > 8:
                entry["competition"] = entry["competition"][-8:]

            # Also record in long-horizon price database.
            self.price_db.update(ingredient, avg_win)

        self._market_turn_stats.clear()

    def ingredient_market_signal(self, ingredient: str) -> dict[str, float]:
        """Return market signal for one ingredient (short + long horizon).

        Short-horizon: last 2 turns from bid_history / SSE market messages.
        Long-horizon: EMA over all recorded history (price_db).
        """
        entry = self.ingredient_market_history.get(ingredient, {})
        avg_win_hist = [float(x) for x in entry.get("avg_win", [])]
        comp_hist = [float(x) for x in entry.get("competition", [])]

        recent_avg = avg_win_hist[-2:]
        recent_comp = comp_hist[-2:]
        avg_win_2 = (sum(recent_avg) / len(recent_avg)) if recent_avg else 0.0
        comp_2 = (sum(recent_comp) / len(recent_comp)) if recent_comp else 0.0
        if len(avg_win_hist) >= 2 and avg_win_hist[-2] > 0:
            trend = avg_win_hist[-1] / avg_win_hist[-2]
        else:
            trend = 1.0

        # Lower competition usually means scarcer/neglected ingredient.
        if comp_2 <= 2:
            rarity = 1.20
        elif comp_2 <= 4:
            rarity = 1.10
        elif comp_2 <= 7:
            rarity = 1.00
        else:
            rarity = 0.90

        # Long-horizon signals from price_db.
        db_signal = self.price_db.get_signal(ingredient)

        return {
            "avg_win_2": float(avg_win_2),
            "competition_2": float(comp_2),
            "trend": float(trend),
            "rarity": float(rarity),
            # Long-horizon moving averages (0.0 if not enough data).
            "ema_short": db_signal["ema_short"],
            "ema_long": db_signal["ema_long"],
            "long_trend": db_signal["trend"],
            "n_obs": db_signal["n"],
        }

    def record_meal_observed(self, meal_id: str, requested_dish: str = "") -> bool:
        """Track one unique meal in the current turn. Returns True if newly observed."""
        if not meal_id:
            return False
        if meal_id in self._seen_meal_ids_this_turn:
            return False
        self._seen_meal_ids_this_turn.add(meal_id)
        self.clients_seen_this_turn += 1
        if requested_dish:
            self.dish_demand[requested_dish] = self.dish_demand.get(requested_dish, 0) + 1
            self.dish_requests_this_turn[requested_dish] = self.dish_requests_this_turn.get(requested_dish, 0) + 1
        return True

    def record_dish_served(self, dish_name: str, earned: int = 0) -> None:
        """Track served dish metrics for pricing and KPI."""
        self.served_this_turn += 1
        self.revenue_this_turn += max(0, int(earned))
        if not dish_name:
            return
        self.dish_served_this_turn[dish_name] = self.dish_served_this_turn.get(dish_name, 0) + 1
        self.dish_served_score[dish_name] = self.dish_served_score.get(dish_name, 0) + 1
        if int(earned) > 0:
            hist = self.dish_realized_prices.setdefault(dish_name, [])
            hist.append(int(earned))
            if len(hist) > 12:
                self.dish_realized_prices[dish_name] = hist[-12:]

    def record_prepare_attempt(self, success: bool) -> None:
        self.prepare_attempts_this_turn += 1
        if not success:
            self.prepare_failures_this_turn += 1

    def record_serve_attempt(self, success: bool, not_ready_error: bool = False, latency_ms: int | None = None) -> None:
        self.serve_attempts_this_turn += 1
        if not success:
            self.serve_failures_this_turn += 1
        if not_ready_error:
            self.serve_not_ready_errors_this_turn += 1
        if latency_ms is not None and latency_ms >= 0:
            self.serve_latency_ms_sum_this_turn += int(latency_ms)
            self.serve_latency_samples_this_turn += 1

    def record_meal_start_time(self, meal_id: str, start_time_iso: str) -> None:
        if not meal_id or not start_time_iso:
            return
        if meal_id in self._meal_started_at_epoch:
            return
        raw = str(start_time_iso).strip()
        if not raw:
            return
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            self._meal_started_at_epoch[meal_id] = float(dt.timestamp())
        except (TypeError, ValueError):
            return

    def meal_latency_ms(self, meal_id: str) -> int | None:
        started = self._meal_started_at_epoch.get(meal_id)
        if started is None:
            return None
        return max(0, int((time.time() - started) * 1000))

    def set_meal_runtime_state(
        self,
        meal_id: str,
        state: str,
        dish_name: str = "",
        reason: str = "",
        operation_id: str = "",
    ) -> None:
        if not meal_id:
            return
        self.meal_runtime_state[meal_id] = state
        if dish_name:
            self.meal_runtime_dish[meal_id] = dish_name
        if reason:
            self.meal_runtime_reason[meal_id] = reason
        if operation_id:
            self.meal_runtime_operation_id[meal_id] = operation_id

    def get_meal_runtime_state(self, meal_id: str) -> str:
        return self.meal_runtime_state.get(meal_id, "unknown")

    def record_operation(self, operation_id: str, ok: bool, response: str, error: str = "") -> None:
        if not operation_id:
            return
        self.operation_journal[operation_id] = {
            "ok": bool(ok),
            "response": str(response),
            "error": str(error),
            "ts": time.time(),
        }
        if len(self.operation_journal) > 400:
            # Keep most recent operations only.
            keys = list(self.operation_journal.keys())[-300:]
            self.operation_journal = {k: self.operation_journal[k] for k in keys}

    def get_operation(self, operation_id: str) -> dict | None:
        out = self.operation_journal.get(operation_id)
        if not isinstance(out, dict):
            return None
        return out

    def record_bids_submitted(self, bids: list[dict], current_inventory: dict[str, int]) -> None:
        """Track submitted bids for spend reporting and win-rate feedback."""
        self._pending_bid_quantities.clear()
        self._pending_bid_inventory_snapshot = dict(current_inventory or {})

        total = 0
        by_ing: dict[str, int] = {}
        for b in bids or []:
            ing = str(b.get("ingredient", "")).strip()
            qty = max(0, int(b.get("quantity", 0)))
            bid = max(0, int(b.get("bid", 0)))
            if not ing or qty <= 0:
                continue
            self._pending_bid_quantities[ing] = self._pending_bid_quantities.get(ing, 0) + qty
            by_ing[ing] = by_ing.get(ing, 0) + qty
            total += bid * qty
            stats = self.bid_win_stats.setdefault(ing, {"wins": 0, "attempts": 0})
            stats["attempts"] = int(stats.get("attempts", 0)) + qty

        self.bid_submitted_spend_this_turn = int(total)
        self.bid_submitted_by_ingredient_this_turn = by_ing
        if int(self.turn_id) > 0 and by_ing:
            self.bid_attempts_recorded_turns.add(int(self.turn_id))

    def finalize_bid_outcomes(self, current_inventory: dict[str, int]) -> None:
        """Update ingredient win stats using post-auction inventory delta."""
        if not self._pending_bid_quantities:
            return
        outcome_turn = max(0, int(self.turn_id) - 1)
        if outcome_turn in self.bid_win_stats_accounted_turns:
            self._pending_bid_quantities.clear()
            self._pending_bid_inventory_snapshot.clear()
            return

        inv_now = current_inventory or {}
        for ing, wanted_qty in self._pending_bid_quantities.items():
            base_qty = int(self._pending_bid_inventory_snapshot.get(ing, 0))
            now_qty = int(inv_now.get(ing, 0))
            gained = max(0, now_qty - base_qty)
            won_qty = min(int(wanted_qty), gained)
            stats = self.bid_win_stats.setdefault(ing, {"wins": 0, "attempts": 0})
            stats["wins"] = int(stats.get("wins", 0)) + won_qty

        self._pending_bid_quantities.clear()
        self._pending_bid_inventory_snapshot.clear()
        if outcome_turn > 0:
            self.bid_win_stats_accounted_turns.add(outcome_turn)

    def bid_adjustments(self) -> dict[str, int]:
        """Return per-ingredient bid adjustments from historical win-rate."""
        adj: dict[str, int] = {}
        for ing, stats in self.bid_win_stats.items():
            attempts = int(stats.get("attempts", 0))
            wins = int(stats.get("wins", 0))
            if attempts < 3:
                continue
            wr = wins / attempts if attempts > 0 else 0.0
            if wr < 0.35:
                adj[ing] = 3
            elif wr < 0.50:
                adj[ing] = 2
            elif wr > 0.90:
                adj[ing] = -3
            elif wr > 0.75:
                adj[ing] = -2
            elif wr > 0.60:
                adj[ing] = -1
        return adj

    def rolling_revenue_avg(self, window: int = 3) -> float:
        """Average revenue over recent completed turns."""
        if window <= 0:
            return 0.0
        values = self.revenue_history[-window:]
        if not values:
            return 0.0
        return float(sum(values)) / float(len(values))

    # ── HTTP response updaters ────────────────────────────────────────

    def update_restaurant(self, data: dict) -> None:
        """Update from GET /restaurant/:id response."""
        self.balance = data.get("balance", self.balance)
        self.inventory = data.get("inventory", self.inventory)
        self.menu = data.get("menu", self.menu)
        # Server payload may use either snake_case or camelCase.
        self.is_open = data.get("is_open", data.get("isOpen", self.is_open))

        # Try to pick up turn_id from restaurant data if we don't have one.
        turn_value = data.get("turn_id", data.get("turnId"))
        if self.turn_id == 0 and turn_value is not None:
            try:
                detected_turn = int(turn_value)
            except (TypeError, ValueError):
                detected_turn = 0
            if detected_turn > 0:
                self.turn_id = detected_turn
                log.info("Detected turn_id=%d from restaurant data", self.turn_id)

    def update_recipes(self, data: list[dict]) -> None:
        self.recipes = data

    def update_market(self, data: list[dict]) -> None:
        self.market_entries = data

    # ── Helpers ────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Compact text summary for the agent's context window."""
        lines = [
            f"Turn: {self.turn_id} | Phase: {self.phase} | Balance: {self.balance}",
            f"Inventory: {self.inventory}",
            f"Menu items: {len(self.menu)}",
            f"Recipes available: {len(self.recipes)}",
            f"Pending clients: {len(self.pending_clients)}",
            f"Prepared dishes: {self.prepared_dishes}",
        ]
        return "\n".join(lines)

    def unserved_clients(self) -> list[ClientOrder]:
        return [c for c in self.pending_clients if not c.served]

    def can_cook_any_menu_recipe(self) -> bool:
        """Return True if at least one menu item can still be cooked with current inventory."""
        if not self.menu or not self.recipes:
            return False
        by_name = {str(r.get("name", "")): r for r in self.recipes if isinstance(r, dict)}
        if isinstance(self.menu, dict):
            menu_items = self.menu.get("items", []) or []
        elif isinstance(self.menu, list):
            menu_items = self.menu
        else:
            menu_items = []
        for item in menu_items:
            if isinstance(item, str):
                name = item.strip()
            elif isinstance(item, dict):
                name = str(item.get("name", "")).strip()
            else:
                continue
            if not name:
                continue
            recipe = by_name.get(name)
            if not recipe:
                continue
            ings = recipe.get("ingredients", {})
            if all(self.inventory.get(ing, 0) >= int(qty) for ing, qty in ings.items()):
                return True
        return False

    def to_snapshot(self) -> dict:
        """Return a compact persistent snapshot for warm restarts."""
        return {
            "dish_demand": dict(self.dish_demand),
            "dish_served_score": dict(self.dish_served_score),
            "dish_realized_prices": dict(self.dish_realized_prices),
            "revenue_history": list(self.revenue_history),
            "bid_win_stats": dict(self.bid_win_stats),
            "ingredient_market_history": dict(self.ingredient_market_history),
        }

    def apply_snapshot(self, data: dict) -> None:
        """Restore non-authoritative memory from snapshot."""
        if not isinstance(data, dict):
            return
        dish_demand = data.get("dish_demand", {})
        dish_served = data.get("dish_served_score", {})
        realized = data.get("dish_realized_prices", {})
        rev_hist = data.get("revenue_history", [])
        bid_stats = data.get("bid_win_stats", {})
        market_hist = data.get("ingredient_market_history", {})

        if isinstance(dish_demand, dict):
            fixed_demand: dict[str, int] = {}
            for k, v in dish_demand.items():
                try:
                    iv = int(v)
                except (TypeError, ValueError):
                    continue
                if iv > 0:
                    fixed_demand[str(k)] = iv
            self.dish_demand = fixed_demand
        if isinstance(dish_served, dict):
            fixed_served: dict[str, int] = {}
            for k, v in dish_served.items():
                try:
                    iv = int(v)
                except (TypeError, ValueError):
                    continue
                if iv > 0:
                    fixed_served[str(k)] = iv
            self.dish_served_score = fixed_served
        if isinstance(realized, dict):
            fixed: dict[str, list[int]] = {}
            for k, vals in realized.items():
                if not isinstance(vals, list):
                    continue
                bucket: list[int] = []
                for x in vals:
                    try:
                        ix = int(x)
                    except (TypeError, ValueError):
                        continue
                    if ix > 0:
                        bucket.append(ix)
                fixed[str(k)] = bucket[-12:]
            self.dish_realized_prices = fixed
        if isinstance(rev_hist, list):
            fixed_rev: list[int] = []
            for x in rev_hist:
                try:
                    fixed_rev.append(int(x))
                except (TypeError, ValueError):
                    continue
            self.revenue_history = fixed_rev[-12:]
        if isinstance(bid_stats, dict):
            normalized: dict[str, dict[str, int]] = {}
            for ing, stats in bid_stats.items():
                if not isinstance(stats, dict):
                    continue
                normalized[str(ing)] = {
                    "wins": int(stats.get("wins", 0)),
                    "attempts": int(stats.get("attempts", 0)),
                }
            self.bid_win_stats = normalized
        if isinstance(market_hist, dict):
            fixed_market: dict[str, dict[str, list[float]]] = {}
            for ing, bucket in market_hist.items():
                if not isinstance(bucket, dict):
                    continue
                avg_win = bucket.get("avg_win", [])
                comp = bucket.get("competition", [])
                if not isinstance(avg_win, list) or not isinstance(comp, list):
                    continue
                fixed_market[str(ing)] = {
                    "avg_win": [float(x) for x in avg_win][-8:],
                    "competition": [float(x) for x in comp][-8:],
                }
            self.ingredient_market_history = fixed_market
