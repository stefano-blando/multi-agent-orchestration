"""Entry point: SSE event loop + robust phase dispatch with fallbacks."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any

import aiohttp
from dotenv import load_dotenv
from datapizza.tracing import ContextTracing

from agent import init_agent, create_agent_for_phase, build_prompt_for_phase, save_agent_memory
from api_get import GameGET
from api_mcp import GameMCP
from game_state import GameState
from utils.decision_engine import (
    build_bidding_decision,
    build_speaking_menu_decision,
    build_waiting_menu_decision,
)
from utils.strategy_flags import (
    menu_max_items,
    strat_v2_report_enabled,
)
from utils.market_utils import build_liquidation_sales
from utils.recipe_utils import compute_ingredient_competition
from utils.serving_engine import build_serving_plan
from utils.state_persistence import load_snapshot, save_snapshot

load_dotenv()

TEAM_API_KEY = os.environ["TEAM_API_KEY"]
TEAM_ID = os.environ["TEAM_ID"]
REGOLO_API_KEY = os.environ["REGOLO"]
BASE_URL = "https://hackapizza.datapizza.tech"
PRICE_DB_PATH = os.getenv("PRICE_DB_PATH", "price_history.json")
BID_ARCHIVE_PATH = os.getenv("BID_ARCHIVE_PATH", "bid_archive.json")

# -- Logging -----------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("agent.log", mode="a"),
    ],
)
log = logging.getLogger("main")

# -- Globals -----------------------------------------------------------------
_tracing = ContextTracing()
state = GameState()
get_api: GameGET | None = None
mcp_api: GameMCP | None = None

_phase_lock = asyncio.Lock()
_phase_queue: asyncio.Queue[str] = asyncio.Queue()
_queued_phases: set[str] = set()
_phase_worker_task: asyncio.Task | None = None
_serving_debounce_task: asyncio.Task | None = None
_serving_keepalive_task: asyncio.Task | None = None
_startup_grace_until: float | None = None
_turn_probe_next_allowed_at: float = 0.0
_bid_history_probe_next_allowed_at: float = 0.0

KNOWN_PHASES = {"speaking", "closed_bid", "waiting", "serving"}
SSE_RECONNECT_DELAY = 3
MAX_AGENT_RETRIES = 3
_EARNED_RE = re.compile(r"\bearned\s+(\d+)\b", re.IGNORECASE)


def _mcp_is_error(result: Any) -> bool:
    return isinstance(result, dict) and bool(result.get("isError"))


def _mcp_error_text(result: Any) -> str:
    if not isinstance(result, dict):
        return "Unknown MCP error"
    content = result.get("content", [])
    if isinstance(content, list):
        texts = [c.get("text", "") for c in content if isinstance(c, dict)]
        msg = "; ".join(t for t in texts if t)
        if msg:
            return msg
    return "Unknown MCP error"


def _extract_earned_from_mcp_result(result: Any) -> int:
    if not isinstance(result, dict):
        return 0
    direct = result.get("earned")
    try:
        if direct is not None:
            return max(0, int(direct))
    except (TypeError, ValueError):
        pass
    content = result.get("content", [])
    if not isinstance(content, list):
        return 0
    for item in content:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "") or "")
        match = _EARNED_RE.search(text)
        if match:
            return int(match.group(1))
    return 0


def _top_counts(values: dict[str, int], n: int = 3) -> str:
    if not values:
        return "-"
    top = sorted(values.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))[:n]
    return ", ".join(f"{name}:{qty}" for name, qty in top)


def _log_turn_kpis(current: bool) -> None:
    if current:
        turn = state.turn_id
        clients = int(state.clients_seen_this_turn)
        served = int(state.served_this_turn)
        req = state.dish_requests_this_turn
        served_by_dish = state.dish_served_this_turn
        label = "current"
        revenue = int(getattr(state, "revenue_this_turn", 0))
        bid_spend = int(getattr(state, "bid_submitted_spend_this_turn", 0))
        prep_attempts = int(getattr(state, "prepare_attempts_this_turn", 0))
        prep_failures = int(getattr(state, "prepare_failures_this_turn", 0))
        serve_attempts = int(getattr(state, "serve_attempts_this_turn", 0))
        serve_failures = int(getattr(state, "serve_failures_this_turn", 0))
        not_ready_errors = int(getattr(state, "serve_not_ready_errors_this_turn", 0))
        latency_sum = int(getattr(state, "serve_latency_ms_sum_this_turn", 0))
        latency_samples = int(getattr(state, "serve_latency_samples_this_turn", 0))
    else:
        turn = max(0, state.turn_id - 1)
        clients = int(state.clients_seen_last_turn)
        served = int(state.served_last_turn)
        req = state.dish_requests_last_turn
        served_by_dish = state.dish_served_last_turn
        label = "previous"
        revenue = int(getattr(state, "revenue_last_turn", 0))
        bid_spend = int(getattr(state, "bid_submitted_spend_last_turn", 0))
        prep_attempts = int(getattr(state, "prepare_attempts_last_turn", 0))
        prep_failures = int(getattr(state, "prepare_failures_last_turn", 0))
        serve_attempts = int(getattr(state, "serve_attempts_last_turn", 0))
        serve_failures = int(getattr(state, "serve_failures_last_turn", 0))
        not_ready_errors = int(getattr(state, "serve_not_ready_errors_last_turn", 0))
        latency_sum = int(getattr(state, "serve_latency_ms_sum_last_turn", 0))
        latency_samples = int(getattr(state, "serve_latency_samples_last_turn", 0))

    if turn <= 0:
        return
    conversion = (served / clients * 100.0) if clients > 0 else 0.0
    skipped = max(0, clients - served)
    net_turn = revenue - bid_spend
    avg_latency_ms = int(latency_sum / latency_samples) if latency_samples > 0 else 0
    if strat_v2_report_enabled():
        log.info(
            "KPI [%s turn %d] clients=%d served=%d skipped=%d conversion=%.1f%% revenue=%d bid_spend=%d net=%d prep=%d/%d serve=%d/%d not_ready=%d avg_latency_ms=%d top_req=[%s] top_served=[%s]",
            label,
            turn,
            clients,
            served,
            skipped,
            conversion,
            revenue,
            bid_spend,
            net_turn,
            prep_failures,
            prep_attempts,
            serve_failures,
            serve_attempts,
            not_ready_errors,
            avg_latency_ms,
            _top_counts(req),
            _top_counts(served_by_dish),
        )
    else:
        log.info(
            "KPI [%s turn %d] clients=%d served=%d skipped=%d conversion=%.1f%% top_req=[%s] top_served=[%s]",
            label,
            turn,
            clients,
            served,
            skipped,
            conversion,
            _top_counts(req),
            _top_counts(served_by_dish),
        )


def _phase_readiness_error(phase: str) -> str | None:
    if phase not in KNOWN_PHASES:
        return f"unknown phase {phase}"
    if phase == "serving":
        if state.turn_id <= 0:
            return "turn_id is unknown"
        if state.phase != "serving":
            return f"state phase mismatch ({state.phase})"
    return None


# -- Agent invocation ---------------------------------------------------------

def _run_agent(phase: str) -> None:
    """Sync function that creates and runs the phase-specific agent."""
    agent = create_agent_for_phase(phase)
    if not agent:
        log.warning("No agent config for phase: %s", phase)
        return

    prompt = build_prompt_for_phase(phase, state)
    if not prompt:
        log.warning("No prompt for phase: %s", phase)
        return

    with _tracing.trace(f"phase_{phase}_turn_{state.turn_id}"):
        result = agent.run(prompt)

    if not result:
        raise RuntimeError(f"Agent [{phase}] returned None")

    text = result.text[:500] if result.text else "(no text)"
    log.info("Agent [%s] done: %s", phase, text)

    if save_agent_memory(phase, agent):
        log.info("Agent [%s] memory saved", phase)


async def _refresh_state() -> None:
    """Refresh core GET state used by all strategies."""
    global _startup_grace_until, _turn_probe_next_allowed_at, _bid_history_probe_next_allowed_at
    try:
        rest_data = await get_api.restaurant()
        state.update_restaurant(rest_data)
    except Exception as e:
        log.error("State refresh: restaurant failed: %s", e)

    if not state.recipes:
        try:
            recipes = await get_api.recipes()
            state.update_recipes(recipes)
        except Exception as e:
            log.error("State refresh: recipes failed: %s", e)

    loop = asyncio.get_running_loop()
    now = loop.time()
    if _startup_grace_until is None:
        # Give SSE a brief chance to deliver game_started / phase_changed first.
        _startup_grace_until = now + 8.0

    # If turn_id is still 0 (mid-game join), probe to find current turn.
    if state.turn_id == 0:
        # During early startup with unknown phase, avoid aggressive probing.
        if state.phase == "unknown" and now < _startup_grace_until:
            log.info("turn_id still 0 and phase unknown: waiting for SSE before probing meals")
            return

        # Avoid repeated probes in short bursts.
        if now < _turn_probe_next_allowed_at:
            return
        _turn_probe_next_allowed_at = now + 15.0

        probe_cache: dict[int, bool] = {}

        async def _is_valid_turn(turn: int) -> bool:
            if turn in probe_cache:
                return probe_cache[turn]
            try:
                await get_api.meals(turn)
                probe_cache[turn] = True
                return True
            except aiohttp.ClientResponseError as e:
                # 400/404 are expected while probing invalid/non-active turns.
                if e.status in (400, 404):
                    probe_cache[turn] = False
                    return False
                log.warning("turn_id probe failed on %d: %s", turn, e)
            except Exception as e:
                log.warning("turn_id probe failed on %d: %s", turn, e)
            probe_cache[turn] = False
            return False

        # Robust search: find a valid upper range by exponential probing,
        # then binary-search the highest valid turn. This works even in late games.
        if not await _is_valid_turn(1):
            log.warning("Could not detect a valid turn_id by probing meals")
            return

        lo = 1
        hi = 2
        max_probe_turn = 2048
        while hi <= max_probe_turn and await _is_valid_turn(hi):
            lo = hi
            hi *= 2

        if hi > max_probe_turn and await _is_valid_turn(max_probe_turn):
            state.turn_id = max_probe_turn
            log.info("Detected turn_id=%d by probing meals (hit probe cap)", state.turn_id)
            return

        left, right = lo + 1, min(hi - 1, max_probe_turn)
        best = lo
        while left <= right:
            mid = (left + right) // 2
            if await _is_valid_turn(mid):
                best = mid
                left = mid + 1
            else:
                right = mid - 1

        state.turn_id = best
        log.info("Detected turn_id=%d by probing meals (binary search)", state.turn_id)

    # Use server bid_history from last 5 turns for short-horizon memory + archive.
    if state.turn_id > 1 and now >= _bid_history_probe_next_allowed_at:
        _bid_history_probe_next_allowed_at = now + 20.0
        for t in range(state.turn_id - 1, max(0, state.turn_id - 6), -1):
            if t <= 0 or (state.has_bid_history_turn(t) and state.bid_archive.has_turn(t)):
                continue
            try:
                entries = await get_api.bid_history(t)
            except aiohttp.ClientResponseError as e:
                if e.status in (400, 404):
                    continue
                log.warning("bid_history fetch failed for turn %d: %s", t, e)
                continue
            except Exception as e:
                log.warning("bid_history fetch failed for turn %d: %s", t, e)
                continue
            state.record_bid_history_turn(t, entries, TEAM_ID)
            log.info("Loaded bid_history turn %d (%d rows)", t, len(entries) if isinstance(entries, list) else 0)


async def fallback_speaking() -> str:
    await _refresh_state()
    max_items = menu_max_items()
    decision = build_speaking_menu_decision(state, max_items=max_items)
    selected = decision.selected
    menu_items = decision.menu_items

    if menu_items:
        result = await mcp_api.save_menu(menu_items)
        if _mcp_is_error(result):
            return f"save_menu failed: {_mcp_error_text(result)}"

    result = await mcp_api.update_restaurant_is_open(True)
    if _mcp_is_error(result):
        return f"open_restaurant failed: {_mcp_error_text(result)}"

    state._selected_recipes = selected
    return f"Fallback speaking: set {len(menu_items)} items and opened restaurant"


async def fallback_bidding() -> str:
    await _refresh_state()
    decision = build_bidding_decision(state)
    bids = decision.bids
    state._latest_bids = bids
    state._latest_bid_prices = {b["ingredient"]: int(b["bid"]) for b in bids}

    if not bids:
        return "Fallback bidding: no bids needed"

    result = await mcp_api.closed_bid(bids)
    if _mcp_is_error(result):
        return f"closed_bid failed: {_mcp_error_text(result)}"
    record_bids = getattr(state, "record_bids_submitted", None)
    if callable(record_bids):
        record_bids(bids, state.inventory)

    total = sum(int(b["bid"]) * int(b["quantity"]) for b in bids)
    return f"Fallback bidding: placed {len(bids)} bids, total={total}"


async def fallback_waiting() -> str:
    await _refresh_state()
    max_items = menu_max_items()
    decision = build_waiting_menu_decision(state, max_items=max_items)
    selected = decision.selected
    menu_items = decision.menu_items

    if menu_items:
        menu_result = await mcp_api.save_menu(menu_items)
        if _mcp_is_error(menu_result):
            return f"save_menu failed: {_mcp_error_text(menu_result)}"

        open_result = await mcp_api.update_restaurant_is_open(True)
        if _mcp_is_error(open_result):
            return f"open_restaurant failed: {_mcp_error_text(open_result)}"
    else:
        close_result = await mcp_api.update_restaurant_is_open(False)
        if _mcp_is_error(close_result):
            return f"close_restaurant failed: {_mcp_error_text(close_result)}"

    state._selected_recipes = selected

    # Sell surplus ingredients before they expire at end of turn.
    competition = compute_ingredient_competition(state.recipes or [])
    sales = build_liquidation_sales(state.inventory, selected, competition)
    sold_count = 0
    for sale in sales:
        result = await mcp_api.create_market_entry(
            "SELL", sale["ingredient"], sale["quantity"], sale["price"],
        )
        if not _mcp_is_error(result):
            sold_count += 1

    return f"Fallback waiting: menu={len(menu_items)} items, listed {sold_count} surplus sales"


async def fallback_serving() -> str:
    await _refresh_state()

    try:
        meals = await get_api.meals(state.turn_id)
    except Exception as e:
        return f"meals fetch failed: {e}"

    recipes = state.recipes or []
    extract_fn = getattr(state, "_extract_requested_dish", None)
    record_meal = getattr(state, "record_meal_observed", None)
    for meal in meals:
        mid = str(meal.get("id", "")).strip()
        req_text = str(meal.get("orderText") or meal.get("request") or "")
        dish_name = str(extract_fn(req_text) if callable(extract_fn) else "")
        start_time = str(meal.get("startTime") or "")
        if mid and start_time:
            state.record_meal_start_time(mid, start_time)
        if callable(record_meal):
            record_meal(mid, dish_name)

    tasks = build_serving_plan(state, meals, recipes)
    if not tasks:
        return "Fallback serving: no unserved meals"

    served_count = 0
    skipped_count = 0
    attempted_ids: set[str] = set()
    for task in tasks:
        if state.phase != "serving":
            break

        meal_id = str(task.meal_id)
        customer_id = task.customer_id
        client_name = task.client_name
        if not meal_id or meal_id in attempted_ids:
            continue
        attempted_ids.add(meal_id)

        if task.action != "serve":
            state.set_meal_runtime_state(meal_id, "skipped", dish_name=task.requested_dish, reason=task.reason)
            skipped_count += 1
            continue

        dish_name = task.dish_to_serve
        state.set_meal_runtime_state(meal_id, "preparing", dish_name=dish_name, reason="fallback_prepare")
        prep_result = await mcp_api.prepare_dish(dish_name)
        if _mcp_is_error(prep_result):
            log.warning("Fallback serving prepare failed (%s): %s", dish_name, _mcp_error_text(prep_result))
            state.set_meal_runtime_state(meal_id, "failed", dish_name=dish_name, reason=f"prepare_error:{_mcp_error_text(prep_result)}")
            skipped_count += 1
            continue

        # Keep local inventory coherent with one in-flight preparation.
        for recipe in recipes:
            if recipe.get("name") != dish_name:
                continue
            for ing, qty in recipe.get("ingredients", {}).items():
                state.inventory[ing] = max(0, state.inventory.get(ing, 0) - int(qty))
            break

        ready = False
        for _ in range(60):
            if dish_name in state.prepared_dishes:
                state.prepared_dishes.remove(dish_name)
                ready = True
                break
            if state.phase != "serving":
                break
            await asyncio.sleep(0.5)

        if not ready:
            log.warning("Fallback serving: timeout waiting %s, trying serve anyway", dish_name)

        # Server behavior can differ between meal.id and customerId. Try both IDs.
        state.set_meal_runtime_state(meal_id, "serving", dish_name=dish_name, reason="fallback_serve")
        serve_result = await mcp_api.serve_dish(dish_name, meal_id)
        if _mcp_is_error(serve_result):
            err_text = _mcp_error_text(serve_result)
            err_lower = err_text.lower()
            alt_id = str(customer_id or "")
            if "not waiting" in err_lower and alt_id and alt_id != meal_id:
                alt_result = await mcp_api.serve_dish(dish_name, alt_id)
                if not _mcp_is_error(alt_result):
                    serve_result = alt_result
                else:
                    log.warning(
                        "Fallback serving serve failed (%s -> %s) with IDs %s/%s: %s",
                        client_name,
                        dish_name,
                        meal_id,
                        alt_id,
                        _mcp_error_text(alt_result),
                    )
                    state.set_meal_runtime_state(meal_id, "failed", dish_name=dish_name, reason=f"serve_error:{_mcp_error_text(alt_result)}")
                    skipped_count += 1
                    continue
            else:
                log.warning(
                    "Fallback serving serve failed (%s -> %s): %s",
                    client_name,
                    dish_name,
                    err_text,
                )
                state.set_meal_runtime_state(meal_id, "failed", dish_name=dish_name, reason=f"serve_error:{err_text}")
                skipped_count += 1
                continue

        served_count += 1
        earned = _extract_earned_from_mcp_result(serve_result)
        record_serve = getattr(state, "record_dish_served", None)
        if callable(record_serve):
            record_serve(dish_name, earned)
        else:
            state.served_this_turn += 1

        for c in state.pending_clients:
            if c.client_id in {meal_id, str(customer_id or "")}:
                c.served = True
        state.set_meal_runtime_state(meal_id, "served", dish_name=dish_name, reason="ok")

    return f"Fallback serving: served {served_count}/{len(tasks)} (skipped={skipped_count})"


async def _end_serving_if_exhausted() -> None:
    """Call by fallback after serving: log cookable state and schedule recheck if needed."""
    if state.can_cook_any_menu_recipe():
        log.info("Fallback serving: cookable inventory remains — waiting for more clients")
    else:
        log.info("Fallback serving: inventory exhausted for all menu items — phase will end when server transitions")


async def run_deterministic_fallback(phase: str) -> None:
    if phase == "speaking":
        summary = await fallback_speaking()
    elif phase == "closed_bid":
        summary = await fallback_bidding()
    elif phase == "waiting":
        summary = await fallback_waiting()
    elif phase == "serving":
        summary = await fallback_serving()
        await _end_serving_if_exhausted()
    else:
        summary = f"No fallback for phase {phase}"
    log.info("Fallback [%s] done: %s", phase, summary)


async def run_phase(phase: str) -> None:
    """Run one phase with retries; fallback to deterministic if agent keeps failing."""
    async with _phase_lock:
        if phase not in KNOWN_PHASES:
            return

        log.info("=== Phase run: %s ===", phase)
        await _refresh_state()

        # Phase can change while refresh/network calls are in flight.
        if phase != state.phase and not (phase == "speaking" and state.phase == "unknown"):
            log.warning("Skipping %s run after refresh: state phase mismatch (%s)", phase, state.phase)
            return

        gate_err = _phase_readiness_error(phase)
        if gate_err:
            log.warning("Skipping %s run: %s", phase, gate_err)
            return

        if phase == "serving":
            waiting = await _fetch_waiting_meals(state.turn_id)
            # If /meals is temporarily unavailable, do not block the run.
            if waiting is not None and not waiting and not state.prepared_dishes:
                log.info("Skipping serving run: no waiting meals and no prepared dishes")
                return

        if phase == "waiting":
            finalize_bids = getattr(state, "finalize_bid_outcomes", None)
            if callable(finalize_bids):
                finalize_bids(state.inventory)

        # Ensure restaurant is open before serving arrives.
        # Rules: open is allowed in speaking/closed_bid/waiting but NOT in serving.
        # So we open in any pre-serving phase if we detect it's closed.
        if phase in ("speaking", "closed_bid", "waiting") and not state.is_open:
            log.info("Restaurant closed in phase %s, opening it", phase)
            result = await mcp_api.update_restaurant_is_open(True)
            if not _mcp_is_error(result):
                state.is_open = True

        loop = asyncio.get_running_loop()
        max_retries = 1 if phase == "serving" else MAX_AGENT_RETRIES

        for attempt in range(1, max_retries + 1):
            try:
                await loop.run_in_executor(None, _run_agent, phase)
                return
            except Exception as e:
                log.error(
                    "Agent [%s] error (attempt %d/%d): %s",
                    phase,
                    attempt,
                    max_retries,
                    e,
                    exc_info=True,
                )
                if attempt < max_retries:
                    await asyncio.sleep(2 * attempt)

        log.warning("Agent [%s] failed %d times, switching to fallback", phase, max_retries)
        try:
            await run_deterministic_fallback(phase)
        except Exception as e:
            log.error("Fallback [%s] failed: %s", phase, e, exc_info=True)


def request_phase_run(phase: str, reason: str) -> None:
    """Queue one phase run without blocking event handlers."""
    if phase not in KNOWN_PHASES:
        return
    if phase == "serving" and state.phase != "serving":
        return
    if phase in _queued_phases:
        return

    _queued_phases.add(phase)
    _phase_queue.put_nowait(phase)
    log.info("Queued phase run: %s (%s)", phase, reason)


async def phase_worker() -> None:
    """Background worker that serializes all phase runs."""
    while True:
        phase = await _phase_queue.get()
        _queued_phases.discard(phase)

        # game_started may arrive before explicit phase event; allow speaking once in unknown.
        if phase == "speaking" and state.phase == "unknown":
            log.info("Running queued speaking while phase is unknown (startup tolerance)")
        elif phase != state.phase:
            log.info("Skipping queued phase %s (current phase is %s)", phase, state.phase)
            continue

        try:
            await run_phase(phase)
        except Exception as e:
            log.error("Background phase run failed [%s]: %s", phase, e, exc_info=True)


def ensure_phase_worker() -> None:
    global _phase_worker_task
    if _phase_worker_task and not _phase_worker_task.done():
        return
    _phase_worker_task = asyncio.create_task(phase_worker())


async def _debounced_serving_rerun(reason: str) -> None:
    await asyncio.sleep(2.5)
    if state.phase != "serving":
        return
    if state.prepared_dishes:
        request_phase_run("serving", reason)
        return
    waiting = await _fetch_waiting_meals(state.turn_id)
    if waiting:
        request_phase_run("serving", reason)


async def _fetch_waiting_meals(turn_id: int) -> list[dict] | None:
    """Return waiting meals for current restaurant/turn, or None on transient GET errors."""
    if turn_id <= 0:
        return []
    if get_api is None:
        return []
    try:
        meals = await get_api.meals(turn_id)
    except Exception:
        return None
    waiting = [
        m
        for m in (meals or [])
        if (not bool(m.get("executed")))
        and str(m.get("status", "")).strip().lower() == "waiting"
    ]
    return waiting


async def _serving_keepalive_loop() -> None:
    """Poll server cheaply and queue serving only when there is actual work."""
    while True:
        await asyncio.sleep(10.0)
        if state.phase != "serving":
            break
        if state.prepared_dishes:
            request_phase_run("serving", "keepalive_poll_local_work")
            continue
        waiting = await _fetch_waiting_meals(state.turn_id)
        if waiting is None:
            continue
        if waiting:
            request_phase_run("serving", "keepalive_poll_waiting_meals")
            continue
        if not state.can_cook_any_menu_recipe():
            log.info("Serving keepalive: no more cookable items, stopping keepalive")
            break


# -- SSE event handlers -------------------------------------------------------

async def on_game_started(data: dict[str, Any]) -> None:
    state.on_game_started(data)
    _log_turn_kpis(current=False)
    await _refresh_state()
    log.info("Turn %d started -- balance=%.2f", state.turn_id, state.balance)

    # Server may skip explicit speaking event at turn start.
    request_phase_run("speaking", "game_started")


async def on_phase_changed(data: dict[str, Any]) -> None:
    global _serving_keepalive_task
    turn_candidate = data.get("turn_id", data.get("turnId"))
    if turn_candidate is not None and state.turn_id == 0:
        try:
            turn_int = int(turn_candidate)
        except (TypeError, ValueError):
            turn_int = 0
        if turn_int > 0:
            state.turn_id = turn_int
            log.info("Detected turn_id=%d from phase event", state.turn_id)

    phase = data.get("phase", "unknown")
    state.on_phase_changed(phase)

    if phase in KNOWN_PHASES:
        request_phase_run(phase, "phase_changed")
        if phase == "serving":
            # Start keepalive loop to keep polling for clients while we have cookable inventory.
            if _serving_keepalive_task and not _serving_keepalive_task.done():
                _serving_keepalive_task.cancel()
            _serving_keepalive_task = asyncio.create_task(_serving_keepalive_loop())
    elif phase == "stopped":
        if _serving_keepalive_task and not _serving_keepalive_task.done():
            _serving_keepalive_task.cancel()
            _serving_keepalive_task = None
        _log_turn_kpis(current=True)
        if save_snapshot(state):
            log.info("State snapshot saved")
        if state.price_db.save(PRICE_DB_PATH):
            log.info("Price database saved")
        if state.bid_archive.save(BID_ARCHIVE_PATH):
            log.info("Bid archive saved")
        log.info("Turn ended.")


async def on_client_spawned(data: dict[str, Any]) -> None:
    global _serving_debounce_task
    turn_candidate = data.get("turn_id", data.get("turnId"))
    if turn_candidate is not None and state.turn_id == 0:
        try:
            turn_int = int(turn_candidate)
        except (TypeError, ValueError):
            turn_int = 0
        if turn_int > 0:
            state.turn_id = turn_int
            log.info("Detected turn_id=%d from client event", state.turn_id)
    state.on_client_spawned(data)

    if state.phase == "serving":
        if _serving_debounce_task and not _serving_debounce_task.done():
            _serving_debounce_task.cancel()
        _serving_debounce_task = asyncio.create_task(_debounced_serving_rerun("client_spawned"))


async def on_preparation_complete(data: dict[str, Any]) -> None:
    global _serving_debounce_task
    state.on_preparation_complete(data)

    if state.phase == "serving" and state.unserved_clients():
        if _serving_debounce_task and not _serving_debounce_task.done():
            _serving_debounce_task.cancel()
        _serving_debounce_task = asyncio.create_task(_debounced_serving_rerun("preparation_complete"))


async def on_message(data: dict[str, Any]) -> None:
    payload = str(data.get("payload", ""))
    record_market = getattr(state, "record_market_activity_message", None)
    if callable(record_market):
        record_market(payload)
    log.info("Message from %s: %s", data.get("sender", "?"), payload)


async def on_game_reset(data: dict[str, Any]) -> None:
    import glob as globmod
    for f in globmod.glob("agent_memory_*.json"):
        try:
            os.remove(f)
            log.info("Cleared memory file: %s", f)
        except OSError:
            pass
    state.on_game_reset()


EVENT_HANDLERS = {
    "game_started": on_game_started,
    "game_phase_changed": on_phase_changed,
    "game_reset": on_game_reset,
    "client_spawned": on_client_spawned,
    "preparation_complete": on_preparation_complete,
    "message": on_message,
}


async def dispatch_event(event_type: str, event_data: dict[str, Any]) -> None:
    handler = EVENT_HANDLERS.get(event_type)
    if not handler:
        return
    try:
        await handler(event_data)
    except Exception as exc:
        log.error("Handler %s failed: %s", event_type, exc, exc_info=True)


# -- SSE parser ---------------------------------------------------------------

async def handle_line(raw_line: bytes) -> None:
    if not raw_line:
        return

    line = raw_line.decode("utf-8", errors="ignore").strip()
    if not line:
        return

    if line.startswith("data:"):
        payload = line[5:].strip()
        if payload == "connected":
            log.info("SSE connected")
            return
        line = payload

    try:
        event_json = json.loads(line)
    except json.JSONDecodeError:
        return

    event_type = event_json.get("type", "unknown")
    event_data = event_json.get("data", {})
    if not isinstance(event_data, dict):
        event_data = {"value": event_data}

    await dispatch_event(event_type, event_data)


async def init_state() -> None:
    """Fetch initial state on connect (handles mid-turn joins and reconnects)."""
    if load_snapshot(state):
        log.info("Restored state snapshot (demand/bid memory preserved)")
    if state.price_db.load(PRICE_DB_PATH):
        log.info("Restored price database from %s", PRICE_DB_PATH)
    if state.bid_archive.load(BID_ARCHIVE_PATH):
        log.info("Restored bid archive from %s", BID_ARCHIVE_PATH)
    await _refresh_state()
    log.info("Init: balance=%.2f, inventory=%d items", state.balance, sum(state.inventory.values()))
    log.info("Init: loaded %d recipes", len(state.recipes))


async def listen(session: aiohttp.ClientSession) -> None:
    url = f"{BASE_URL}/events/{TEAM_ID}"
    headers = {"Accept": "text/event-stream", "x-api-key": TEAM_API_KEY}

    async with session.get(url, headers=headers) as resp:
        resp.raise_for_status()
        log.info("SSE connection open")
        await init_state()

        async for line in resp.content:
            await handle_line(line)


async def listen_forever(session: aiohttp.ClientSession) -> None:
    """SSE with auto-reconnect on drop."""
    while True:
        try:
            await listen(session)
            log.warning("SSE connection closed by server")
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            log.warning("SSE error: %s", e)
        except Exception as e:
            log.error("SSE unexpected error: %s", e, exc_info=True)

        log.info("Reconnecting SSE in %ds...", SSE_RECONNECT_DELAY)
        await asyncio.sleep(SSE_RECONNECT_DELAY)


async def main() -> None:
    global get_api, mcp_api
    log.info("Starting -- team=%s", TEAM_ID)

    timeout = aiohttp.ClientTimeout(total=None, sock_connect=15, sock_read=None)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        get_api = GameGET(session, BASE_URL, TEAM_API_KEY, TEAM_ID)
        mcp_api = GameMCP(session, BASE_URL, TEAM_API_KEY)
        init_agent(get_api, mcp_api, state, REGOLO_API_KEY)
        ensure_phase_worker()

        log.info("Agent ready, connecting SSE...")
        await listen_forever(session)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Stopped")
