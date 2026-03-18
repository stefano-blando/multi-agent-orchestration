"""datapizza-ai tool definitions that wrap our API calls."""

from __future__ import annotations

import asyncio
import difflib
import json
import logging
import re
import time

from datapizza.tools import tool

from api_get import GameGET
from api_mcp import GameMCP
from game_state import GameState

log = logging.getLogger(__name__)

# ── Globals — set once by init_tools() ────────────────────────────────
_get: GameGET | None = None
_mcp: GameMCP | None = None
_state: GameState | None = None
_loop: asyncio.AbstractEventLoop | None = None
_valid_meal_ids: set[str] = set()
_attempted_meal_ids: set[str] = set()  # IDs we already tried serving (success or permanent fail)
_attempted_turn_id: int | None = None
_meal_id_to_customer_id: dict[str, str] = {}
_customer_id_to_meal_id: dict[str, str] = {}
_meal_request_by_id: dict[str, str] = {}
_demand_counted_meal_ids: set[str] = set()
_recently_ready_until: dict[str, float] = {}
_inflight_prepares_until: dict[str, float] = {}
_prepare_started_at: dict[str, float] = {}
_prepare_min_ready_at: dict[str, float] = {}

# Errors that should NOT be retried — the LLM should skip this client.
_PERMANENT_ERRORS = {
    "not waiting",
    "not found",
    "invalid",
    "already served",
    "already executed",
    "does not match requested dish",
}
_EARNED_RE = re.compile(r"\bearned\s+(\d+)\b", re.IGNORECASE)


def init_tools(get: GameGET, mcp: GameMCP, state: GameState) -> None:
    global _get, _mcp, _state, _loop
    _get = get
    _mcp = mcp
    _state = state
    _loop = asyncio.get_event_loop()


def _run(coro):
    """Run async coroutine from sync tool context.

    agent.run() executes in a thread (run_in_executor), but the aiohttp
    session lives on the main event loop. We schedule the coroutine on
    the main loop and wait for the result from this thread.
    """
    future = asyncio.run_coroutine_threadsafe(coro, _loop)
    return future.result(timeout=30)


def _norm(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _operation_id(action: str, meal_id: str, dish_name: str) -> str:
    turn_id = int(getattr(_state, "turn_id", 0) or 0)
    return f"{action}:{turn_id}:{meal_id}:{_norm(dish_name)}"


def _all_dish_names() -> list[str]:
    names: set[str] = set()
    for r in (_state.recipes or []):
        name = str(r.get("name", "")).strip()
        if name:
            names.add(name)

    menu = _state.menu
    items = []
    if isinstance(menu, dict):
        items = menu.get("items", []) or []
    elif isinstance(menu, list):
        items = menu

    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if name:
            names.add(name)

    return sorted(names)


def _canonicalize_dish_name(dish_name: str) -> str:
    clean = " ".join(str(dish_name).split()).strip()
    if not clean:
        return clean

    names = _all_dish_names()
    if not names:
        return clean
    if clean in names:
        return clean

    by_norm: dict[str, str] = {}
    for n in names:
        by_norm.setdefault(_norm(n), n)

    clean_norm = _norm(clean)
    if clean_norm in by_norm:
        return by_norm[clean_norm]

    matches = difflib.get_close_matches(clean_norm, list(by_norm.keys()), n=1, cutoff=0.84)
    if matches:
        fixed = by_norm[matches[0]]
        log.warning("Canonicalized dish name '%s' -> '%s'", clean, fixed)
        return fixed
    return clean


def _dish_matches_request(dish_name: str, request_text: str) -> bool:
    """Match request by exact dish name or by ingredient set compatibility."""
    d = _norm(dish_name)
    r = _norm(request_text)
    if not d or not r:
        return False
    pattern = r"(?<!\w)" + re.escape(d) + r"(?!\w)"
    if bool(re.search(pattern, r)):
        return True

    # Ingredient-based requests (e.g. "something with A, B, C").
    requested_ings = _requested_ingredients_from_text(request_text)
    if not requested_ings:
        return False

    recipe_ings: set[str] = set()
    for recipe in (_state.recipes or []):
        if str(recipe.get("name", "")).strip() == dish_name:
            recipe_ings = {_norm(str(ing)) for ing in (recipe.get("ingredients", {}) or {}).keys()}
            break
    if not recipe_ings:
        return False
    return requested_ings.issubset(recipe_ings)


def _requested_ingredients_from_text(request_text: str) -> set[str]:
    """Extract known ingredient names mentioned in a free-form request."""
    r = _norm(request_text)
    if not r:
        return set()

    ingredient_names: set[str] = set()
    for recipe in (_state.recipes or []):
        for ing in (recipe.get("ingredients", {}) or {}).keys():
            ing_name = str(ing).strip()
            if ing_name:
                ingredient_names.add(ing_name)

    found: set[str] = set()
    for ing in ingredient_names:
        ing_n = _norm(ing)
        if not ing_n:
            continue
        pattern = r"(?<!\w)" + re.escape(ing_n) + r"(?!\w)"
        if re.search(pattern, r):
            found.add(ing_n)
    return found


def _kitchen_dish_names() -> set[str]:
    """Fetch current kitchen from server and return normalized dish names."""
    try:
        data = _run(_get.restaurant())
        if isinstance(data, dict):
            _state.update_restaurant(data)
        kitchen = data.get("kitchen", []) if isinstance(data, dict) else []
    except Exception as e:
        log.warning("Kitchen refresh failed: %s", e)
        return set()

    names: set[str] = set()
    for item in kitchen or []:
        if isinstance(item, dict):
            name = (
                item.get("name")
                or item.get("dish")
                or item.get("dishName")
                or item.get("recipeName")
            )
        else:
            name = str(item)
        if name:
            names.add(_norm(str(name)))
    return names


def _dish_in_kitchen(canonical_name: str) -> bool:
    return _norm(canonical_name) in _kitchen_dish_names()


def _dish_prep_seconds(canonical_name: str) -> float:
    for recipe in (_state.recipes or []):
        if str(recipe.get("name", "")) == canonical_name:
            try:
                prep_ms = int(recipe.get("preparationTimeMs", 0))
            except (TypeError, ValueError):
                prep_ms = 0
            return max(0.5, prep_ms / 1000.0)
    return 3.0


def _min_ready_elapsed(canonical_name: str) -> bool:
    min_ready_at = _prepare_min_ready_at.get(canonical_name)
    if min_ready_at is None:
        return True
    return time.time() >= min_ready_at


def _is_not_ready_error(err_text: str) -> bool:
    e = err_text.lower()
    return (
        "dish not found in kitchen" in e
        or "not found in kitchen" in e
        or "not ready" in e
    )


def _wait_until_probably_ready(canonical_dish: str, timeout_s: float = 10.0) -> bool:
    """Wait for strong readiness evidence to mitigate MCP eventual-consistency races."""
    deadline = time.time() + max(0.1, timeout_s)
    kitchen_hits = 0

    while time.time() < deadline:
        if canonical_dish in _state.prepared_dishes and _min_ready_elapsed(canonical_dish):
            return True

        if _min_ready_elapsed(canonical_dish) and _dish_in_kitchen(canonical_dish):
            kitchen_hits += 1
            if kitchen_hits >= 3:
                return True
        else:
            kitchen_hits = 0

        time.sleep(0.55)

    return False


def _mcp_call(coro) -> str:
    """Run an MCP action and handle errors gracefully.

    If the server returns isError=true, log it and return a short
    error message to the agent instead of the raw JSON.
    """
    try:
        result = _run(coro)
    except Exception as e:
        log.error("MCP call failed: %s", e)
        return json.dumps({"error": str(e)})
    if isinstance(result, dict) and result.get("isError"):
        texts = [c.get("text", "") for c in result.get("content", [])]
        msg = "; ".join(texts) or "Unknown error"
        log.warning("MCP error: %s", msg)
        return json.dumps({"error": msg})
    return json.dumps(result, ensure_ascii=False)


def _extract_earned(parsed: dict) -> int:
    if not isinstance(parsed, dict):
        return 0
    direct = parsed.get("earned")
    try:
        if direct is not None:
            return max(0, int(direct))
    except (TypeError, ValueError):
        pass

    content = parsed.get("content", [])
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


# ── GET tools ─────────────────────────────────────────────────────────

@tool
def get_restaurant_info() -> str:
    """Get our restaurant: balance, inventory, menu, open status."""
    data = _run(_get.restaurant())
    _state.update_restaurant(data)
    # Return a compact payload to reduce prompt bloat.
    menu = data.get("menu", {})
    menu_items = []
    if isinstance(menu, dict):
        menu_items = menu.get("items", []) or []
    elif isinstance(menu, list):
        menu_items = menu

    kitchen_raw = data.get("kitchen", []) if isinstance(data, dict) else []
    kitchen_names: list[str] = []
    for item in kitchen_raw or []:
        if isinstance(item, dict):
            name = item.get("name") or item.get("dish") or item.get("dishName") or item.get("recipeName")
        else:
            name = item
        if name:
            kitchen_names.append(str(name))

    compact = {
        "id": data.get("id"),
        "name": data.get("name"),
        "balance": data.get("balance"),
        "inventory": data.get("inventory", {}),
        "reputation": data.get("reputation"),
        "isOpen": data.get("isOpen", data.get("is_open")),
        "kitchen": kitchen_names,
        "menu": {"items": menu_items},
    }
    return json.dumps(compact, ensure_ascii=False)


@tool
def get_recipes() -> str:
    """Get all available recipes with ingredients and prep time."""
    data = _run(_get.recipes())
    _state.update_recipes(data)
    return json.dumps(data, ensure_ascii=False)


@tool
def get_meals(turn_id: int) -> str:
    """Get meals for a turn. Returns compact waiting meals only; use str(id) as client_id for serve_dish."""
    global _attempted_turn_id
    if int(turn_id) <= 0:
        log.warning("get_meals called with invalid turn_id=%s; returning empty list", turn_id)
        return "[]"
    data = _run(_get.meals(turn_id))
    # SSE client_spawned can arrive a bit before /meals reflects the new meal.
    # If we already know there are pending clients, do a short consistency wait.
    if not data and _state.unserved_clients():
        for _ in range(4):
            time.sleep(0.6)
            data = _run(_get.meals(turn_id))
            if data:
                break

    # Track valid meal IDs so serve_dish can reject invented ones.
    _valid_meal_ids.clear()
    _meal_id_to_customer_id.clear()
    _customer_id_to_meal_id.clear()
    _meal_request_by_id.clear()
    if _attempted_turn_id != turn_id:
        _attempted_meal_ids.clear()
        _demand_counted_meal_ids.clear()
        _attempted_turn_id = turn_id
    for m in data:
        mid = m.get("id")
        if mid is not None:
            mid_s = str(mid)
            _valid_meal_ids.add(mid_s)
            executed = bool(m.get("executed"))
            status = str(m.get("status", "")).strip().lower()
            if executed or status == "served":
                _state.set_meal_runtime_state(mid_s, "served")
            else:
                _state.set_meal_runtime_state(mid_s, "waiting")
            req = str(m.get("orderText") or m.get("request") or "")
            if req:
                _meal_request_by_id[mid_s] = req
            start_time = str(m.get("startTime") or "")
            if start_time:
                _state.record_meal_start_time(mid_s, start_time)
            if mid_s not in _demand_counted_meal_ids:
                extract_fn = getattr(_state, "_extract_requested_dish", None)
                dish_name = ""
                if callable(extract_fn):
                    dish_name = str(extract_fn(req) or "")
                record_fn = getattr(_state, "record_meal_observed", None)
                if callable(record_fn):
                    record_fn(mid_s, dish_name)
                elif dish_name:
                    _state.dish_demand[dish_name] = _state.dish_demand.get(dish_name, 0) + 1
                _demand_counted_meal_ids.add(mid_s)
                # If a waiting request maps neither to a known dish name nor to
                # any known ingredients, mark it as attempted/skipped to avoid
                # infinite serving loops.
                if (
                    not executed
                    and status == "waiting"
                    and req
                    and not dish_name
                    and not _requested_ingredients_from_text(req)
                ):
                    _attempted_meal_ids.add(mid_s)
                    _state.set_meal_runtime_state(
                        mid_s,
                        "skipped",
                        reason="requested_dish_not_recognized",
                    )
            customer_id = m.get("customerId")
            if customer_id is not None:
                cid_s = str(customer_id)
                _valid_meal_ids.add(cid_s)
                _meal_id_to_customer_id[mid_s] = cid_s
                _customer_id_to_meal_id[cid_s] = mid_s
                if executed or status == "served":
                    _state.set_meal_runtime_state(cid_s, "served")
                else:
                    _state.set_meal_runtime_state(cid_s, "waiting")
                if req:
                    _meal_request_by_id[cid_s] = req
                if start_time:
                    _state.record_meal_start_time(cid_s, start_time)
                if mid_s in _attempted_meal_ids:
                    _attempted_meal_ids.add(cid_s)

    # Backfill pending SSE clients with true /meals IDs when possible.
    unserved = [m for m in data if not m.get("executed")]
    for client in _state.pending_clients:
        if client.client_id:
            continue
        matches = [
            m for m in unserved
            if _norm(str(m.get("clientName") or (m.get("customer") or {}).get("name") or "")) == _norm(client.name)
            and _norm(str(m.get("orderText") or m.get("request") or "")) == _norm(client.order_text)
        ]
        if len(matches) == 1:
            client.client_id = str(matches[0].get("id"))

    # Keep LLM context small: return only waiting meals with compact fields.
    waiting_compact: list[dict] = []
    for m in data:
        mid_s = str(m.get("id", "")).strip()
        if mid_s and mid_s in _attempted_meal_ids:
            continue
        executed = bool(m.get("executed"))
        status = str(m.get("status", "")).strip().lower()
        if executed or status != "waiting":
            continue
        customer = m.get("customer") if isinstance(m.get("customer"), dict) else {}
        waiting_compact.append(
            {
                "id": m.get("id"),
                "turnId": m.get("turnId"),
                "customerId": m.get("customerId"),
                "restaurantId": m.get("restaurantId"),
                "request": m.get("orderText") or m.get("request"),
                "startTime": m.get("startTime"),
                "servedDishId": m.get("servedDishId"),
                "status": m.get("status"),
                "customer": {"name": customer.get("name")},
                "executed": m.get("executed"),
            }
        )

    return json.dumps(waiting_compact, ensure_ascii=False)


@tool
def get_market() -> str:
    """Get active market entries (buy/sell from all restaurants)."""
    data = _run(_get.market_entries())
    _state.update_market(data)
    return json.dumps(data, ensure_ascii=False)


@tool
def get_competitors() -> str:
    """Get overview of all restaurants."""
    data = _run(_get.restaurants())
    return json.dumps(data, ensure_ascii=False)


# ── MCP action tools ─────────────────────────────────────────────────

@tool
def set_menu(items: list) -> str:
    """Set menu. items: [{"name": "dish_name", "price": 50}, ...] — price int, max 1000."""
    if str(getattr(_state, "phase", "")).strip().lower() == "serving":
        return json.dumps({"error": "Blocked: cannot set menu during serving phase."}, ensure_ascii=False)
    resp = _mcp_call(_mcp.save_menu(items))
    if len(items) <= 5:
        return resp
    try:
        parsed = json.loads(resp)
    except (json.JSONDecodeError, TypeError):
        return resp
    err = str(parsed.get("error", "") or "").lower()
    if any(token in err for token in ("too many", "max", "limit", "items")):
        trimmed = items[:5]
        log.warning("set_menu fallback: retrying with %d items", len(trimmed))
        return _mcp_call(_mcp.save_menu(trimmed))
    return resp


@tool
def place_bid(bids: list) -> str:
    """Place closed bids. bids: [{"ingredient": "X", "bid": 5, "quantity": 3}, ...] — bid = price per unit (int)."""
    if not bids:
        # MCP rejects empty arrays; treat as explicit no-op success.
        return json.dumps(
            {
                "isError": False,
                "content": [{"type": "text", "text": "No bids to place (noop)"}],
            },
            ensure_ascii=False,
        )
    resp = _mcp_call(_mcp.closed_bid(bids))
    try:
        parsed = json.loads(resp)
    except (json.JSONDecodeError, TypeError):
        parsed = {}
    if not parsed.get("error"):
        record_bids = getattr(_state, "record_bids_submitted", None)
        if callable(record_bids):
            record_bids(bids, _state.inventory)
    return resp


@tool
def prepare_dish(dish_name: str) -> str:
    """Start preparing a dish. Returns immediately. Call wait_for_preparation() next."""
    canonical_name = _canonicalize_dish_name(dish_name)
    now = time.time()
    # Guard against duplicate prepare while the same dish is already in-flight.
    if _inflight_prepares_until.get(canonical_name, 0.0) > now:
        record_prepare = getattr(_state, "record_prepare_attempt", None)
        if callable(record_prepare):
            record_prepare(True)
        return json.dumps(
            {
                "isError": False,
                "content": [{"type": "text", "text": f"{canonical_name} already preparing"}],
            },
            ensure_ascii=False,
        )
    # Guard against immediate duplicate prepare after a recent "ready" signal.
    if _recently_ready_until.get(canonical_name, 0.0) > now:
        if canonical_name in _state.prepared_dishes or _dish_in_kitchen(canonical_name):
            record_prepare = getattr(_state, "record_prepare_attempt", None)
            if callable(record_prepare):
                record_prepare(True)
            return json.dumps(
                {
                    "isError": False,
                    "content": [{"type": "text", "text": f"{canonical_name} already prepared"}],
                },
                ensure_ascii=False,
            )
        # Stale local readiness marker: clear and proceed with real prepare.
        _recently_ready_until.pop(canonical_name, None)
    if canonical_name in _state.prepared_dishes:
        if _dish_in_kitchen(canonical_name):
            record_prepare = getattr(_state, "record_prepare_attempt", None)
            if callable(record_prepare):
                record_prepare(True)
            return json.dumps(
                {
                    "isError": False,
                    "content": [{"type": "text", "text": f"{canonical_name} already prepared"}],
                },
                ensure_ascii=False,
            )
        # Stale SSE dish-ready event not reflected in server kitchen.
        while canonical_name in _state.prepared_dishes:
            _state.prepared_dishes.remove(canonical_name)
    resp = _mcp_call(_mcp.prepare_dish(canonical_name))
    try:
        parsed = json.loads(resp)
    except (json.JSONDecodeError, TypeError):
        parsed = {}
    if not parsed.get("error"):
        record_prepare = getattr(_state, "record_prepare_attempt", None)
        if callable(record_prepare):
            record_prepare(True)
        now_ts = time.time()
        prep_s = _dish_prep_seconds(canonical_name)
        # Preparation times are usually <= ~10s; keep a broad in-flight guard.
        _inflight_prepares_until[canonical_name] = now_ts + max(18.0, prep_s + 12.0)
        _prepare_started_at[canonical_name] = now_ts
        # Avoid optimistic "ready" before a plausible fraction of prep time elapsed.
        _prepare_min_ready_at[canonical_name] = now_ts + min(12.0, max(1.5, prep_s * 0.75))
        for r in (_state.recipes or []):
            if r["name"] == canonical_name:
                for ing, qty in r.get("ingredients", {}).items():
                    cur = _state.inventory.get(ing, 0)
                    _state.inventory[ing] = max(0, cur - qty)
                break
    else:
        record_prepare = getattr(_state, "record_prepare_attempt", None)
        if callable(record_prepare):
            record_prepare(False)
    return resp


@tool
def wait_for_preparation(dish_name: str) -> str:
    """Block until dish_name appears in prepared dishes (max 15s). Call after prepare_dish()."""
    canonical_name = _canonicalize_dish_name(dish_name)
    deadline = time.time() + 18
    next_kitchen_check_at = 0.0
    kitchen_hits = 0
    while time.time() < deadline:
        now = time.time()
        if canonical_name in _state.prepared_dishes:
            if not _min_ready_elapsed(canonical_name):
                # Very early "ready" is usually stale/out-of-order; wait a bit more.
                time.sleep(0.3)
                continue
            if not _dish_in_kitchen(canonical_name):
                # Drop stale local ready marker and keep waiting.
                while canonical_name in _state.prepared_dishes:
                    _state.prepared_dishes.remove(canonical_name)
                time.sleep(0.3)
                continue
            _inflight_prepares_until.pop(canonical_name, None)
            _prepare_min_ready_at.pop(canonical_name, None)
            _prepare_started_at.pop(canonical_name, None)
            _recently_ready_until[canonical_name] = time.time() + 10.0
            _state.prepared_dishes.remove(canonical_name)
            return json.dumps({"status": "ready", "dish": canonical_name})
        if now >= next_kitchen_check_at:
            next_kitchen_check_at = now + 0.8
            if _min_ready_elapsed(canonical_name) and _dish_in_kitchen(canonical_name):
                kitchen_hits += 1
            else:
                kitchen_hits = 0
            if kitchen_hits >= 3:
                _inflight_prepares_until.pop(canonical_name, None)
                _prepare_min_ready_at.pop(canonical_name, None)
                _prepare_started_at.pop(canonical_name, None)
                _recently_ready_until[canonical_name] = time.time() + 10.0
                return json.dumps({"status": "ready", "dish": canonical_name})
        time.sleep(0.5)
    _inflight_prepares_until.pop(canonical_name, None)
    _prepare_min_ready_at.pop(canonical_name, None)
    _prepare_started_at.pop(canonical_name, None)
    return json.dumps({"error": f"Timeout waiting for {canonical_name}"})


@tool
def serve_dish(dish_name: str, client_id: str) -> str:
    """Serve a prepared dish to a client. client_id = str(meal.id) from get_meals."""
    # Normalize ID to string of int (strip quotes, whitespace).
    canonical_dish = _canonicalize_dish_name(dish_name)
    clean_id = str(client_id).strip().strip('"').strip("'")
    canonical_meal_id = _customer_id_to_meal_id.get(clean_id, clean_id)
    op_id = _operation_id("serve", canonical_meal_id, canonical_dish)
    op_get = getattr(_state, "get_operation", None)
    if callable(op_get):
        cached = op_get(op_id)
        if isinstance(cached, dict):
            cached_resp = str(cached.get("response", "") or "")
            if cached_resp:
                return cached_resp

    if _valid_meal_ids and clean_id not in _valid_meal_ids:
        log.warning("serve_dish blocked: client_id=%s not in valid meal IDs", clean_id)
        resp = json.dumps({"error": f"Invalid client_id {clean_id}. Use IDs from get_meals."})
        _state.set_meal_runtime_state(canonical_meal_id, "failed", dish_name=canonical_dish, reason="invalid_client_id", operation_id=op_id)
        _state.record_operation(op_id, False, resp, "invalid_client_id")
        return resp

    if canonical_meal_id in _attempted_meal_ids:
        log.warning("serve_dish blocked: client_id=%s already attempted", clean_id)
        resp = json.dumps({"error": f"Already attempted client_id {clean_id}. Skip to next client."})
        _state.record_operation(op_id, False, resp, "already_attempted")
        return resp

    req = _meal_request_by_id.get(clean_id) or _meal_request_by_id.get(canonical_meal_id, "")
    if req and not _dish_matches_request(canonical_dish, req):
        log.warning(
            "serve_dish blocked: dish '%s' does not match request for client_id=%s",
            canonical_dish,
            clean_id,
        )
        resp = json.dumps({"error": f"Dish '{canonical_dish}' does not match requested dish for client {clean_id}."})
        _attempted_meal_ids.add(canonical_meal_id)
        _state.set_meal_runtime_state(canonical_meal_id, "failed", dish_name=canonical_dish, reason="request_mismatch", operation_id=op_id)
        _state.record_operation(op_id, False, resp, "request_mismatch")
        return resp

    alias_id = _meal_id_to_customer_id.get(clean_id) or _customer_id_to_meal_id.get(clean_id)
    _state.set_meal_runtime_state(canonical_meal_id, "serving", dish_name=canonical_dish, operation_id=op_id)

    def _serve_once(target_id: str) -> tuple[str, str]:
        raw = _mcp_call(_mcp.serve_dish(canonical_dish, target_id))
        try:
            parsed_local = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            parsed_local = {}
        return raw, str(parsed_local.get("error", "") or "")

    def _serve_with_alias_fallback() -> tuple[str, str]:
        raw, err_local = _serve_once(clean_id)
        if err_local:
            err_lower_local = err_local.lower()
            if "not waiting" in err_lower_local and alias_id and alias_id != clean_id:
                alias_resp = _serve_once(alias_id)[0]
                try:
                    alias_parsed = json.loads(alias_resp)
                except (json.JSONDecodeError, TypeError):
                    alias_parsed = {}
                alias_err = str(alias_parsed.get("error", "") or "")
                if not alias_err:
                    return alias_resp, ""
                return alias_resp, alias_err or err_local
        return raw, err_local

    resp, err = _serve_with_alias_fallback()
    if err and _is_not_ready_error(err):
        _state.set_meal_runtime_state(canonical_meal_id, "preparing", dish_name=canonical_dish, reason="not_ready", operation_id=op_id)
        # Race condition guard: kitchen readiness can lag behind prepare/wait events.
        for wait_budget in (2.5, 3.0, 4.0):
            if _wait_until_probably_ready(canonical_dish, timeout_s=wait_budget):
                retry_resp, retry_err = _serve_with_alias_fallback()
                resp, err = retry_resp, retry_err
                if not err:
                    break
                if not _is_not_ready_error(err):
                    break

    if err:
        err_lower = err.lower()
        record_serve = getattr(_state, "record_serve_attempt", None)
        if callable(record_serve):
            record_serve(False, not_ready_error=_is_not_ready_error(err))
        if any(pe in err_lower for pe in _PERMANENT_ERRORS):
            _attempted_meal_ids.add(canonical_meal_id)
            log.info("serve_dish permanent error for %s: %s — skip", canonical_meal_id, err)
        _state.set_meal_runtime_state(canonical_meal_id, "failed", dish_name=canonical_dish, reason=err, operation_id=op_id)
        _state.record_operation(op_id, False, resp, err)
        return resp

    _attempted_meal_ids.add(canonical_meal_id)
    _inflight_prepares_until.pop(canonical_dish, None)
    _recently_ready_until.pop(canonical_dish, None)
    _prepare_min_ready_at.pop(canonical_dish, None)
    _prepare_started_at.pop(canonical_dish, None)
    try:
        parsed = json.loads(resp)
    except (json.JSONDecodeError, TypeError):
        parsed = {}
    earned = _extract_earned(parsed)
    latency_ms = _state.meal_latency_ms(canonical_meal_id) or _state.meal_latency_ms(clean_id)
    record_serve = getattr(_state, "record_serve_attempt", None)
    if callable(record_serve):
        record_serve(True, latency_ms=latency_ms)
    record_serve = getattr(_state, "record_dish_served", None)
    if callable(record_serve):
        record_serve(canonical_dish, earned)
    else:
        _state.served_this_turn += 1
    for c in _state.pending_clients:
        if c.client_id in (clean_id, canonical_meal_id):
            c.served = True
            break
    _state.set_meal_runtime_state(canonical_meal_id, "served", dish_name=canonical_dish, operation_id=op_id)
    _state.record_operation(op_id, True, resp, "")
    return resp


@tool
def open_close_restaurant(is_open: bool) -> str:
    """Open (true) or close (false) the restaurant."""
    return _mcp_call(_mcp.update_restaurant_is_open(is_open))


@tool
def market_sell(ingredient_name: str, quantity: int, price: int) -> str:
    """Create a SELL entry on the market. price = TOTAL for all units (int)."""
    return _mcp_call(_mcp.create_market_entry("SELL", ingredient_name, quantity, price))


@tool
def market_buy_create(ingredient_name: str, quantity: int, price: int) -> str:
    """Create a BUY order on the market. price = TOTAL for all units (int)."""
    return _mcp_call(_mcp.create_market_entry("BUY", ingredient_name, quantity, price))


@tool
def market_execute(market_entry_id: int) -> str:
    """Accept an existing market entry by its ID."""
    return _mcp_call(_mcp.execute_transaction(market_entry_id))


@tool
def market_delete(market_entry_id: int) -> str:
    """Delete your own market entry."""
    return _mcp_call(_mcp.delete_market_entry(market_entry_id))


@tool
def get_bid_archive_summary(ingredient: str) -> str:
    """Get historical price analysis for an ingredient from past auctions."""
    summary = _state.bid_archive.summary_for_ingredient(ingredient)
    return json.dumps(summary, ensure_ascii=False)


@tool
def get_bid_archive_cheapest() -> str:
    """Get the 10 cheapest ingredients based on historical auction data."""
    cheapest = _state.bid_archive.cheapest_ingredients(top_n=10)
    result = [{"ingredient": ing, "avg_price": round(price, 1)} for ing, price in cheapest]
    return json.dumps(result, ensure_ascii=False)


@tool
def get_bid_archive_price_check(ingredients: list) -> str:
    """Get estimated prices for a list of ingredients based on past auctions."""
    result = []
    for ing in ingredients:
        ing_str = str(ing).strip()
        if not ing_str:
            continue
        avg = _state.bid_archive.avg_clearing_price(ing_str)
        records = _state.bid_archive._recent_records(ing_str, 20)
        n_obs = len([r for r in records if r["qty_won"] > 0])
        suggested = int(round(avg * 1.05)) if avg > 0 else 0
        result.append({
            "ingredient": ing_str,
            "avg_price": round(avg, 1),
            "suggested_bid": suggested,
            "confidence": n_obs,
        })
    return json.dumps(result, ensure_ascii=False)


@tool(end=True)
def end_phase(summary: str) -> str:
    """Call when done with this phase. Provide a brief summary of actions taken."""
    return summary


# ── All tools list (imported by agent.py) ─────────────────────────────

ALL_TOOLS = [
    get_restaurant_info,
    get_recipes,
    get_meals,
    get_market,
    get_competitors,
    set_menu,
    place_bid,
    prepare_dish,
    wait_for_preparation,
    serve_dish,
    open_close_restaurant,
    market_sell,
    market_buy_create,
    market_execute,
    market_delete,
    get_bid_archive_summary,
    get_bid_archive_cheapest,
    get_bid_archive_price_check,
    end_phase,
]
