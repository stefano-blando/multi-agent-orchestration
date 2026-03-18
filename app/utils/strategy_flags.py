"""Environment-driven strategy flags and shared heuristics."""

from __future__ import annotations

import os


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


def strat_v2_bid_enabled() -> bool:
    return env_flag("STRAT_V2_BID", default=True)


def strat_v2_report_enabled() -> bool:
    return env_flag("STRAT_V2_REPORT", default=True)


def strat_v2_pricing_enabled() -> bool:
    return env_flag("STRAT_V2_PRICING", default=False)


def strat_value_bidding_enabled() -> bool:
    return env_flag("STRAT_VALUE_BID", default=True)


def strat_value_pricing_enabled() -> bool:
    return env_flag("STRAT_VALUE_PRICE", default=True)


def menu_max_items(default: int = 5) -> int:
    raw = os.getenv("MENU_MAX_ITEMS")
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return max(3, min(10, value))


def rolling_revenue_budget_cap(
    balance: float,
    rolling_revenue_avg: float,
    num_cookable: int,
    low_demand_mode: bool,
) -> int | None:
    """Return an extra cap derived from recent revenue; None if not applicable."""
    if rolling_revenue_avg <= 0:
        return None

    if low_demand_mode:
        multiplier = 0.85
        floor = 120
        ceiling = 800
    else:
        multiplier = 1.10
        floor = 180
        ceiling = 1400

    if num_cookable >= 3:
        multiplier *= 0.80
    elif num_cookable >= 1:
        multiplier *= 0.95

    cap_from_revenue = int(max(floor, min(ceiling, rolling_revenue_avg * multiplier)))
    cap_from_balance = int(max(0.0, balance * 0.60))
    if cap_from_balance <= 0:
        return 0
    return min(cap_from_revenue, cap_from_balance)
