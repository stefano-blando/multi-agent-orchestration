"""Persistent price history database for ingredient bid intelligence.

Stores per-ingredient clearing price history across turns (up to 30 data points).
Provides moving average signals (EMA/SMA) and trend detection for smarter bidding.
Persists to a JSON file so data survives process restarts.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile

log = logging.getLogger(__name__)

_HISTORY_CAP = 30  # Keep last 30 price observations per ingredient
_DEFAULT_PATH = "price_history.json"


class PriceDatabase:
    """Tracks per-ingredient price history with moving average computation."""

    def __init__(self) -> None:
        self._prices: dict[str, list[float]] = {}  # ingredient -> [price history]

    # ── Data Ingestion ─────────────────────────────────────────────────

    def update(self, ingredient: str, avg_price: float) -> None:
        """Record a new clearing price observation for an ingredient."""
        if not ingredient or avg_price <= 0:
            return
        bucket = self._prices.setdefault(str(ingredient), [])
        bucket.append(float(avg_price))
        if len(bucket) > _HISTORY_CAP:
            self._prices[str(ingredient)] = bucket[-_HISTORY_CAP:]

    # ── Signal Computation ─────────────────────────────────────────────

    def get_history(self, ingredient: str) -> list[float]:
        """Return full price history for an ingredient (oldest first)."""
        return list(self._prices.get(ingredient, []))

    def get_sma(self, ingredient: str, window: int = 10) -> float:
        """Simple moving average of the last N observations."""
        prices = self._prices.get(ingredient, [])
        if not prices:
            return 0.0
        recent = prices[-window:]
        return sum(recent) / len(recent)

    def get_ema(self, ingredient: str, window: int = 8) -> float:
        """Exponential moving average — more weight on recent prices.

        Uses alpha = 2 / (N+1) smoothing factor (standard EMA formula).
        Returns 0.0 if no data available.
        """
        prices = self._prices.get(ingredient, [])
        if not prices:
            return 0.0
        recent = prices[-window:]
        if len(recent) == 1:
            return float(recent[0])
        alpha = 2.0 / (len(recent) + 1)
        ema = recent[0]
        for p in recent[1:]:
            ema = alpha * p + (1 - alpha) * ema
        return float(ema)

    def get_trend(self, ingredient: str) -> float:
        """Price trend ratio: recent short-window avg divided by long-window avg.

        Returns > 1.0 if prices are rising, < 1.0 if falling, 1.0 if stable/no data.
        """
        prices = self._prices.get(ingredient, [])
        if len(prices) < 6:
            return 1.0
        long_avg = sum(prices) / len(prices)
        short_avg = sum(prices[-4:]) / min(4, len(prices))
        if long_avg <= 0:
            return 1.0
        return float(short_avg / long_avg)

    def n_observations(self, ingredient: str) -> int:
        """Number of price observations recorded for this ingredient."""
        return len(self._prices.get(ingredient, []))

    def get_signal(self, ingredient: str) -> dict[str, float]:
        """Return a combined signal dict for use in bidding decisions.

        Keys:
          ema_short   — short-window EMA (5 obs); 0 if no data
          ema_long    — long-window EMA (15 obs); 0 if no data
          trend       — price trend ratio (>1 rising, <1 falling)
          n           — number of observations
        """
        return {
            "ema_short": self.get_ema(ingredient, window=5),
            "ema_long": self.get_ema(ingredient, window=15),
            "trend": self.get_trend(ingredient),
            "n": float(self.n_observations(ingredient)),
        }

    # ── Persistence ────────────────────────────────────────────────────

    def save(self, path: str = _DEFAULT_PATH) -> bool:
        """Atomically save price history to a JSON file."""
        folder = os.path.dirname(os.path.abspath(path)) or "."
        os.makedirs(folder, exist_ok=True)
        try:
            fd, tmp_path = tempfile.mkstemp(
                prefix=".price_db-", suffix=".json", dir=folder, text=True
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(self._prices, f, ensure_ascii=False)
                os.replace(tmp_path, path)
            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
        except OSError as e:
            log.warning("PriceDatabase.save failed: %s", e)
            return False
        log.info(
            "PriceDatabase: saved %d ingredients (%d total obs) to %s",
            len(self._prices),
            sum(len(v) for v in self._prices.values()),
            path,
        )
        return True

    def load(self, path: str = _DEFAULT_PATH) -> bool:
        """Load price history from a JSON file (merges into existing data)."""
        if not os.path.exists(path):
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            log.warning("PriceDatabase.load failed: %s", e)
            return False
        if not isinstance(data, dict):
            return False
        loaded = 0
        for ing, prices in data.items():
            if not isinstance(prices, list):
                continue
            bucket: list[float] = []
            for p in prices:
                try:
                    v = float(p)
                    if v > 0:
                        bucket.append(v)
                except (TypeError, ValueError):
                    continue
            if bucket:
                self._prices[str(ing)] = bucket[-_HISTORY_CAP:]
                loaded += 1
        log.info("PriceDatabase: loaded %d ingredients from %s", loaded, path)
        return loaded > 0
