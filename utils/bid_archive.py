"""Persistent bid archive — stores all past bid records for historical analysis.

Keeps up to 50 turns of bid history and provides analytical methods
for price estimation, competition analysis, and trend detection.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from statistics import median

log = logging.getLogger(__name__)

_MAX_TURNS = 50
_DEFAULT_PATH = "bid_archive.json"


class BidArchive:
    """Structured archive of all past bid records with analytical methods."""

    def __init__(self) -> None:
        # turn_id -> list of record dicts
        self._turns: dict[int, list[dict]] = {}

    # ── Data Ingestion ─────────────────────────────────────────────────

    def record_turn(
        self,
        turn_id: int,
        entries: list[dict],
        my_team_id: str,
    ) -> None:
        """Ingest one turn of bid_history into the archive.

        Each entry is stored as:
          {turn_id, ingredient, bid_unit, qty_requested, qty_won, restaurant_id, is_mine}
        """
        t = int(turn_id)
        if t <= 0:
            return
        if t in self._turns:
            return  # already archived

        my_team = str(my_team_id).strip()
        my_team_i = _safe_int(my_team_id, -1)

        records: list[dict] = []
        for raw in entries:
            if not isinstance(raw, dict):
                continue
            ingredient = str(
                _pick(raw, ("ingredient", "ingredientName", "item", "name"), "")
            ).strip()
            if not ingredient:
                continue

            qty_req = max(0, _safe_int(
                _pick(raw, ("quantity", "qty", "requestedQuantity", "requested_qty", "amount"), 0), 0
            ))
            bid_unit = _safe_int(
                _pick(raw, ("bid", "price", "unitPrice", "unit_price", "singlePrice", "single_price"), 0), 0
            )
            if bid_unit <= 0:
                total_price = _safe_int(_pick(raw, ("totalPrice", "total_price", "cost"), 0), 0)
                if qty_req > 0 and total_price > 0:
                    bid_unit = max(1, int(total_price // qty_req))

            qty_won = max(0, _safe_int(
                _pick(raw, ("wonQuantity", "won_qty", "boughtQuantity", "executedQuantity", "filledQuantity"), 0), 0
            ))

            rid_raw = _pick(raw, ("restaurantId", "restaurant_id", "teamId", "team_id", "restaurant"), "")
            rid = str(rid_raw).strip()
            rid_i = _safe_int(rid_raw, -1)
            is_mine = (rid and rid == my_team) or (rid_i >= 0 and my_team_i >= 0 and rid_i == my_team_i)

            records.append({
                "turn_id": t,
                "ingredient": ingredient,
                "bid_unit": bid_unit,
                "qty_requested": qty_req,
                "qty_won": qty_won,
                "restaurant_id": rid,
                "is_mine": is_mine,
            })

        if records:
            self._turns[t] = records
            self._trim()

    def _trim(self) -> None:
        if len(self._turns) > _MAX_TURNS:
            sorted_keys = sorted(self._turns.keys())
            excess = len(sorted_keys) - _MAX_TURNS
            for k in sorted_keys[:excess]:
                del self._turns[k]

    # ── Analysis Methods ───────────────────────────────────────────────

    def _recent_records(self, ingredient: str, last_n_turns: int = 10) -> list[dict]:
        """Get records for an ingredient from the most recent N turns."""
        sorted_turns = sorted(self._turns.keys(), reverse=True)[:last_n_turns]
        out: list[dict] = []
        for t in sorted_turns:
            for r in self._turns.get(t, []):
                if r["ingredient"] == ingredient:
                    out.append(r)
        return out

    def _all_ingredients(self) -> set[str]:
        ingredients: set[str] = set()
        for records in self._turns.values():
            for r in records:
                ingredients.add(r["ingredient"])
        return ingredients

    def avg_clearing_price(self, ingredient: str, last_n_turns: int = 10) -> float:
        """Average winning bid price for an ingredient over recent turns."""
        records = self._recent_records(ingredient, last_n_turns)
        winning = [r for r in records if r["qty_won"] > 0 and r["bid_unit"] > 0]
        if not winning:
            return 0.0
        total_spend = sum(r["bid_unit"] * r["qty_won"] for r in winning)
        total_qty = sum(r["qty_won"] for r in winning)
        return total_spend / total_qty if total_qty > 0 else 0.0

    def price_range(self, ingredient: str, last_n_turns: int = 10) -> tuple[float, float, float]:
        """(min, max, median) of winning bid prices for an ingredient."""
        records = self._recent_records(ingredient, last_n_turns)
        prices = [float(r["bid_unit"]) for r in records if r["qty_won"] > 0 and r["bid_unit"] > 0]
        if not prices:
            return (0.0, 0.0, 0.0)
        return (min(prices), max(prices), median(prices))

    def win_rate_at_price(self, ingredient: str, price: float) -> float:
        """Fraction of bids at or below `price` that won something."""
        records = self._recent_records(ingredient, 20)
        relevant = [r for r in records if r["bid_unit"] > 0 and r["bid_unit"] <= price * 1.1]
        if not relevant:
            return 0.0
        winners = sum(1 for r in relevant if r["qty_won"] > 0)
        return winners / len(relevant)

    def competition_level(self, ingredient: str, last_n_turns: int = 5) -> float:
        """Average number of distinct bidders per turn for this ingredient."""
        sorted_turns = sorted(self._turns.keys(), reverse=True)[:last_n_turns]
        if not sorted_turns:
            return 0.0
        counts: list[int] = []
        for t in sorted_turns:
            bidders = set()
            for r in self._turns.get(t, []):
                if r["ingredient"] == ingredient:
                    bidders.add(r["restaurant_id"])
            if bidders:
                counts.append(len(bidders))
        return sum(counts) / len(counts) if counts else 0.0

    def price_trend(self, ingredient: str) -> str:
        """'rising', 'falling', or 'stable' based on recent price movement."""
        sorted_turns = sorted(self._turns.keys())
        if len(sorted_turns) < 4:
            return "stable"

        # Compute per-turn average winning price
        turn_avgs: list[float] = []
        for t in sorted_turns[-10:]:
            winning = [r for r in self._turns[t]
                       if r["ingredient"] == ingredient and r["qty_won"] > 0 and r["bid_unit"] > 0]
            if winning:
                avg = sum(r["bid_unit"] * r["qty_won"] for r in winning) / sum(r["qty_won"] for r in winning)
                turn_avgs.append(avg)

        if len(turn_avgs) < 3:
            return "stable"

        # Simple linear slope via first/last halves
        mid = len(turn_avgs) // 2
        first_half = sum(turn_avgs[:mid]) / mid
        second_half = sum(turn_avgs[mid:]) / (len(turn_avgs) - mid)
        if first_half <= 0:
            return "stable"
        ratio = second_half / first_half
        if ratio > 1.10:
            return "rising"
        elif ratio < 0.90:
            return "falling"
        return "stable"

    def cheapest_ingredients(self, top_n: int = 10) -> list[tuple[str, float]]:
        """Return the top_n cheapest ingredients by average clearing price."""
        ingredients = self._all_ingredients()
        prices: list[tuple[str, float]] = []
        for ing in ingredients:
            avg = self.avg_clearing_price(ing)
            if avg > 0:
                prices.append((ing, avg))
        prices.sort(key=lambda x: x[1])
        return prices[:top_n]

    def most_contested(self, top_n: int = 10) -> list[tuple[str, float]]:
        """Return the top_n most contested ingredients by competition level."""
        ingredients = self._all_ingredients()
        levels: list[tuple[str, float]] = []
        for ing in ingredients:
            lvl = self.competition_level(ing)
            if lvl > 0:
                levels.append((ing, lvl))
        levels.sort(key=lambda x: -x[1])
        return levels[:top_n]

    def summary_for_ingredient(self, ingredient: str) -> dict:
        """Combined analytical summary for one ingredient."""
        avg = self.avg_clearing_price(ingredient)
        p_min, p_max, p_med = self.price_range(ingredient)
        trend = self.price_trend(ingredient)
        comp = self.competition_level(ingredient)
        wr = self.win_rate_at_price(ingredient, avg) if avg > 0 else 0.0
        records = self._recent_records(ingredient, 20)
        n_obs = len([r for r in records if r["qty_won"] > 0])
        return {
            "ingredient": ingredient,
            "avg_price": round(avg, 1),
            "min_price": round(p_min, 1),
            "max_price": round(p_max, 1),
            "median_price": round(p_med, 1),
            "trend": trend,
            "competition": round(comp, 1),
            "win_rate_at_avg": round(wr, 2),
            "n_observations": n_obs,
        }

    # ── Persistence ────────────────────────────────────────────────────

    def save(self, path: str = _DEFAULT_PATH) -> bool:
        """Atomically save archive to JSON file."""
        folder = os.path.dirname(os.path.abspath(path)) or "."
        os.makedirs(folder, exist_ok=True)
        # Convert int keys to strings for JSON
        data = {str(k): v for k, v in self._turns.items()}
        try:
            fd, tmp_path = tempfile.mkstemp(
                prefix=".bid_archive-", suffix=".json", dir=folder, text=True
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False)
                os.replace(tmp_path, path)
            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
        except OSError as e:
            log.warning("BidArchive.save failed: %s", e)
            return False
        log.info(
            "BidArchive: saved %d turns (%d total records) to %s",
            len(self._turns),
            sum(len(v) for v in self._turns.values()),
            path,
        )
        return True

    def load(self, path: str = _DEFAULT_PATH) -> bool:
        """Load archive from JSON file."""
        if not os.path.exists(path):
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            log.warning("BidArchive.load failed: %s", e)
            return False
        if not isinstance(data, dict):
            return False
        loaded = 0
        for key, records in data.items():
            if not isinstance(records, list):
                continue
            try:
                turn_id = int(key)
            except (TypeError, ValueError):
                continue
            valid_records: list[dict] = []
            for r in records:
                if not isinstance(r, dict):
                    continue
                if not r.get("ingredient"):
                    continue
                valid_records.append(r)
            if valid_records:
                self._turns[turn_id] = valid_records
                loaded += 1
        self._trim()
        log.info("BidArchive: loaded %d turns from %s", loaded, path)
        return loaded > 0

    def has_turn(self, turn_id: int) -> bool:
        return int(turn_id) in self._turns


# ── Helpers ────────────────────────────────────────────────────────────

def _pick(entry: dict, keys: tuple[str, ...], default=None):
    for k in keys:
        if k in entry:
            return entry.get(k)
    return default


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
