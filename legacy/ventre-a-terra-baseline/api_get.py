"""HTTP GET endpoints for reading game state from the server."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import aiohttp

log = logging.getLogger(__name__)

_RETRYABLE = frozenset({429, 500, 502, 503, 504})
_MAX_RETRIES = 2
_RETRY_DELAY = 1.0


class GameGET:
    """Read-only HTTP calls to the Hackapizza server."""

    def __init__(self, session: aiohttp.ClientSession, base_url: str, api_key: str, team_id: str):
        self.session = session
        self.base_url = base_url
        self.api_key = api_key
        self.team_id = team_id

    @property
    def _headers(self) -> dict[str, str]:
        return {"x-api-key": self.api_key}

    async def _get(self, path: str, params: dict | None = None) -> Any:
        url = f"{self.base_url}{path}"
        for attempt in range(_MAX_RETRIES + 1):
            async with self.session.get(url, headers=self._headers, params=params) as resp:
                if resp.status in _RETRYABLE and attempt < _MAX_RETRIES:
                    log.warning("GET %s → %d, retry %d/%d", path, resp.status, attempt + 1, _MAX_RETRIES)
                    await asyncio.sleep(_RETRY_DELAY * (attempt + 1))
                    continue
                resp.raise_for_status()
                return await resp.json()

    async def restaurant(self) -> dict:
        """GET /restaurant/:id — our restaurant details."""
        return await self._get(f"/restaurant/{self.team_id}")

    async def restaurant_menu(self) -> list[dict]:
        """GET /restaurant/:id/menu — our current menu."""
        return await self._get(f"/restaurant/{self.team_id}/menu")

    async def restaurants(self) -> list[dict]:
        """GET /restaurants — all restaurants overview."""
        return await self._get("/restaurants")

    async def recipes(self) -> list[dict]:
        """GET /recipes — available recipes."""
        return await self._get("/recipes")

    async def market_entries(self) -> list[dict]:
        """GET /market/entries — active market entries."""
        return await self._get("/market/entries")

    async def meals(self, turn_id: int) -> list[dict]:
        """GET /meals — client requests for a turn."""
        return await self._get("/meals", {"turn_id": turn_id, "restaurant_id": self.team_id})

    async def bid_history(self, turn_id: int) -> list[dict]:
        """GET /bid_history — all bids from a turn."""
        return await self._get("/bid_history", {"turn_id": turn_id})
