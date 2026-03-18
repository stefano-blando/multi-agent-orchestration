"""MCP (JSON-RPC) action calls to the Hackapizza server."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import aiohttp

log = logging.getLogger(__name__)

_RETRYABLE = frozenset({429, 500, 502, 503, 504})
_MAX_RETRIES = 2
_RETRY_DELAY = 1.0

_rpc_id = 0


class GameMCP:
    """MCP tool calls via JSON-RPC POST /mcp."""

    def __init__(self, session: aiohttp.ClientSession, base_url: str, api_key: str):
        self.session = session
        self.base_url = base_url
        self.api_key = api_key

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }

    async def call(self, tool_name: str, arguments: dict) -> Any:
        """Raw MCP call. Returns the result or full response on error."""
        global _rpc_id
        _rpc_id += 1
        payload = {
            "jsonrpc": "2.0",
            "id": _rpc_id,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        }
        log.info("MCP %s → %s", tool_name, json.dumps(arguments, ensure_ascii=False))

        url = f"{self.base_url}/mcp"
        for attempt in range(_MAX_RETRIES + 1):
            async with self.session.post(url, headers=self._headers, json=payload) as resp:
                if resp.status in _RETRYABLE and attempt < _MAX_RETRIES:
                    log.warning("MCP %s → %d, retry %d/%d", tool_name, resp.status, attempt + 1, _MAX_RETRIES)
                    await asyncio.sleep(_RETRY_DELAY * (attempt + 1))
                    continue
                resp.raise_for_status()
                content_type = resp.headers.get("Content-Type", "")

                if "text/event-stream" in content_type:
                    data = await self._parse_sse(resp)
                else:
                    data = await resp.json()
                break

        if "error" in data:
            log.error("MCP error %s: %s", tool_name, data["error"])
        else:
            log.info("MCP %s OK", tool_name)
        return data.get("result", data)

    async def _parse_sse(self, resp: aiohttp.ClientResponse) -> dict:
        """Parse SSE response, return first JSON data line."""
        text = await resp.text()
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("data:"):
                payload = line[5:].strip()
                try:
                    return json.loads(payload)
                except json.JSONDecodeError:
                    pass
        return {}

    # ── Typed wrappers ────────────────────────────────────────────────

    async def save_menu(self, items: list[dict]) -> Any:
        """items: [{"name": "dish", "price": 50}, ...] — price is int, max 1000."""
        return await self.call("save_menu", {"items": items})

    async def closed_bid(self, bids: list[dict]) -> Any:
        """bids: [{"ingredient": "X", "bid": 5, "quantity": 3}, ...] — bid is price per unit (int)."""
        return await self.call("closed_bid", {"bids": bids})

    async def prepare_dish(self, dish_name: str) -> Any:
        return await self.call("prepare_dish", {"dish_name": dish_name})

    async def serve_dish(self, dish_name: str, client_id: str) -> Any:
        return await self.call("serve_dish", {"dish_name": dish_name, "client_id": client_id})

    async def update_restaurant_is_open(self, is_open: bool) -> Any:
        return await self.call("update_restaurant_is_open", {"is_open": is_open})

    async def create_market_entry(self, side: str, ingredient_name: str, quantity: int, price: int) -> Any:
        """side: 'BUY' or 'SELL'. price is TOTAL for all units (int)."""
        return await self.call("create_market_entry", {
            "side": side, "ingredient_name": ingredient_name,
            "quantity": quantity, "price": price,
        })

    async def execute_transaction(self, market_entry_id: int) -> Any:
        return await self.call("execute_transaction", {"market_entry_id": market_entry_id})

    async def delete_market_entry(self, market_entry_id: int) -> Any:
        return await self.call("delete_market_entry", {"market_entry_id": market_entry_id})

    async def send_message(self, recipient_id: int, text: str) -> Any:
        return await self.call("send_message", {"recipient_id": recipient_id, "text": text})
