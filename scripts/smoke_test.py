"""Smoke test: verify connectivity to server + MCP tools/list."""

import asyncio
import json
import os

import aiohttp
from dotenv import load_dotenv

load_dotenv()

TEAM_API_KEY = os.environ["TEAM_API_KEY"]
TEAM_ID = os.environ["TEAM_ID"]
BASE_URL = "https://hackapizza.datapizza.tech"
HEADERS = {"x-api-key": TEAM_API_KEY}
MCP_HEADERS = {**HEADERS, "Content-Type": "application/json", "Accept": "application/json, text/event-stream"}


async def test_get(session, path, label):
    url = f"{BASE_URL}{path}"
    try:
        async with session.get(url, headers=HEADERS) as resp:
            body = await resp.text()
            status = resp.status
            short = body[:200] if body else "(empty)"
            print(f"  {'✓' if status < 400 else '✗'} {label}: {status} — {short}")
    except Exception as e:
        print(f"  ✗ {label}: {e}")


async def test_mcp_list(session):
    url = f"{BASE_URL}/mcp"
    payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
    try:
        async with session.post(url, headers=MCP_HEADERS, json=payload) as resp:
            data = await resp.json()
            if "result" in data:
                tools = data["result"].get("tools", [])
                names = [t["name"] for t in tools]
                print(f"  ✓ MCP tools/list: {len(tools)} tools — {names}")
            else:
                print(f"  ✗ MCP tools/list: {data}")
    except Exception as e:
        print(f"  ✗ MCP tools/list: {e}")


async def test_sse(session):
    url = f"{BASE_URL}/events/{TEAM_ID}"
    headers = {"Accept": "text/event-stream", "x-api-key": TEAM_API_KEY}
    try:
        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=5)) as resp:
            print(f"  ✓ SSE /events/{TEAM_ID}: {resp.status}")
            async for line in resp.content:
                decoded = line.decode("utf-8", errors="ignore").strip()
                if decoded:
                    print(f"    → {decoded[:100]}")
                    break
    except asyncio.TimeoutError:
        print(f"  ✓ SSE /events/{TEAM_ID}: connected (timed out waiting for events, which is normal)")
    except Exception as e:
        print(f"  ✗ SSE: {e}")


async def main():
    print(f"Smoke test — team={TEAM_ID} server={BASE_URL}\n")

    async with aiohttp.ClientSession() as session:
        print("GET endpoints:")
        await test_get(session, f"/restaurant/{TEAM_ID}", f"GET /restaurant/{TEAM_ID}")
        await test_get(session, "/restaurants", "GET /restaurants")
        await test_get(session, "/recipes", "GET /recipes")
        await test_get(session, "/market/entries", "GET /market/entries")

        print("\nMCP:")
        await test_mcp_list(session)

        print("\nSSE:")
        await test_sse(session)

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
