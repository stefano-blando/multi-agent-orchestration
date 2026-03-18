"""Test the agent by simulating game phases without SSE."""

import asyncio
import logging
import os
import sys

import aiohttp
from dotenv import load_dotenv

# Allow imports from app/
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "app"))

from agent import init_agent, create_agent_for_phase, build_prompt_for_phase
from api_get import GameGET
from api_mcp import GameMCP
from game_state import GameState

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("test")

TEAM_API_KEY = os.environ["TEAM_API_KEY"]
TEAM_ID = os.environ["TEAM_ID"]
REGOLO_API_KEY = os.environ["REGOLO"]
BASE_URL = "https://hackapizza.datapizza.tech"


def run_phase_sync(state, phase):
    """Create phase-specific agent, build prompt, run."""
    state.on_phase_changed(phase)
    agent = create_agent_for_phase(phase)
    prompt = build_prompt_for_phase(phase, state)

    print(f"\n{'='*60}")
    print(f"  PHASE: {phase}")
    print(f"{'='*60}")
    print(f"Prompt ({len(prompt)} chars):\n{prompt[:800]}")
    print(f"\n--- Agent thinking ---\n")

    result = agent.run(prompt)

    print(f"\n--- Agent result ---")
    if result:
        print(f"Text: {result.text[:1000] if result.text else '(none)'}")
        print(f"Tools used: {[t.name for t in result.tools_used] if result.tools_used else '(none)'}")
    else:
        print("(None returned)")
    return result


async def main():
    phase = sys.argv[1] if len(sys.argv) > 1 else "speaking"

    timeout = aiohttp.ClientTimeout(total=None, sock_connect=15, sock_read=None)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        get_api = GameGET(session, BASE_URL, TEAM_API_KEY, TEAM_ID)
        mcp_api = GameMCP(session, BASE_URL, TEAM_API_KEY)

        # Fetch real state from server
        state = GameState()
        try:
            rest_data = await get_api.restaurant()
            state.update_restaurant(rest_data)
            log.info("Restaurant: balance=%s inventory=%s", state.balance, state.inventory)
        except Exception as e:
            log.warning("Could not fetch restaurant: %s", e)

        try:
            recipes = await get_api.recipes()
            state.update_recipes(recipes)
            log.info("Loaded %d recipes", len(recipes))
        except Exception as e:
            log.warning("Could not fetch recipes: %s", e)

        state.turn_id = 1

        # Init agent system (tools + LLM client)
        init_agent(get_api, mcp_api, state, REGOLO_API_KEY)

        if phase == "serving":
            state.on_client_spawned({
                "clientId": "client_42",
                "clientName": "Zorg il Magnifico",
                "orderText": "I'd like something cosmic and prestigious",
            })
            state.on_client_spawned({
                "clientId": "client_43",
                "clientName": "Luna Stellare",
                "orderText": "Give me your best dish please",
            })

        loop = asyncio.get_event_loop()
        if phase == "all":
            for p in ["speaking", "closed_bid", "waiting", "serving"]:
                if p == "serving":
                    state.on_client_spawned({
                        "clientId": "client_42",
                        "clientName": "Zorg",
                        "orderText": "Best dish please",
                    })
                await loop.run_in_executor(None, lambda p=p: run_phase_sync(state, p))
        else:
            await loop.run_in_executor(None, lambda: run_phase_sync(state, phase))


if __name__ == "__main__":
    print("Usage: python scripts/test_agent.py [speaking|closed_bid|waiting|serving|all]")
    print()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped")
