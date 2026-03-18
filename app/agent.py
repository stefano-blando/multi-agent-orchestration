"""Agent factory — creates phase-specific agents with memory and sub-agents."""

from __future__ import annotations

import json
import logging
import os

from datapizza.agents import Agent
from datapizza.clients.openai_like import OpenAILikeClient
from datapizza.memory import Memory

from api_get import GameGET
from api_mcp import GameMCP
from game_state import GameState
from phases import speaking, bidding, waiting, serving
from tools import (
    init_tools,
    get_bid_archive_summary,
    get_bid_archive_cheapest,
    get_bid_archive_price_check,
    get_competitors,
    get_market,
)

log = logging.getLogger(__name__)

PHASE_CONFIG = {
    "speaking": speaking,
    "closed_bid": bidding,
    "waiting": waiting,
    "serving": serving,
}

_client: OpenAILikeClient | None = None

# Phases that benefit from persistent memory across turns.
# Serving memory can bloat quickly when get_meals payloads are large.
_MEMORY_PHASES = {"closed_bid"}
_MAX_MEMORY_TURNS = 6

MARKET_ANALYST_PROMPT = """\
You are a market analyst assistant for restaurant "ventre a terra" in Hackapizza 2.0.
Your job is to answer questions about ingredient prices, market trends, and competition.
Be concise: respond with numbers, short bullet points, and actionable insights.
Do NOT perform any actions — only analyze and report.
"""


def init_agent(
    get: GameGET,
    mcp: GameMCP,
    state: GameState,
    regolo_api_key: str,
) -> None:
    """Initialize tools and LLM client (call once at startup)."""
    global _client
    init_tools(get, mcp, state)
    _client = OpenAILikeClient(
        api_key=regolo_api_key,
        model="gpt-oss-120b",
        base_url="https://api.regolo.ai/v1",
    )


# -- Memory helpers ----------------------------------------------------------

def _memory_path(phase: str) -> str:
    return f"agent_memory_{phase}.json"


def load_agent_memory(phase: str) -> Memory | None:
    """Load memory from disk for a phase. Returns None if no file exists."""
    path = _memory_path(phase)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            data = f.read()
        mem = Memory()
        mem.json_loads(data)
        # Trim to max turns to avoid context bloat.
        turns = mem.to_dict()
        if len(turns) > _MAX_MEMORY_TURNS:
            trimmed = Memory()
            trimmed.json_loads(json.dumps(turns[-_MAX_MEMORY_TURNS:]))
            return trimmed
        return mem
    except Exception as e:
        log.warning("Failed to load memory for %s: %s", phase, e)
        return None


def save_agent_memory(phase: str, agent: Agent) -> bool:
    """Save agent's memory to disk (only for memory-enabled phases)."""
    if phase not in _MEMORY_PHASES:
        return False
    try:
        mem = agent._memory
        # Trim before saving.
        turns = mem.to_dict()
        if len(turns) > _MAX_MEMORY_TURNS:
            trimmed = Memory()
            trimmed.json_loads(json.dumps(turns[-_MAX_MEMORY_TURNS:]))
            mem = trimmed
        path = _memory_path(phase)
        with open(path, "w") as f:
            f.write(mem.json_dumps())
        log.info("Saved memory for %s (%d turns)", phase, len(mem.to_dict()))
        return True
    except Exception as e:
        log.warning("Failed to save memory for %s: %s", phase, e)
        return False


# -- Sub-agent: MarketAnalyst -----------------------------------------------

def _create_market_analyst() -> Agent:
    """Create a MarketAnalyst sub-agent for price/market analysis."""
    analyst = Agent(
        name="market_analyst",
        client=_client,
        system_prompt=MARKET_ANALYST_PROMPT,
        tools=[
            get_bid_archive_summary,
            get_bid_archive_cheapest,
            get_bid_archive_price_check,
            get_competitors,
            get_market,
        ],
        max_steps=5,
        terminate_on_text=True,
    )
    analyst.__doc__ = "Analyze market: historical bid prices, trends, competition, market entries. Call with a question."
    return analyst


# -- Agent factory -----------------------------------------------------------

def create_agent_for_phase(phase: str) -> Agent | None:
    """Create an agent configured for the given phase."""
    config = PHASE_CONFIG.get(phase)
    if not config:
        return None

    max_steps = 16 if phase == "serving" else 10

    # Memory: load for bidding and serving phases.
    memory = None
    stateless = True
    if phase in _MEMORY_PHASES:
        stateless = False
        memory = load_agent_memory(phase)

    # Sub-agents: market_analyst for bidding and waiting.
    can_call = None
    if phase in ("closed_bid", "waiting"):
        can_call = [_create_market_analyst()]

    return Agent(
        name=f"agent_{phase}",
        client=_client,
        system_prompt=config.SYSTEM_PROMPT,
        tools=config.TOOLS,
        max_steps=max_steps,
        terminate_on_text=False,
        stateless=stateless,
        memory=memory,
        can_call=can_call,
    )


def build_prompt_for_phase(phase: str, state: GameState) -> str | None:
    """Build the phase-specific prompt."""
    config = PHASE_CONFIG.get(phase)
    if not config:
        return None
    return config.build_prompt(state)
