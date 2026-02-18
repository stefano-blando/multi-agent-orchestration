"""
Hello World con datapizza-ai.
Testa: client, tool use, agente singolo, multi-agent (can_call).

Modalità:
- Senza OPENAI_API_KEY: test strutturale con MockClient (no costi)
- Con OPENAI_API_KEY: test reale con gpt-4o-mini

Uso:
    python hello_datapizza.py
    OPENAI_API_KEY=sk-... python hello_datapizza.py
"""

import os
from dotenv import load_dotenv
from datapizza.agents import Agent
from datapizza.tools import tool
from datapizza.clients.mock_client import MockClient

load_dotenv()

# ---------------------------------------------------------------------------
# Scegli il client in base alla disponibilità della chiave
# ---------------------------------------------------------------------------

api_key = os.getenv("OPENAI_API_KEY")

if api_key:
    from datapizza.clients.openai import OpenAIClient
    client = OpenAIClient(api_key=api_key, model="gpt-4o-mini")
    print("Modalità: OpenAI (reale)")
else:
    client = MockClient()
    print("Modalità: MockClient (strutturale, no API key)")

print()

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

@tool
def get_menu(restaurant_name: str) -> str:
    """Ritorna il menù di un ristorante dato il suo nome."""
    menus = {
        "Ristorante Galattico": "Pizza cosmica, Pasta nebulosa, Gelato stellare",
        "Trattoria Aliena": "Risotto di asteroidi, Bistecca di meteorite",
    }
    return menus.get(restaurant_name, "Ristorante non trovato.")

@tool
def check_allergens(dish: str) -> str:
    """Verifica gli allergeni di un piatto."""
    allergens = {
        "Pizza cosmica": "glutine, latticini",
        "Pasta nebulosa": "glutine, uova",
        "Gelato stellare": "latticini",
        "Risotto di asteroidi": "nessun allergene noto",
    }
    return allergens.get(dish, "Piatto non nel database allergeni.")

# ---------------------------------------------------------------------------
# Test 1: Tool — verifica che il decorator funzioni
# ---------------------------------------------------------------------------

print("=" * 60)
print("TEST 1: Tool decorator")
print("=" * 60)

print(f"Nome tool:    {get_menu.name}")
print(f"Descrizione:  {get_menu.description}")
print(f"Parametri:    {get_menu.properties}")
print(f"Call diretta: {get_menu('Ristorante Galattico')}")

# ---------------------------------------------------------------------------
# Test 2: Agente singolo
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("TEST 2: Agente singolo")
print("=" * 60)

menu_agent = Agent(
    name="menu_agent",
    client=client,
    tools=[get_menu],
    system_prompt="Sei un assistente per ristoranti. Usa i tool disponibili per rispondere.",
)

if api_key:
    result = menu_agent.run("Cosa posso mangiare al Ristorante Galattico?")
    print(f"Risposta: {result.text}")
    print(f"Token:    {result.usage}")
else:
    print("Agente istanziato correttamente.")
    print(f"Tools registrati: {[t.name for t in menu_agent._tools]}")

# ---------------------------------------------------------------------------
# Test 3: Multi-agent con can_call
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("TEST 3: Multi-agent (can_call)")
print("=" * 60)

validator_agent = Agent(
    name="validator",
    client=client,
    tools=[check_allergens],
    system_prompt="Sei un esperto di allergeni. Verifica sempre gli allergeni prima di raccomandare un piatto.",
)

orchestrator = Agent(
    name="orchestrator",
    client=client,
    tools=[get_menu],
    system_prompt=(
        "Sei un assistente culinario. "
        "Prima recupera il menù, poi usa il validator per controllare gli allergeni."
    ),
)
orchestrator.can_call(validator_agent)

if api_key:
    result = orchestrator.run(
        "Sono intollerante al glutine. Cosa posso mangiare al Ristorante Galattico?"
    )
    print(f"Risposta: {result.text}")
    print(f"Token:    {result.usage}")
else:
    all_tools = [t.name for t in orchestrator._tools]
    print("Orchestrator istanziato correttamente.")
    print(f"Tools disponibili: {all_tools}")
    # validator è accessibile come tool tramite can_call
    assert "validator" in all_tools, "validator non trovato tra i tools!"
    print("can_call funziona: validator registrato come tool.")

# ---------------------------------------------------------------------------
# Test 4: Agent-as-tool (alternativa a can_call)
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("TEST 4: Agent.as_tool()")
print("=" * 60)

validator_tool = validator_agent.as_tool()
print(f"Agent convertito in tool: {validator_tool.name}")
print(f"Descrizione: {validator_tool.description}")

print("\n✅ Tutti i test strutturali passati.")
if not api_key:
    print("   (Per test con LLM reale: OPENAI_API_KEY=sk-... python hello_datapizza.py)")
