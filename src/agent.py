"""
Multi-agent system con datapizza-ai.

Architettura:
  OrchestratorAgent
    ├── RetrieverAgent  (cerca nei documenti con hybrid search)
    ├── ValidatorAgent  (controlla vincoli: dieta, normative)
    └── SynthesizerAgent (genera risposta finale)

Note API datapizza-ai:
  - Agent richiede: name, client, system_prompt (tutti obbligatori)
  - agent.run(task) → StepResult (.text, .tools_used, .usage)
  - orchestrator.can_call(sub_agent) → registra sub_agent come tool
  - agent.as_tool() → Tool (description presa dal system_prompt)
"""

import os
from dotenv import load_dotenv
from datapizza.agents import Agent
from datapizza.tools import tool, Tool

load_dotenv()


def build_client():
    """Costruisce il client LLM in base alle variabili d'ambiente disponibili."""
    if os.getenv("OPENAI_API_KEY"):
        from datapizza.clients.openai import OpenAIClient
        return OpenAIClient(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-mini",
        )
    # Fallback: MockClient per test senza API
    from datapizza.clients.mock_client import MockClient
    return MockClient()


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def semantic_search(query: str, top_k: int = 5) -> str:
    """
    Cerca semanticamente nei documenti indicizzati su Qdrant.
    Ritorna i chunk più rilevanti con source e testo.
    """
    from src.retrieval import hybrid_search
    results = hybrid_search(query, top_k=top_k)
    if not results:
        return "Nessun documento trovato per questa query."
    return "\n\n".join(
        f"[{r['source']}] (score: {r['score']:.3f})\n{r['text']}"
        for r in results
    )


@tool
def check_constraint(item: str, constraint_type: str) -> str:
    """
    Verifica se un item rispetta un vincolo (dieta, normativa, etc.).
    Restituisce 'compliant' o 'non_compliant' con spiegazione.

    Args:
        item: nome del piatto o ingrediente da verificare
        constraint_type: tipo di vincolo (es. 'gluten_free', 'vegan', 'regulation_X')
    """
    # TODO: implementare con i dati reali del challenge al kickoff
    return f"Verifica '{item}' per vincolo '{constraint_type}': da implementare con dati challenge."


# ---------------------------------------------------------------------------
# Agenti specializzati
# ---------------------------------------------------------------------------

def build_retriever_agent(client) -> Agent:
    return Agent(
        name="retriever",
        client=client,
        tools=[semantic_search],
        system_prompt=(
            "Sei un esperto di ricerca documentale. "
            "Data una query, usa semantic_search per trovare i documenti più rilevanti. "
            "Riporta sempre il nome del documento sorgente (source) per ogni risultato. "
            "Se i risultati non sono sufficienti, riformula la query e riprova."
        ),
    )


def build_validator_agent(client) -> Agent:
    return Agent(
        name="validator",
        client=client,
        tools=[check_constraint],
        system_prompt=(
            "Sei un esperto di vincoli dietetici e normative di sicurezza alimentare. "
            "Data una lista di candidati e i vincoli della query, "
            "verifica ogni candidato con check_constraint, "
            "scarta quelli non conformi e spiega il motivo. "
            "Ritorna solo i candidati che rispettano tutti i vincoli."
        ),
    )


def build_synthesizer_agent(client) -> Agent:
    return Agent(
        name="synthesizer",
        client=client,
        tools=[],
        system_prompt=(
            "Sei un sintetizzatore di risposte precise. "
            "Ricevi documenti rilevanti già validati e produci una risposta strutturata "
            "alla query originale. Cita sempre le fonti (nome documento). "
            "Sii conciso e diretto."
        ),
    )


def build_orchestrator(client) -> Agent:
    retriever = build_retriever_agent(client)
    validator = build_validator_agent(client)
    synthesizer = build_synthesizer_agent(client)

    orchestrator = Agent(
        name="orchestrator",
        client=client,
        tools=[],
        system_prompt=(
            "Sei l'orchestratore di un sistema multi-agente per rispondere a query "
            "su documenti. Segui sempre questi passi:\n"
            "1. Usa 'retriever' per trovare documenti rilevanti\n"
            "2. Usa 'validator' per verificare che i risultati rispettino i vincoli della query\n"
            "3. Usa 'synthesizer' per produrre la risposta finale\n"
            "Sii efficiente: non ripetere passi inutili."
        ),
    )
    orchestrator.can_call([retriever, validator, synthesizer])
    return orchestrator


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_pipeline(query: str) -> str:
    """
    Esegue la pipeline multi-agente completa su una query.
    Costruisce gli agenti fresh ad ogni chiamata (stateless by default).
    """
    client = build_client()
    orchestrator = build_orchestrator(client)
    result = orchestrator.run(query)
    return result.text if result else "Nessuna risposta generata."


if __name__ == "__main__":
    query = input("Query: ")
    print(run_pipeline(query))
