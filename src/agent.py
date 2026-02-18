"""
Multi-agent system con datapizza-ai.

Architettura:
  OrchestratorAgent
    ├── RetrieverAgent  (cerca nei documenti con hybrid search)
    ├── ValidatorAgent  (controlla vincoli: dieta, normative)
    └── SynthesizerAgent (genera risposta finale)
"""

import os
from dotenv import load_dotenv
from datapizza.agents import Agent
from datapizza.tools import tool
from datapizza.clients.openai import OpenAIClient

load_dotenv()

# ---------------------------------------------------------------------------
# Client LLM
# ---------------------------------------------------------------------------

client = OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"))


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def semantic_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Cerca semanticamente nei documenti indicizzati su Qdrant.
    Ritorna i chunk più rilevanti con source e testo.
    """
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer
    from src.utils import get_device

    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection = os.getenv("QDRANT_COLLECTION", "hackapizza")

    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)
    model = model.to(get_device())
    vector = model.encode(query, normalize_embeddings=True).tolist()

    client_q = QdrantClient(url=qdrant_url)
    results = client_q.search(
        collection_name=collection,
        query_vector=vector,
        limit=top_k,
    )
    return [{"text": r.payload["text"], "source": r.payload["source"], "score": r.score} for r in results]


@tool
def bm25_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Ricerca BM25 (keyword-based) sui chunk indicizzati.
    Utile per nomi propri, codici, termini tecnici con typo.
    """
    # TODO: popolare questo corpus al momento dell'ingestion
    # Per ora è un placeholder — da collegare al corpus reale
    raise NotImplementedError("Collegare al corpus al momento dell'ingestion.")


@tool
def check_constraint(item: str, constraint_type: str) -> dict:
    """
    Verifica se un item rispetta un vincolo (es. dieta, normativa).

    Args:
        item: nome del piatto o ingrediente
        constraint_type: tipo di vincolo (es. "gluten_free", "vegan", "regulation_X")

    Returns:
        {"compliant": bool, "reason": str}
    """
    # Placeholder — da implementare con i dati reali del challenge
    return {"compliant": True, "reason": "Da implementare con dati challenge"}


# ---------------------------------------------------------------------------
# Agenti specializzati
# ---------------------------------------------------------------------------

retriever_agent = Agent(
    name="retriever",
    client=client,
    tools=[semantic_search, bm25_search],
    system_prompt=(
        "Sei un esperto di ricerca documentale. "
        "Data una query, usa semantic_search e bm25_search per trovare "
        "i documenti più rilevanti. Combina i risultati ed elimina duplicati. "
        "Ritorna sempre le source dei documenti trovati."
    ),
)

validator_agent = Agent(
    name="validator",
    client=client,
    tools=[check_constraint],
    system_prompt=(
        "Sei un esperto di vincoli e normative. "
        "Data una lista di candidati, verifica che rispettino tutti i vincoli "
        "specificati nella query. Scarta quelli non conformi e spiega perché."
    ),
)

synthesizer_agent = Agent(
    name="synthesizer",
    client=client,
    tools=[],
    system_prompt=(
        "Sei un sintetizzatore di risposte. "
        "Ricevi documenti rilevanti già validati e devi produrre una risposta "
        "precisa e strutturata alla query originale. "
        "Sii conciso, preciso e cita sempre le fonti."
    ),
)

orchestrator_agent = Agent(
    name="orchestrator",
    client=client,
    tools=[],
    system_prompt=(
        "Sei l'orchestratore di un sistema multi-agente. "
        "Il tuo compito è coordinare retriever, validator e synthesizer "
        "per rispondere alla query dell'utente nel modo più preciso ed efficiente. "
        "Prima recupera i documenti rilevanti, poi valida i vincoli, poi sintetizza."
    ),
)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_pipeline(query: str) -> str:
    """
    Esegue la pipeline multi-agente completa su una query.

    Returns:
        Risposta finale come stringa.
    """
    # Step 1: Retrieval
    retrieval_result = retriever_agent.run(
        f"Query: {query}\nRecupera i documenti più rilevanti."
    )

    # Step 2: Validation
    validation_result = validator_agent.run(
        f"Query originale: {query}\n"
        f"Documenti trovati: {retrieval_result.text}\n"
        f"Verifica i vincoli e filtra i risultati non conformi."
    )

    # Step 3: Synthesis
    final_result = synthesizer_agent.run(
        f"Query originale: {query}\n"
        f"Documenti validati: {validation_result.text}\n"
        f"Produci la risposta finale."
    )

    return final_result.text


if __name__ == "__main__":
    query = input("Query: ")
    print(run_pipeline(query))
