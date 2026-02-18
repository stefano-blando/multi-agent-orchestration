"""
Agentic AI system con datapizza-ai — Hackapizza 2.0.

Architettura: singolo MainAgent con ReAct loop e planning.
L'agente decide autonomamente quali tool usare, in che ordine,
e quante iterazioni fare prima di rispondere.

Caratteristiche agentiche:
  - planning_interval=1: pianifica esplicitamente prima di agire
  - terminate_on_text=False: DEVE chiamare submit_answer per rispondere
  - submit_answer(end=True): forza output strutturato (lista) per scoring Jaccard
  - max_steps=15: evita loop infiniti

Tool disponibili:
  - hybrid_search: BM25 + semantic search con RRF fusion
  - get_document: recupera tutti i chunk di un documento specifico
  - check_constraint: verifica un vincolo su un item
  - submit_answer: risposta finale strutturata (end=True)
"""

import json
import os
from dotenv import load_dotenv
from datapizza.agents import Agent
from datapizza.tools import tool

load_dotenv()


def build_client():
    if os.getenv("OPENAI_API_KEY"):
        from datapizza.clients.openai import OpenAIClient
        return OpenAIClient(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-mini",
        )
    from datapizza.clients.mock_client import MockClient
    return MockClient()


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

@tool
def hybrid_search(query: str, top_k: int = 5) -> str:
    """
    Cerca nei documenti indicizzati usando ricerca ibrida (BM25 + semantica).
    Usa questo tool ogni volta che hai bisogno di trovare informazioni nei documenti.
    Puoi chiamarlo più volte con query diverse per esplorare aspetti differenti.

    Args:
        query: la query di ricerca in linguaggio naturale
        top_k: numero di risultati da ritornare (default 5, max 10)

    Returns:
        Lista di chunk rilevanti con fonte e score di rilevanza.
    """
    from src.retrieval import hybrid_search as _hybrid_search
    results = _hybrid_search(query, top_k=min(top_k, 10))
    if not results:
        return "Nessun documento trovato per questa query. Prova a riformularla."
    return "\n\n---\n\n".join(
        f"[Fonte: {r['source']} | Score: {r['score']:.3f}]\n{r['text']}"
        for r in results
    )


@tool
def get_document(source: str) -> str:
    """
    Recupera il contenuto completo di un documento specifico per nome file.
    Usa questo tool quando hai trovato un documento rilevante tramite hybrid_search
    e vuoi approfondire il suo contenuto completo.

    Args:
        source: nome del file documento (es. 'menu_galattico.pdf')

    Returns:
        Tutto il testo del documento.
    """
    from src.retrieval import corpus
    if corpus.is_empty():
        return "Corpus non ancora caricato. Esegui prima l'ingestion."
    chunks = [
        chunk for chunk, src in zip(corpus.chunks, corpus.sources)
        if src == source
    ]
    if not chunks:
        return f"Documento '{source}' non trovato nel corpus."
    return f"[Documento: {source}]\n\n" + "\n\n".join(chunks)


@tool
def check_constraint(item: str, constraint: str) -> str:
    """
    Verifica se un item (piatto, ingrediente, prodotto) rispetta un vincolo specifico.
    Usa questo tool per validare ogni candidato rispetto ai requisiti della query
    prima di includerlo nella risposta finale.

    Args:
        item: nome del piatto o ingrediente da verificare
        constraint: descrizione del vincolo (es. 'senza glutine', 'vegano', 'normativa X')

    Returns:
        'CONFORME' o 'NON CONFORME' con motivazione.
    """
    # TODO al kickoff: implementare con i dati reali del challenge
    # Per ora: ricerca nel corpus se ci sono info sul vincolo per questo item
    from src.retrieval import hybrid_search as _hybrid_search
    results = _hybrid_search(f"{item} {constraint}", top_k=3)
    if not results:
        return f"Nessuna informazione trovata su '{item}' e vincolo '{constraint}'."
    context = "\n".join(r["text"] for r in results)
    return f"Informazioni trovate per '{item}' / '{constraint}':\n{context}"


@tool(end=True)
def submit_answer(items: list) -> str:
    """
    Sottomette la risposta finale come lista di item.
    CHIAMARE SOLO quando sei completamente sicuro della risposta.
    NON chiamare prima di aver verificato tutti i vincoli della query.
    La risposta verrà valutata con Jaccard Similarity — includi SOLO item certi.

    Args:
        items: lista di nomi (piatti, prodotti, etc.) che rispondono alla query

    Returns:
        JSON della risposta finale.
    """
    return json.dumps({"answer": items}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """Sei un agente AI specializzato nell'analisi di documenti per rispondere
a query complesse. Operi in un loop ReAct: Ragiona → Agisci → Osserva → Ragiona.

REGOLE FONDAMENTALI:
1. Prima di agire, pianifica: "Cosa devo trovare per rispondere a questa query?"
2. Usa hybrid_search più volte con angolazioni diverse se necessario
3. Se trovi un documento rilevante, approfondiscilo con get_document
4. Per ogni vincolo nella query (dieta, normative, restrizioni), verifica con check_constraint
5. Includi nella risposta SOLO item di cui sei certo
6. Termina SEMPRE chiamando submit_answer con la lista finale — mai rispondere in prosa

STRATEGIA DI RICERCA:
- Inizia con ricerche generali, poi raffina
- Cerca anche le eccezioni e le esclusioni
- Se la query menziona vincoli multipli, verifica ognuno separatamente
- In caso di dubbio su un item, escludilo dalla risposta
"""


def build_agent() -> Agent:
    client = build_client()
    return Agent(
        name="hackapizza_agent",
        client=client,
        tools=[hybrid_search, get_document, check_constraint, submit_answer],
        system_prompt=SYSTEM_PROMPT,
        planning_interval=1,      # pianifica prima di ogni ciclo
        max_steps=15,             # evita loop infiniti
        terminate_on_text=False,  # DEVE usare submit_answer
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(query: str) -> list[str]:
    """
    Esegue il pipeline agentico su una query.

    Returns:
        Lista di item che rispondono alla query (per scoring Jaccard).
    """
    agent = build_agent()
    result = agent.run(query)

    if not result or not result.text:
        return []

    # Estrai la lista dal JSON di submit_answer
    try:
        data = json.loads(result.text)
        return data.get("answer", [])
    except json.JSONDecodeError:
        # Fallback: ritorna il testo grezzo come lista singola
        return [result.text]


if __name__ == "__main__":
    query = input("Query: ")
    answer = run(query)
    print(f"\nRisposta: {answer}")
    print(f"N. item: {len(answer)}")
