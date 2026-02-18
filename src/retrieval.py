"""
Hybrid search: BM25 (keyword) + Semantic (vector) con score fusion.

Strategia:
  - BM25: ottimo per nomi propri, codici, termini tecnici con typo
  - Semantic: ottimo per sinonimi, parafasi, concetti simili
  - Reciprocal Rank Fusion (RRF): combina i due ranking in modo robusto

Il corpus BM25 viene popolato al momento dell'ingestion.
Il vector store Qdrant gestisce la ricerca semantica.

Uso standalone:
    python src/retrieval.py
"""

import os
import math
from dataclasses import dataclass, field
from typing import Optional

from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# ---------------------------------------------------------------------------
# Corpus BM25 in-memory (popolato da ingest.py)
# ---------------------------------------------------------------------------

@dataclass
class Corpus:
    """Corpus di testi per BM25. Singleton condiviso tra ingest e retrieval."""
    chunks: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    _bm25: Optional[BM25Okapi] = field(default=None, repr=False)

    def add(self, text: str, source: str):
        self.chunks.append(text)
        self.sources.append(source)
        self._bm25 = None  # invalida cache

    def get_bm25(self) -> BM25Okapi:
        if self._bm25 is None:
            tokenized = [doc.lower().split() for doc in self.chunks]
            self._bm25 = BM25Okapi(tokenized)
        return self._bm25

    def is_empty(self) -> bool:
        return len(self.chunks) == 0


# Istanza globale — importare da altri moduli
corpus = Corpus()

# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """
    Combina più ranking in un unico score RRF.

    Args:
        rankings: lista di liste di indici, ordinate per rilevanza
        k: costante di smoothing (default 60, standard letteratura)

    Returns:
        Lista di (indice, score_rrf) ordinata per score decrescente
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# Hybrid search
# ---------------------------------------------------------------------------

def bm25_search(query: str, top_k: int = 10) -> list[dict]:
    """Ricerca BM25 sul corpus in-memory."""
    if corpus.is_empty():
        return []

    bm25 = corpus.get_bm25()
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)

    # Indici ordinati per score decrescente
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    return [
        {
            "text": corpus.chunks[i],
            "source": corpus.sources[i],
            "score": float(scores[i]),
            "index": i,
        }
        for i in ranked_indices
        if scores[i] > 0
    ]


def semantic_search_qdrant(query_vector: list[float], top_k: int = 10) -> list[dict]:
    """Ricerca semantica su Qdrant."""
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection = os.getenv("QDRANT_COLLECTION", "hackapizza")

    try:
        import httpx
        httpx.get(qdrant_url, timeout=2)
        client = QdrantClient(url=qdrant_url)
    except Exception:
        # Fallback in-memory non ha dati — ritorna vuoto
        return []

    try:
        results = client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=top_k,
        )
        return [
            {
                "text": r.payload.get("text", ""),
                "source": r.payload.get("source", "unknown"),
                "score": r.score,
                "index": r.payload.get("chunk_index", -1),
            }
            for r in results.points
        ]
    except Exception:
        return []


def hybrid_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Hybrid search: BM25 + Semantic con Reciprocal Rank Fusion.

    Se il corpus BM25 è vuoto, usa solo semantic.
    Se Qdrant non è disponibile, usa solo BM25.
    """
    bm25_results = bm25_search(query, top_k=top_k * 2)

    # Embedding della query
    query_vector = None
    embedding_mode = os.getenv("EMBEDDING_MODE", "local")

    if embedding_mode == "local":
        try:
            from sentence_transformers import SentenceTransformer
            from src.utils import get_device
            model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)
            model = model.to(get_device())
            query_vector = model.encode(query, normalize_embeddings=True).tolist()
        except Exception:
            query_vector = None

    semantic_results = semantic_search_qdrant(query_vector, top_k=top_k * 2) if query_vector else []

    # Se solo uno dei due ha risultati, usalo direttamente
    if not bm25_results and not semantic_results:
        return []
    if not bm25_results:
        return semantic_results[:top_k]
    if not semantic_results:
        return bm25_results[:top_k]

    # RRF fusion
    # Mappiamo i testi ai loro indici nei due ranking
    all_texts = list({r["text"]: r for r in bm25_results + semantic_results}.values())
    text_to_idx = {r["text"]: i for i, r in enumerate(all_texts)}

    bm25_ranking = [text_to_idx[r["text"]] for r in bm25_results]
    semantic_ranking = [text_to_idx[r["text"]] for r in semantic_results]

    fused = reciprocal_rank_fusion([bm25_ranking, semantic_ranking])

    return [
        {**all_texts[idx], "score": score}
        for idx, score in fused[:top_k]
    ]


# ---------------------------------------------------------------------------
# Test standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Test BM25 con corpus mock...")

    # Popola il corpus
    docs = [
        ("La pizza cosmica è preparata con ingredienti galattici senza glutine.", "menu_galattico.pdf"),
        ("Il risotto di asteroidi contiene formaggio spaziale e burro stellare.", "menu_alieno.pdf"),
        ("Normativa 42-X: tutti i piatti devono essere certificati dalla Galactic Food Agency.", "normative.pdf"),
        ("Gli ospiti con allergie al lattosio devono evitare i prodotti caseari.", "allergie.pdf"),
        ("Pizza Nebulosa: impasto tradizionale con lievito cosmico, contiene glutine.", "menu_galattico.pdf"),
    ]
    for text, source in docs:
        corpus.add(text, source)

    query = "pizza senza glutine"
    print(f"\nQuery: '{query}'")
    results = bm25_search(query, top_k=3)
    print(f"\nBM25 results ({len(results)}):")
    for r in results:
        print(f"  [{r['source']}] score={r['score']:.3f} | {r['text'][:60]}...")

    query2 = "allergie latticini"
    print(f"\nQuery: '{query2}'")
    results2 = bm25_search(query2, top_k=3)
    print(f"\nBM25 results ({len(results2)}):")
    for r in results2:
        print(f"  [{r['source']}] score={r['score']:.3f} | {r['text'][:60]}...")

    print("\nBM25 OK.")
