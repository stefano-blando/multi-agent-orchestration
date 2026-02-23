"""
Hybrid search: BM25 (keyword) + semantic (vector) con score fusion.

Il corpus BM25 e' persistito su disco durante l'ingestion per poter
funzionare anche in processi separati da `ingest.py`.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from rank_bm25 import BM25Okapi


def _bm25_corpus_path() -> str:
    return os.getenv("BM25_CORPUS_PATH", "data/index/bm25_corpus.jsonl")


def _embedding_mock_dim() -> int:
    return max(8, int(os.getenv("EMBEDDING_MOCK_DIM", "64")))


def _rerank_mode() -> str:
    return os.getenv("RERANK_MODE", "heuristic").strip().lower()


def _rerank_cross_encoder_model() -> str:
    return os.getenv("RERANK_CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")


@dataclass
class Corpus:
    """Corpus di testi per BM25 con lazy-load da file."""

    chunks: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    chunk_indices: list[int] = field(default_factory=list)
    _bm25: Optional[BM25Okapi] = field(default=None, repr=False)
    _loaded_from_disk: bool = field(default=False, repr=False)
    _loaded_path: str = field(default="", repr=False)

    def add(self, text: str, source: str, chunk_index: int = -1):
        self.chunks.append(text)
        self.sources.append(source)
        self.chunk_indices.append(chunk_index)
        self._bm25 = None

    def load_from_jsonl(self, path: str):
        if self._loaded_from_disk and self._loaded_path == path and not self.is_empty():
            return
        if self._loaded_path != path:
            self.chunks.clear()
            self.sources.clear()
            self.chunk_indices.clear()
            self._bm25 = None
            self._loaded_from_disk = False
            self._loaded_path = path
        corpus_path = Path(path)
        if not corpus_path.exists():
            return
        self._loaded_from_disk = True
        with corpus_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = row.get("text")
                source = row.get("source")
                chunk_index = row.get("chunk_index", -1)
                if not text or not source:
                    continue
                self.add(text=text, source=source, chunk_index=int(chunk_index))

    def get_bm25(self) -> BM25Okapi:
        if self._bm25 is None:
            tokenized = [doc.lower().split() for doc in self.chunks]
            self._bm25 = BM25Okapi(tokenized)
        return self._bm25

    def is_empty(self) -> bool:
        return len(self.chunks) == 0

    def by_source(self, source: str) -> list[dict]:
        rows = []
        for text, src, idx in zip(self.chunks, self.sources, self.chunk_indices):
            if src == source:
                rows.append({"text": text, "source": src, "index": idx, "score": 0.0})
        return sorted(rows, key=lambda r: (r["index"], r["text"]))


corpus = Corpus()
_HYBRID_CACHE: dict[str, tuple[float, list[dict]]] = {}


def _ensure_corpus_loaded():
    path = _bm25_corpus_path()
    if corpus.is_empty() or corpus._loaded_path != path:
        corpus.load_from_jsonl(path)


def _hybrid_cache_ttl_seconds() -> int:
    return max(0, int(os.getenv("HYBRID_CACHE_TTL_SECONDS", "90")))


def _hybrid_cache_key(query: str, top_k: int) -> str:
    corpus_path = _bm25_corpus_path()
    try:
        bm25_mtime = os.path.getmtime(corpus_path)
    except OSError:
        bm25_mtime = 0.0
    return json.dumps(
        {
            "query": query.strip().lower(),
            "top_k": int(top_k),
            "bm25_path": corpus_path,
            "bm25_mtime": round(float(bm25_mtime), 3),
            "qdrant_collection": os.getenv("QDRANT_COLLECTION", "hackapizza"),
            "embedding_mode": os.getenv("EMBEDDING_MODE", "local"),
            "rerank_mode": _rerank_mode(),
        },
        sort_keys=True,
        ensure_ascii=False,
    )


@lru_cache(maxsize=1)
def _get_qdrant_client() -> Optional[QdrantClient]:
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    try:
        import httpx

        httpx.get(qdrant_url, timeout=2)
        return QdrantClient(url=qdrant_url)
    except Exception:
        return None


@lru_cache(maxsize=1)
def _get_embedder():
    embedding_mode = os.getenv("EMBEDDING_MODE", "local")
    if embedding_mode == "local":
        from sentence_transformers import SentenceTransformer

        from src.utils import get_device

        model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)
        return model.to(get_device())
    if embedding_mode == "openai":
        from datapizza.embedders.openai import OpenAIEmbedder

        return OpenAIEmbedder(api_key=os.getenv("OPENAI_API_KEY"))
    if embedding_mode == "mock":
        return None
    return None


def _embed_text(text: str) -> Optional[list[float]]:
    embedding_mode = os.getenv("EMBEDDING_MODE", "local")
    if embedding_mode == "mock":
        from src.utils import deterministic_embedding

        return deterministic_embedding(text, dim=_embedding_mock_dim())
    embedder = _get_embedder()
    if embedder is None:
        return None
    try:
        if embedding_mode == "local":
            return embedder.encode(text, normalize_embeddings=True).tolist()
        vector = embedder.embed(text)
        if hasattr(vector, "tolist"):
            return vector.tolist()
        return list(vector)
    except Exception:
        return None


def reset_runtime_state():
    """Resetta cache e stato in-memory; utile per benchmark/test offline."""
    corpus.chunks.clear()
    corpus.sources.clear()
    corpus.chunk_indices.clear()
    corpus._bm25 = None
    corpus._loaded_from_disk = False
    corpus._loaded_path = ""
    _get_qdrant_client.cache_clear()
    _get_embedder.cache_clear()
    _get_cross_encoder.cache_clear()
    _HYBRID_CACHE.clear()


@lru_cache(maxsize=1)
def _get_cross_encoder():
    mode = _rerank_mode()
    if mode != "cross_encoder":
        return None
    try:
        from sentence_transformers import CrossEncoder

        return CrossEncoder(_rerank_cross_encoder_model())
    except Exception:
        return None


def _lexical_score(query: str, text: str) -> float:
    q_tokens = set(re.findall(r"\w+", query.lower()))
    t_tokens = set(re.findall(r"\w+", text.lower()))
    if not q_tokens:
        return 0.0
    return len(q_tokens & t_tokens) / len(q_tokens)


def _heuristic_rerank(query: str, results: list[dict]) -> list[dict]:
    if not results:
        return []
    base_scores = [float(r.get("score", 0.0)) for r in results]
    b_min, b_max = min(base_scores), max(base_scores)
    denom = (b_max - b_min) if b_max != b_min else 1.0
    reranked = []
    for row in results:
        base = float(row.get("score", 0.0))
        base_norm = (base - b_min) / denom
        lex = _lexical_score(query, row.get("text", ""))
        final = 0.55 * base_norm + 0.45 * lex
        reranked.append({**row, "score": float(final)})
    reranked.sort(key=lambda r: r["score"], reverse=True)
    return reranked


def _cross_encoder_rerank(query: str, results: list[dict]) -> list[dict]:
    model = _get_cross_encoder()
    if model is None or not results:
        return _heuristic_rerank(query, results)
    try:
        pairs = [[query, row.get("text", "")] for row in results]
        ce_scores = model.predict(pairs)
        reranked = []
        for row, score in zip(results, ce_scores):
            reranked.append({**row, "score": float(score)})
        reranked.sort(key=lambda r: r["score"], reverse=True)
        return reranked
    except Exception:
        return _heuristic_rerank(query, results)


def _rerank_results(query: str, results: list[dict], top_k: int) -> list[dict]:
    mode = _rerank_mode()
    if mode in {"none", "off"}:
        return results[:top_k]
    if mode == "cross_encoder":
        return _cross_encoder_rerank(query, results)[:top_k]
    return _heuristic_rerank(query, results)[:top_k]


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Combina piu' ranking in un unico score RRF."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def bm25_search(query: str, top_k: int = 10) -> list[dict]:
    """Ricerca BM25 sul corpus persistito."""
    _ensure_corpus_loaded()
    if corpus.is_empty():
        return []

    bm25 = corpus.get_bm25()
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    rows = [
        {
            "text": corpus.chunks[i],
            "source": corpus.sources[i],
            "score": float(scores[i]),
            "index": corpus.chunk_indices[i],
        }
        for i in ranked_indices
    ]
    positive = [row for row in rows if row["score"] > 0]
    return positive if positive else rows


def semantic_search_qdrant(query_vector: Optional[list[float]], top_k: int = 10) -> list[dict]:
    """Ricerca semantica su Qdrant."""
    if not query_vector:
        return []

    client = _get_qdrant_client()
    if client is None:
        return []

    collection = os.getenv("QDRANT_COLLECTION", "hackapizza")
    try:
        results = client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=top_k,
        )
        rows = []
        for point in results.points:
            payload = point.payload or {}
            rows.append(
                {
                    "text": payload.get("text", ""),
                    "source": payload.get("source", "unknown"),
                    "score": float(point.score),
                    "index": int(payload.get("chunk_index", -1)),
                }
            )
        return rows
    except Exception:
        return []


def get_document_chunks(source: str, max_chunks: Optional[int] = None) -> list[dict]:
    """Ritorna tutti i chunk di un documento, da Qdrant o fallback BM25 corpus."""
    client = _get_qdrant_client()
    collection = os.getenv("QDRANT_COLLECTION", "hackapizza")

    if client is not None:
        try:
            chunk_rows: list[dict] = []
            next_offset = None
            query_filter = Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=source))]
            )
            while True:
                points, next_offset = client.scroll(
                    collection_name=collection,
                    scroll_filter=query_filter,
                    limit=256,
                    offset=next_offset,
                    with_payload=True,
                    with_vectors=False,
                )
                if not points:
                    break
                for point in points:
                    payload = point.payload or {}
                    chunk_rows.append(
                        {
                            "text": payload.get("text", ""),
                            "source": payload.get("source", source),
                            "score": 0.0,
                            "index": int(payload.get("chunk_index", -1)),
                        }
                    )
                if next_offset is None:
                    break
                if max_chunks is not None and len(chunk_rows) >= max_chunks:
                    break
            chunk_rows.sort(key=lambda r: (r["index"], r["text"]))
            if max_chunks is not None:
                chunk_rows = chunk_rows[:max_chunks]
            if chunk_rows:
                return chunk_rows
        except Exception:
            pass

    _ensure_corpus_loaded()
    rows = corpus.by_source(source)
    if max_chunks is not None:
        rows = rows[:max_chunks]
    return rows


def hybrid_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Hybrid search: BM25 + semantic con Reciprocal Rank Fusion.

    Se BM25 e' vuoto usa solo semantic, se Qdrant non disponibile usa solo BM25.
    """
    cache_ttl = _hybrid_cache_ttl_seconds()
    cache_key = _hybrid_cache_key(query, top_k)
    now = time.time()
    if cache_ttl > 0:
        cached = _HYBRID_CACHE.get(cache_key)
        if cached and (now - cached[0]) <= cache_ttl:
            return [dict(row) for row in cached[1]]

    bm25_results = bm25_search(query, top_k=top_k * 2)
    query_vector = _embed_text(query)
    semantic_results = semantic_search_qdrant(query_vector, top_k=top_k * 2)

    def _store_cache(rows: list[dict]) -> list[dict]:
        if cache_ttl > 0:
            _HYBRID_CACHE[cache_key] = (now, [dict(row) for row in rows])
            if len(_HYBRID_CACHE) > 512:
                oldest = sorted(_HYBRID_CACHE.keys(), key=lambda k: _HYBRID_CACHE[k][0])[:128]
                for key in oldest:
                    _HYBRID_CACHE.pop(key, None)
        return rows

    if not bm25_results and not semantic_results:
        return []
    if not bm25_results:
        return _store_cache(_rerank_results(query, semantic_results, top_k=top_k))
    if not semantic_results:
        return _store_cache(_rerank_results(query, bm25_results, top_k=top_k))

    # Chiave robusta: evita collisioni tra chunk uguali da fonti diverse.
    def key_of(row: dict) -> tuple[str, int, str]:
        return (row.get("source", "unknown"), int(row.get("index", -1)), row.get("text", ""))

    merged: dict[tuple[str, int, str], dict] = {}
    for row in bm25_results + semantic_results:
        merged[key_of(row)] = row

    keys = list(merged.keys())
    key_to_idx = {k: i for i, k in enumerate(keys)}
    bm25_ranking = [key_to_idx[key_of(r)] for r in bm25_results if key_of(r) in key_to_idx]
    semantic_ranking = [key_to_idx[key_of(r)] for r in semantic_results if key_of(r) in key_to_idx]
    fused = reciprocal_rank_fusion([bm25_ranking, semantic_ranking])
    fused_rows = [{**merged[keys[idx]], "score": score} for idx, score in fused]
    final_rows = _rerank_results(query, fused_rows, top_k=top_k)
    return _store_cache(final_rows)


def suggest_candidates(query: str, top_k_docs: int = 8, max_candidates: int = 20) -> list[str]:
    """
    Suggerisce item candidati da validare estraendo nomi dai risultati retrieval.
    """
    rows = hybrid_search(query, top_k=top_k_docs)
    candidates: list[str] = []
    seen: set[str] = set()
    pattern = re.compile(r"([A-Z][A-Za-z0-9' ]{2,50})\s*:")
    for row in rows:
        text = row.get("text", "")
        for match in pattern.findall(text):
            candidate = match.strip()
            if len(candidate.split()) <= 8 and candidate not in seen:
                seen.add(candidate)
                candidates.append(candidate)
                if len(candidates) >= max_candidates:
                    return candidates
    return candidates
