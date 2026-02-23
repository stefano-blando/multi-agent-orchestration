"""
Pipeline di ingestion: parsing documenti → chunking → embedding → Qdrant.

Supporta: PDF, DOCX, HTML, CSV (formati attesi dalla challenge).
Gestisce documenti rumorosi (typo, formati misti).

Uso:
    python src/ingest.py --data_dir data/raw/
"""

import os
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "hackapizza")
EMBEDDING_MODE = os.getenv("EMBEDDING_MODE", "local")
BM25_CORPUS_PATH = os.getenv("BM25_CORPUS_PATH", "data/index/bm25_corpus.jsonl")
EMBEDDING_MOCK_DIM = int(os.getenv("EMBEDDING_MOCK_DIM", "64"))


def get_embedder():
    """Ritorna l'embedder corretto in base alla configurazione."""
    if EMBEDDING_MODE == "local":
        # Embedding gratis su GPU (RTX A1000 / Kaggle T4)
        from sentence_transformers import SentenceTransformer
        from src.utils import get_device
        model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)
        model = model.to(get_device())
        return model
    elif EMBEDDING_MODE == "openai":
        # Fallback API
        from datapizza.embedders.openai import OpenAIEmbedder
        return OpenAIEmbedder(api_key=os.getenv("OPENAI_API_KEY"))
    elif EMBEDDING_MODE == "mock":
        # Modalita' offline deterministica (no API/GPU).
        return None
    else:
        raise ValueError(f"EMBEDDING_MODE non supportato: {EMBEDDING_MODE}")


def embed_text(embedder, text: str) -> list[float]:
    """Genera embedding indipendentemente dal provider scelto."""
    if EMBEDDING_MODE == "local":
        return embedder.encode(text, normalize_embeddings=True).tolist()
    if EMBEDDING_MODE == "mock":
        from src.utils import deterministic_embedding

        return deterministic_embedding(text, dim=EMBEDDING_MOCK_DIM)
    vector = embedder.embed(text)
    if hasattr(vector, "tolist"):
        return vector.tolist()
    return list(vector)


def infer_vector_size(embedder) -> int:
    """Inferisce la dimensionalita' del vettore dal provider attivo."""
    if EMBEDDING_MODE == "local":
        return 768
    if EMBEDDING_MODE == "mock":
        return EMBEDDING_MOCK_DIM
    probe = embed_text(embedder, "dimension probe")
    if not probe:
        raise RuntimeError("Impossibile inferire dimensione embedding.")
    return len(probe)


def parse_document(file_path: Path) -> str:
    """
    Parsea un documento in testo grezzo.
    Usa Docling per PDF/DOCX, BeautifulSoup per HTML, pandas per CSV.
    """
    suffix = file_path.suffix.lower()

    if suffix in [".pdf", ".docx"]:
        from docling.document_converter import DocumentConverter
        converter = DocumentConverter()
        result = converter.convert(str(file_path))
        return result.document.export_to_markdown()

    elif suffix in [".html", ".htm"]:
        from bs4 import BeautifulSoup
        with open(file_path, encoding="utf-8", errors="replace") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
        return soup.get_text(separator="\n", strip=True)

    elif suffix == ".csv":
        import pandas as pd
        df = pd.read_csv(file_path, on_bad_lines="skip")
        return df.to_string()

    else:
        # Fallback: testo grezzo
        with open(file_path, encoding="utf-8", errors="replace") as f:
            return f.read()


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """Divide il testo in chunk con overlap."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def _get_collection_vector_size(client, collection_name: str) -> int | None:
    """Ritorna la dimensionalita' vettore della collection, se leggibile."""
    try:
        info = client.get_collection(collection_name=collection_name)
        vectors_cfg = info.config.params.vectors
        if hasattr(vectors_cfg, "size"):
            return int(vectors_cfg.size)
        if isinstance(vectors_cfg, dict) and vectors_cfg:
            first_key = next(iter(vectors_cfg))
            first_vector = vectors_cfg[first_key]
            if hasattr(first_vector, "size"):
                return int(first_vector.size)
    except Exception:
        return None
    return None


def ingest(data_dir: str, recreate: bool = False, batch_size: int = 256):
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    import uuid

    data_path = Path(data_dir)
    files = list(data_path.rglob("*"))
    files = [f for f in files if f.is_file()]

    print(f"Trovati {len(files)} file in {data_dir}")

    embedder = get_embedder()
    # Usa Docker se disponibile, altrimenti in-memory come fallback
    try:
        import httpx
        httpx.get(QDRANT_URL, timeout=2)
        client = QdrantClient(url=QDRANT_URL)
        print(f"Qdrant: connesso a {QDRANT_URL}")
    except Exception:
        print("Qdrant Docker non disponibile, uso in-memory (dati non persistiti).")
        client = QdrantClient(":memory:")

    vector_size = infer_vector_size(embedder)

    # Crea collection se non esiste
    collections = [c.name for c in client.get_collections().collections]
    if recreate and QDRANT_COLLECTION in collections:
        client.delete_collection(collection_name=QDRANT_COLLECTION)
        collections.remove(QDRANT_COLLECTION)
        print(f"Collection '{QDRANT_COLLECTION}' rimossa (--recreate).")

    if QDRANT_COLLECTION not in collections:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        print(f"Collection '{QDRANT_COLLECTION}' creata (dim={vector_size}).")
    else:
        existing_size = _get_collection_vector_size(client, QDRANT_COLLECTION)
        if existing_size is not None and existing_size != vector_size:
            raise ValueError(
                f"Collection '{QDRANT_COLLECTION}' ha dim={existing_size}, "
                f"ma l'embedder corrente usa dim={vector_size}. "
                "Rilancia con --recreate per rigenerarla."
            )

    points_batch = []
    total_chunks = 0
    corpus_path = Path(BM25_CORPUS_PATH)
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Corpus BM25 su file: {corpus_path}")

    with corpus_path.open("w", encoding="utf-8") as corpus_file:
        for file_path in files:
            print(f"  Parsing: {file_path.name}")
            try:
                text = parse_document(file_path)
                chunks = chunk_text(text)

                for i, chunk in enumerate(chunks):
                    vector = embed_text(embedder, chunk)
                    points_batch.append(
                        PointStruct(
                            id=str(uuid.uuid4()),
                            vector=vector,
                            payload={
                                "text": chunk,
                                "source": file_path.name,
                                "chunk_index": i,
                            },
                        )
                    )
                    corpus_file.write(
                        json.dumps(
                            {
                                "text": chunk,
                                "source": file_path.name,
                                "chunk_index": i,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    total_chunks += 1

                    if len(points_batch) >= batch_size:
                        client.upsert(collection_name=QDRANT_COLLECTION, points=points_batch)
                        points_batch = []
            except Exception as e:
                print(f"  ERRORE su {file_path.name}: {e}")

    if points_batch:
        client.upsert(collection_name=QDRANT_COLLECTION, points=points_batch)

    if total_chunks:
        print(f"\nIndicizzati {total_chunks} chunk in Qdrant.")
        print(f"Corpus BM25 persistito in: {corpus_path}")
    else:
        print("Nessun chunk indicizzato.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/raw/")
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()
    ingest(args.data_dir, recreate=args.recreate, batch_size=max(1, args.batch_size))
