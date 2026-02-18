"""
Pipeline di ingestion: parsing documenti → chunking → embedding → Qdrant.

Supporta: PDF, DOCX, HTML, CSV (formati attesi dalla challenge).
Gestisce documenti rumorosi (typo, formati misti).

Uso:
    python src/ingest.py --data_dir data/raw/
"""

import os
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "hackapizza")
EMBEDDING_MODE = os.getenv("EMBEDDING_MODE", "local")


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
    else:
        raise ValueError(f"EMBEDDING_MODE non supportato: {EMBEDDING_MODE}")


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


def ingest(data_dir: str):
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

    # Crea collection se non esiste
    collections = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION not in collections:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )
        print(f"Collection '{QDRANT_COLLECTION}' creata.")

    points = []
    for file_path in files:
        print(f"  Parsing: {file_path.name}")
        try:
            text = parse_document(file_path)
            chunks = chunk_text(text)

            for i, chunk in enumerate(chunks):
                if EMBEDDING_MODE == "local":
                    vector = embedder.encode(chunk, normalize_embeddings=True).tolist()
                else:
                    vector = embedder.embed(chunk)

                points.append(
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
        except Exception as e:
            print(f"  ERRORE su {file_path.name}: {e}")

    if points:
        client.upsert(collection_name=QDRANT_COLLECTION, points=points)
        print(f"\nIndicizzati {len(points)} chunk in Qdrant.")
    else:
        print("Nessun chunk indicizzato.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/raw/")
    args = parser.parse_args()
    ingest(args.data_dir)
