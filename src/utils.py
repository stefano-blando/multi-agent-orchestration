import os
import hashlib
import torch


def get_device() -> str:
    """Ritorna il device disponibile: cuda (Linux/NVIDIA), mps (Mac), cpu."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_embedding_mode() -> str:
    return os.getenv("EMBEDDING_MODE", "local")


def deterministic_embedding(text: str, dim: int = 64) -> list[float]:
    """
    Embedding deterministico leggero per smoke test/offline benchmark.
    Non sostituisce embedding semantici reali, ma consente pipeline testabile senza API/GPU.
    """
    dim = max(8, int(dim))
    vec = [0.0] * dim
    tokens = text.lower().split()
    if not tokens:
        return vec

    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % dim
        sign = -1.0 if digest[4] % 2 else 1.0
        mag = (digest[5] / 255.0) + 0.5
        vec[idx] += sign * mag

    norm = sum(x * x for x in vec) ** 0.5
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec
