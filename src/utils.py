import os
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
