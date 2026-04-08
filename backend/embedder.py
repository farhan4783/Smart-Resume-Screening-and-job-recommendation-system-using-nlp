"""
backend/embedder.py
--------------------
Singleton sentence-transformer embedder using 'all-MiniLM-L6-v2'.
Generates 384-dimensional contextual embeddings for text.
"""

# Force sentence-transformers to use PyTorch backend only
# Prevents TensorFlow DLL load failures on Windows
import os as _os
_os.environ["USE_TF"] = "0"
_os.environ["USE_JAX"] = "0"
_os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
_os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import numpy as np

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def encode(text: str) -> np.ndarray:
    """
    Encode text into a 384-dimensional embedding vector.

    Args:
        text: Input string (resume text, job description, etc.)

    Returns:
        np.ndarray of shape (384,)
    """
    model = _get_model()
    # Truncate to 512-token-equivalent characters to stay within model limits
    truncated = text[:8000]
    embedding = model.encode(truncated, convert_to_numpy=True, normalize_embeddings=True)
    return embedding


def encode_batch(texts: list[str]) -> np.ndarray:
    """
    Encode a batch of texts.

    Args:
        texts: List of strings

    Returns:
        np.ndarray of shape (N, 384)
    """
    model = _get_model()
    truncated = [t[:8000] for t in texts]
    embeddings = model.encode(truncated, convert_to_numpy=True, normalize_embeddings=True, batch_size=16, show_progress_bar=False)
    return embeddings
