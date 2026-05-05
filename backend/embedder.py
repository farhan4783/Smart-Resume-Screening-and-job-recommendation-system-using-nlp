"""
backend/embedder.py
--------------------
Singleton S-BERT embedder using 'all-mpnet-base-v2' (as specified in the
research paper).  Generates 768-dimensional dense semantic vectors for
resume-to-JD cosine comparison.

Reference:
    Paper Section V-D — "The text is pre-processed and then it is fed into
    a pre-trained S-BERT model (particularly, all-mpnet-base-v2
    architecture) … outputs a sparse human-vector of size 768."
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

# Model specified in the research paper (Section V-D)
_MODEL_NAME = "all-mpnet-base-v2"


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print(f"[embedder] Loading S-BERT model: {_MODEL_NAME} (768-D) ...")
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def encode(text: str) -> np.ndarray:
    """
    Encode text into a 768-dimensional S-BERT embedding vector.

    Args:
        text: Input string (resume text, job description, etc.)

    Returns:
        np.ndarray of shape (768,)
    """
    model = _get_model()
    # Truncate to stay within model's 384-token context window
    truncated = text[:8000]
    embedding = model.encode(truncated, convert_to_numpy=True, normalize_embeddings=True)
    return embedding


def encode_batch(texts: list[str]) -> np.ndarray:
    """
    Encode a batch of texts.

    Args:
        texts: List of strings

    Returns:
        np.ndarray of shape (N, 768)
    """
    model = _get_model()
    truncated = [t[:8000] for t in texts]
    embeddings = model.encode(truncated, convert_to_numpy=True, normalize_embeddings=True, batch_size=16, show_progress_bar=False)
    return embeddings
