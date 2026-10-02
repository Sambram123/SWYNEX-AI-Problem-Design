"""
similarity.py
-------------
Generates sentence embeddings using paraphrase-MiniLM-L3-v2 and computes
pairwise cosine similarity scores between review texts.

Model: sentence-transformers/paraphrase-MiniLM-L3-v2
  - Lightweight (~17 MB), CPU-friendly, no GPU required.
  - Produces 384-dimensional sentence embeddings.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

MODEL_NAME = "paraphrase-MiniLM-L3-v2"
_model: SentenceTransformer | None = None  # module-level singleton


def get_model() -> SentenceTransformer:
    """
    Return the shared SentenceTransformer model, loading it on first call.

    The model is cached in a module-level variable so it is only downloaded
    and initialised once per Python process.

    Returns
    -------
    SentenceTransformer
        The loaded paraphrase-MiniLM-L3-v2 model.
    """
    global _model
    if _model is None:
        print(f"[similarity] Loading model '{MODEL_NAME}' ...")
        _model = SentenceTransformer(MODEL_NAME)
        print("[similarity] Model loaded.")
    return _model


# ---------------------------------------------------------------------------
# Embedding generation
# ---------------------------------------------------------------------------

def encode_texts(
    texts: list[str],
    batch_size: int = 64,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Encode a list of strings into sentence embeddings.

    Parameters
    ----------
    texts : list[str]
        Review texts to encode.
    batch_size : int
        Number of texts processed per batch (default 64).
    show_progress : bool
        Show a tqdm progress bar while encoding (default True).

    Returns
    -------
    np.ndarray
        Array of shape (len(texts), 384) containing L2-normalised embeddings.
    """
    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2 normalise -> cosine = dot product
    )
    return embeddings


# ---------------------------------------------------------------------------
# Cosine similarity helpers
# ---------------------------------------------------------------------------

def cosine_similarity_pair(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """
    Compute cosine similarity between two 1-D embedding vectors.

    Parameters
    ----------
    emb_a, emb_b : np.ndarray
        1-D embedding arrays (already L2-normalised).

    Returns
    -------
    float
        Cosine similarity in [-1, 1].
    """
    return float(np.dot(emb_a, emb_b))


def pairwise_cosine_similarity(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute the full NxN pairwise cosine-similarity matrix.

    Parameters
    ----------
    embeddings : np.ndarray
        Shape (N, D) - L2-normalised embedding matrix.

    Returns
    -------
    np.ndarray
        Shape (N, N) similarity matrix with values in [-1, 1].
    """
    return sk_cosine_similarity(embeddings)
