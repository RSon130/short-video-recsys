"""
FAISS index for approximate nearest-neighbour retrieval.

Tier 1: IndexFlatIP (exact inner product — no approximation needed at small scale).
"""
import faiss
import numpy as np
from pathlib import Path


def build_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a FAISS IndexFlatIP (exact inner-product search) from item embeddings.

    Why IndexFlatIP?
        - Inner product of two L2-normalised vectors equals cosine similarity.
        - 'Flat' means every vector is compared exhaustively — no approximation.
        - At Tier 1 scale (1K–100K items) this is fast enough; Tier 2 switches
          to IVFFlat or IVFPQ for million-scale datasets.

    Args:
        embeddings: Item embedding matrix, shape (n_items, embedding_dim).
                    Values should be L2-normalised before calling this function.

    Returns:
        A populated faiss.IndexFlatIP ready for search.
    """
    embeddings = embeddings.astype("float32")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def query_index(index, user_emb: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Retrieve the top-K most similar items for a given user embedding.

    The search computes the inner product between `user_emb` and every item
    vector in the index. Because all vectors are L2-normalised, this is
    equivalent to cosine similarity. FAISS returns results sorted by score
    descending.

    Args:
        index:    A populated faiss.Index (built by build_index).
        user_emb: The query user embedding, shape (embedding_dim,) or
                  (1, embedding_dim). Must be L2-normalised.
        top_k:    Number of nearest neighbours to return.

    Returns:
        scores:  Inner-product similarity scores, shape (top_k,). Higher = more similar.
        indices: Integer item indices into the original embedding matrix,
                 shape (top_k,). Pass these to the ranker or item ID map.
    """
    if user_emb.ndim == 1:
        user_emb = user_emb.reshape(1, -1)
    user_emb = user_emb.astype("float32")
    scores, indices = index.search(user_emb, top_k)
    return scores[0], indices[0]


def save_index(index, path) -> None:
    """
    Persist a FAISS index to disk.

    The saved file is binary and can be reloaded with load_index() without
    rebuilding from embeddings. This is the checkpoint between the
    build_index script and the serving API.

    Args:
        index: A populated faiss.Index.
        path:  Destination file path (str or Path). Parent directory must exist.
    """
    faiss.write_index(index, str(path))


def load_index(path) -> faiss.Index:
    """
    Load a previously saved FAISS index from disk.

    Called once at API startup — the loaded index is held in memory for the
    lifetime of the process, so query latency is not affected by disk I/O.

    Args:
        path: Path to the saved index file (str or Path).

    Returns:
        The deserialized faiss.Index, ready to search.
    """
    return faiss.read_index(str(path))
