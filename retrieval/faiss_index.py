"""
FAISS index for candidate retrieval — exact and approximate.

Three index types, selectable so the trade-off can be measured rather than
asserted (see scripts/benchmark_index.py and docs/index_benchmark.md):

  flat     IndexFlatIP. Exhaustive inner-product scan. Exact by construction,
           so recall against itself is 1.0 by definition. Cost is O(n * d) per
           query — linear in catalogue size.
  ivfflat  IndexIVFFlat. k-means partitions the space into `nlist` Voronoi
           cells; a query scans only the `nprobe` cells nearest to it. Cost
           falls roughly by nlist/nprobe. Recall falls because a true neighbour
           sitting in an unprobed cell can never be found.
  hnsw     IndexHNSWFlat. A navigable small-world graph, searched by greedy
           descent through progressively finer layers. Fast and high-recall,
           but the graph costs memory (M links per node) and builds far slower
           than either alternative.

All three use inner product, which equals cosine similarity because the towers
L2-normalise their output.
"""
import faiss
import numpy as np
from pathlib import Path

INDEX_TYPES = ("flat", "ivfflat", "hnsw")


def default_nlist(n_items: int) -> int:
    """
    Cell count for IVF, using the usual sqrt(n) heuristic.

    FAISS wants roughly 39-256 training points per centroid; sqrt(n) keeps cells
    populated enough to train while cutting the search space by about sqrt(n).
    """
    return max(1, int(np.sqrt(n_items)))


def build_index(
    embeddings: np.ndarray,
    index_type: str = "flat",
    nlist: int = None,
    nprobe: int = 10,
    hnsw_m: int = 32,
    ef_construction: int = 200,
    ef_search: int = 64,
) -> faiss.Index:
    """
    Build a FAISS index over item embeddings.

    Args:
        embeddings:      (n_items, dim), L2-normalised.
        index_type:      one of INDEX_TYPES.
        nlist:           IVF cell count; defaults to sqrt(n_items).
        nprobe:          IVF cells searched per query — the recall/speed knob.
        hnsw_m:          HNSW links per node; drives memory and recall.
        ef_construction: HNSW build-time search width (graph quality).
        ef_search:       HNSW query-time search width — its recall/speed knob.

    Returns:
        A populated, searchable faiss.Index.
    """
    if index_type not in INDEX_TYPES:
        raise ValueError(f"index_type must be one of {INDEX_TYPES}, got {index_type!r}")

    embeddings = np.ascontiguousarray(embeddings.astype("float32"))
    n_items, dim = embeddings.shape

    if index_type == "flat":
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        return index

    if index_type == "ivfflat":
        nlist = nlist or default_nlist(n_items)
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        # IVF must learn its cell centroids before anything can be added.
        index.train(embeddings)
        index.add(embeddings)
        index.nprobe = min(nprobe, nlist)
        return index

    index = faiss.IndexHNSWFlat(dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    index.add(embeddings)
    index.hnsw.efSearch = ef_search
    return index


def index_size_bytes(index: faiss.Index) -> int:
    """Serialised size — what the index costs to store and to ship in an image."""
    return len(faiss.serialize_index(index))


def query_index(index, user_emb: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Retrieve the top-K most similar items for one user embedding.

    Inner product over L2-normalised vectors is cosine similarity. FAISS returns
    results already sorted by score descending.

    Args:
        index:    a populated faiss.Index.
        user_emb: (dim,) or (1, dim), L2-normalised.
        top_k:    neighbours to return.

    Returns:
        scores, indices — both shape (top_k,).
    """
    if user_emb.ndim == 1:
        user_emb = user_emb.reshape(1, -1)
    user_emb = np.ascontiguousarray(user_emb.astype("float32"))
    scores, indices = index.search(user_emb, top_k)
    return scores[0], indices[0]


def save_index(index, path) -> None:
    """
    Persist a FAISS index to disk.

    The saved file is binary and reloads without rebuilding from embeddings.
    This is the checkpoint between build_index.py and the serving API.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def load_index(path) -> faiss.Index:
    """
    Load a previously saved FAISS index.

    Called once at API startup; the index is held in memory for the process
    lifetime, so query latency is not affected by disk I/O.
    """
    return faiss.read_index(str(path))
