import numpy as np
import pytest
from retrieval.faiss_index import build_index, query_index, save_index, load_index


@pytest.fixture
def embeddings():
    rng = np.random.default_rng(0)
    embs = rng.random((50, 32)).astype("float32")
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / norms


@pytest.fixture
def index(embeddings):
    return build_index(embeddings)


def test_build_index_ntotal(index):
    assert index.ntotal == 50


def test_build_index_dimension(index):
    assert index.d == 32


def test_query_returns_correct_shapes(index, embeddings):
    scores, indices = query_index(index, embeddings[0], top_k=5)
    assert scores.shape == (5,)
    assert indices.shape == (5,)


def test_query_top_result_is_self(index, embeddings):
    scores, indices = query_index(index, embeddings[7], top_k=5)
    assert indices[0] == 7
    assert scores[0] > 0.99


def test_query_accepts_1d_input(index, embeddings):
    assert embeddings[0].ndim == 1
    _, indices = query_index(index, embeddings[0], top_k=3)
    assert len(indices) == 3


def test_query_scores_descending(index, embeddings):
    scores, _ = query_index(index, embeddings[0], top_k=10)
    for i in range(9):
        assert scores[i] >= scores[i + 1]


def test_save_load_round_trip(index, embeddings, tmp_path):
    path = tmp_path / "test.index"
    save_index(index, path)
    loaded = load_index(path)
    assert loaded.ntotal == 50
    _, idx = query_index(loaded, embeddings[7], top_k=1)
    assert idx[0] == 7


def test_build_accepts_float64(embeddings):
    result = build_index(embeddings.astype("float64"))
    assert result.ntotal == 50


def _unit(n, d=16, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d)).astype("float32")
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_all_index_types_build_and_search():
    from retrieval.faiss_index import INDEX_TYPES, build_index, query_index

    emb = _unit(500)
    for index_type in INDEX_TYPES:
        index = build_index(emb, index_type=index_type)
        scores, ids = query_index(index, emb[0], top_k=10)
        assert len(ids) == 10
        assert index.ntotal == 500


def test_flat_is_exact_so_a_vector_retrieves_itself():
    from retrieval.faiss_index import build_index, query_index

    emb = _unit(500)
    index = build_index(emb, index_type="flat")
    _, ids = query_index(index, emb[7], top_k=1)
    assert ids[0] == 7


def test_unknown_index_type_is_rejected():
    from retrieval.faiss_index import build_index

    with pytest.raises(ValueError, match="index_type must be one of"):
        build_index(_unit(50), index_type="annoy")


def test_ivf_nprobe_trades_recall_for_speed():
    """
    The knob must actually move recall. nprobe=1 scans one cell; scanning every
    cell must recover strictly more of the exact neighbours.
    """
    from retrieval.faiss_index import build_index, default_nlist

    emb = _unit(4000, d=32)
    exact = build_index(emb, index_type="flat")
    _, truth = exact.search(emb[:50], 20)

    nlist = default_nlist(4000)
    narrow = build_index(emb, index_type="ivfflat", nprobe=1)
    wide = build_index(emb, index_type="ivfflat", nprobe=nlist)

    def overlap(index):
        _, got = index.search(emb[:50], 20)
        return np.mean([len(set(t) & set(g)) / 20 for t, g in zip(truth, got)])

    assert overlap(narrow) < overlap(wide)
    assert overlap(wide) > 0.95        # exhaustive IVF ~ exact


def test_index_size_grows_with_catalogue():
    from retrieval.faiss_index import build_index, index_size_bytes

    small = index_size_bytes(build_index(_unit(100), index_type="flat"))
    large = index_size_bytes(build_index(_unit(1000), index_type="flat"))
    assert large > small * 5


def test_hnsw_costs_more_storage_than_flat():
    """The graph is the price of HNSW's speed — worth asserting, not assuming."""
    from retrieval.faiss_index import build_index, index_size_bytes

    emb = _unit(2000)
    flat = index_size_bytes(build_index(emb, index_type="flat"))
    hnsw = index_size_bytes(build_index(emb, index_type="hnsw"))
    assert hnsw > flat
