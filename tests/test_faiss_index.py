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
