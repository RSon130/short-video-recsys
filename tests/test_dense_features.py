import numpy as np
import pandas as pd
import pytest

from features.dense_features import DenseFeatureStore


@pytest.fixture
def store():
    user_features = pd.DataFrame({
        "user_id": [0, 1, 2],
        "u_a": [0.1, 0.2, 0.3],
        "u_b": [1.0, 2.0, 3.0],
    })
    item_features = pd.DataFrame({
        "item_id": [0, 1],
        "i_a": [0.5, 0.6],
    })
    return DenseFeatureStore(user_features, item_features)


@pytest.fixture
def embeddings():
    rng = np.random.default_rng(0)
    return rng.random((3, 4), dtype=np.float32), rng.random((2, 4), dtype=np.float32)


def test_dims_exclude_id_column(store):
    assert store.user_dense_dim == 2
    assert store.item_dense_dim == 1


def test_lookup_returns_the_right_row(store):
    assert store.user(1).tolist() == pytest.approx([0.2, 2.0])
    assert store.item(1).tolist() == pytest.approx([0.6])


def test_unknown_id_falls_back_to_zeros(store):
    assert store.user(99).tolist() == [0.0, 0.0]
    assert store.item(99).tolist() == [0.0]


def test_build_input_column_order(store, embeddings):
    user_embs, item_embs = embeddings
    x = store.build_input(user_embs[1], item_embs[0], uid=1, iid=0)

    assert x.shape == (4 + 4 + 2 + 1,)
    assert x[:4].tolist() == pytest.approx(user_embs[1].tolist())
    assert x[4:8].tolist() == pytest.approx(item_embs[0].tolist())
    assert x[8:10].tolist() == pytest.approx([0.2, 2.0])
    assert x[10:].tolist() == pytest.approx([0.5])


def test_build_matrix_matches_build_input_row_for_row(store, embeddings):
    """
    The vectorised path must produce exactly what the per-sample path produces.
    Training uses build_matrix and serving uses build_input; any divergence
    between them is train/serve skew reintroduced by the back door.
    """
    user_embs, item_embs = embeddings
    uids = np.array([0, 1, 2, 1])
    iids = np.array([0, 1, 0, 0])

    batch = store.build_matrix(user_embs[uids], item_embs[iids], uids, iids)
    expected = np.stack([
        store.build_input(user_embs[u], item_embs[i], u, i)
        for u, i in zip(uids, iids)
    ])

    np.testing.assert_array_equal(batch, expected)


def test_build_batch_matches_build_input_row_for_row(store, embeddings):
    user_embs, item_embs = embeddings
    candidates = [1, 0, 1]

    batch = store.build_batch(user_embs[2], item_embs, uid=2, iids=candidates)
    expected = np.stack([
        store.build_input(user_embs[2], item_embs[i], 2, i) for i in candidates
    ])

    np.testing.assert_array_equal(batch, expected)


def test_build_matrix_handles_unknown_ids(store, embeddings):
    user_embs, item_embs = embeddings
    uids = np.array([99])
    iids = np.array([0])

    batch = store.build_matrix(user_embs[[1]], item_embs[iids], uids, iids)

    assert batch[0, 8:10].tolist() == [0.0, 0.0]


def test_output_is_float32(store, embeddings):
    user_embs, item_embs = embeddings
    uids = np.array([0, 1])
    iids = np.array([0, 1])

    assert store.build_matrix(user_embs[uids], item_embs[iids], uids, iids).dtype == np.float32
    assert store.build_input(user_embs[0], item_embs[0], 0, 0).dtype == np.float32
