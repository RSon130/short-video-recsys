import math

import numpy as np
import pandas as pd
import pytest
import torch

from features.dense_features import DenseFeatureStore
from training.train_ranking import (
    ListwiseRankingDataset,
    PairwiseRankingDataset,
    listwise_loss,
    pairwise_loss,
)


def make_df(rows):
    return pd.DataFrame(rows, columns=["user_id", "item_id", "watch_ratio"])


@pytest.fixture
def store():
    return DenseFeatureStore(
        pd.DataFrame({"user_id": [0, 1], "u_a": [0.1, 0.2]}),
        pd.DataFrame({"item_id": [0, 1, 2, 3, 4, 5], "i_a": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}),
    )


@pytest.fixture
def embeddings():
    rng = np.random.default_rng(0)
    return rng.random((2, 4), dtype=np.float32), rng.random((6, 4), dtype=np.float32)


@pytest.fixture
def toy_df():
    # user 0: item 0 watched through; items 1,2,3 abandoned
    # user 1: item 4 watched through; item 5 abandoned
    return make_df([
        (0, 0, 0.95), (0, 1, 0.10), (0, 2, 0.15), (0, 3, 0.20),
        (1, 4, 0.90), (1, 5, 0.05),
    ])


def test_batch_holds_one_positive_and_n_negatives(toy_df, store, embeddings):
    user_embs, item_embs = embeddings
    ds = ListwiseRankingDataset(toy_df, user_embs, item_embs, store, num_neg=3)

    batch = ds.collate([0, 1])

    assert batch["n_candidates"] == 4                 # 1 positive + 3 negatives
    assert batch["x"].shape[0] == 2 * 4               # rows are flattened


def test_positive_occupies_index_zero_of_each_block(toy_df, store, embeddings):
    """
    listwise_loss takes cross-entropy against target 0, so the positive must be
    the first candidate in every block or the loss trains the wrong item.
    """
    user_embs, item_embs = embeddings
    ds = ListwiseRankingDataset(toy_df, user_embs, item_embs, store, num_neg=2)

    n_cand = 3
    batch = ds.collate([0])
    rows = batch["x"].numpy().reshape(1, n_cand, -1)

    # Candidate 0 must equal the row built for the known positive (user 0, item 0).
    expected = store.build_input(user_embs[0], item_embs[0], 0, 0)
    np.testing.assert_allclose(rows[0, 0], expected, rtol=1e-6)


def test_negatives_stay_within_the_same_user(toy_df, store, embeddings):
    user_embs, item_embs = embeddings
    ds = ListwiseRankingDataset(toy_df, user_embs, item_embs, store, num_neg=5)

    for _ in range(10):
        negs = ds.sample_negatives(np.array([0, 1]), 5)
        assert set(negs[0].tolist()) <= {1, 2, 3}     # user 0's abandoned items
        assert set(negs[1].tolist()) <= {5}           # user 1's only negative


def test_more_negatives_means_more_contrasts_per_step(toy_df, store, embeddings):
    """The point of listwise: each update sees several alternatives, not one."""
    user_embs, item_embs = embeddings
    few = ListwiseRankingDataset(toy_df, user_embs, item_embs, store, num_neg=1)
    many = ListwiseRankingDataset(toy_df, user_embs, item_embs, store, num_neg=4)

    assert few.collate([0])["n_candidates"] == 2
    assert many.collate([0])["n_candidates"] == 5


def test_listwise_loss_rewards_the_positive_ranking_first():
    winning = torch.tensor([[5.0, 0.0, 0.0, 0.0]])
    losing = torch.tensor([[0.0, 5.0, 5.0, 5.0]])

    assert listwise_loss(winning, 4) < listwise_loss(losing, 4)


def test_listwise_loss_at_uniform_scores_is_log_n():
    """Uniform scores means chance; loss must equal log(n_candidates)."""
    scores = torch.zeros(1, 4)

    assert abs(listwise_loss(scores, 4).item() - math.log(4)) < 1e-6


def test_listwise_loss_reshapes_flat_scores_into_blocks():
    # Two users, three candidates each; positives already ranked first.
    flat = torch.tensor([3.0, 0.0, 0.0, 3.0, 0.0, 0.0])

    loss = listwise_loss(flat, 3)

    assert loss.shape == torch.Size([])
    assert loss.item() < math.log(3)


def test_listwise_with_one_negative_matches_pairwise_loss():
    """
    Pairwise BPR is the N=1 case of a softmax over two candidates:
    -log sigma(p - n) == -log softmax([p, n])[0]. Sharing the sampling code
    between the two datasets is only safe because this identity holds.
    """
    pos, neg = 1.3, -0.4

    listwise = listwise_loss(torch.tensor([[pos, neg]]), 2)
    pairwise = pairwise_loss(torch.tensor([pos]), torch.tensor([neg]))

    assert abs(listwise.item() - pairwise.item()) < 1e-6


def test_listwise_loss_gradients_flow():
    scores = torch.tensor([[1.0, 0.5, 0.2]], requires_grad=True)

    listwise_loss(scores, 3).backward()

    assert scores.grad is not None
    assert scores.grad[0, 0] < 0      # pushes the positive score up


def test_listwise_dataset_inherits_pairwise_labelling(toy_df, store, embeddings):
    """Both objectives must agree on what a positive is."""
    user_embs, item_embs = embeddings
    listwise = ListwiseRankingDataset(toy_df, user_embs, item_embs, store, num_neg=2)
    pairwise = PairwiseRankingDataset(toy_df, user_embs, item_embs, store)

    assert len(listwise) == len(pairwise)
    assert listwise.pos_iids.tolist() == pairwise.pos_iids.tolist()
