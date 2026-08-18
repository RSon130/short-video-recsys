import math

import numpy as np
import pandas as pd
import pytest
import torch

from features.dense_features import DenseFeatureStore
from models.ranker import MLPRanker
from training.train_ranking import PairwiseRankingDataset, pairwise_loss


def make_df(rows):
    return pd.DataFrame(rows, columns=["user_id", "item_id", "watch_ratio"])


@pytest.fixture
def store():
    return DenseFeatureStore(
        pd.DataFrame({"user_id": [0, 1], "u_a": [0.1, 0.2]}),
        pd.DataFrame({"item_id": [0, 1, 2, 3], "i_a": [0.5, 0.6, 0.7, 0.8]}),
    )


@pytest.fixture
def embeddings():
    rng = np.random.default_rng(0)
    return rng.random((2, 4), dtype=np.float32), rng.random((4, 4), dtype=np.float32)


@pytest.fixture
def toy_df():
    return make_df([
        (0, 0, 0.95), (0, 1, 0.10),
        (1, 2, 0.90), (1, 3, 0.05),
        (0, 2, 0.50),      # ambiguous middle band
    ])


def test_pairs_come_only_from_confident_positives(toy_df, store, embeddings):
    user_embs, item_embs = embeddings
    ds = PairwiseRankingDataset(toy_df, user_embs, item_embs, store)

    assert len(ds) == 2                      # (0,0) and (1,2)
    assert sorted(ds.pos_iids.tolist()) == [0, 2]


def test_negative_is_drawn_from_the_same_user(toy_df, store, embeddings):
    """
    The ranker must contrast two items the same user saw. Cross-user negatives
    would reintroduce the popularity signal the pairwise loss is meant to avoid.
    """
    user_embs, item_embs = embeddings
    ds = PairwiseRankingDataset(toy_df, user_embs, item_embs, store)

    batch_indices = list(range(len(ds)))
    for _ in range(20):
        batch = ds.collate(batch_indices)
        assert batch["x_pos"].shape == batch["x_neg"].shape


def test_collate_builds_both_matrices_via_shared_layout(toy_df, store, embeddings):
    user_embs, item_embs = embeddings
    ds = PairwiseRankingDataset(toy_df, user_embs, item_embs, store)

    batch = ds.collate([0])
    expected_width = 4 + 4 + store.user_dense_dim + store.item_dense_dim

    assert batch["x_pos"].shape == torch.Size([1, expected_width])
    assert batch["x_neg"].shape == torch.Size([1, expected_width])


def test_raises_without_positives(store, embeddings):
    user_embs, item_embs = embeddings
    with pytest.raises(ValueError, match="watch_ratio >="):
        PairwiseRankingDataset(make_df([(0, 0, 0.1)]), user_embs, item_embs, store)


def test_raises_without_negatives(store, embeddings):
    user_embs, item_embs = embeddings
    with pytest.raises(ValueError, match="watch_ratio <="):
        PairwiseRankingDataset(make_df([(0, 0, 0.9)]), user_embs, item_embs, store)


def test_pairwise_loss_rewards_correct_ordering():
    good = pairwise_loss(torch.tensor([2.0]), torch.tensor([-2.0]))
    bad = pairwise_loss(torch.tensor([-2.0]), torch.tensor([2.0]))

    assert good < bad
    assert good.item() < math.log(2) < bad.item()


def test_pairwise_loss_at_zero_margin_is_log2():
    loss = pairwise_loss(torch.tensor([0.5]), torch.tensor([0.5]))
    assert abs(loss.item() - math.log(2)) < 1e-6


def test_pairwise_loss_is_scale_free_in_absolute_score():
    """
    Only the margin matters. This is the point of dropping the sigmoid: the
    ranker may place scores anywhere on the real line provided the order holds.
    """
    a = pairwise_loss(torch.tensor([1.0]), torch.tensor([0.0]))
    b = pairwise_loss(torch.tensor([101.0]), torch.tensor([100.0]))

    assert abs(a.item() - b.item()) < 1e-5


def test_pairwise_loss_gradients_flow():
    pos = torch.tensor([0.5], requires_grad=True)
    neg = torch.tensor([0.2], requires_grad=True)

    pairwise_loss(pos, neg).backward()

    assert pos.grad is not None and neg.grad is not None


def test_predict_preserves_the_ordering_of_forward():
    """
    Sigmoid moved out of the network. It is monotonic, so every ranking metric
    is unchanged — serving may report calibrated scores without altering order.
    """
    torch.manual_seed(0)
    model = MLPRanker(input_dim=6, hidden_dims=[8], dropout=0.0)
    model.eval()
    x = torch.randn(32, 6)

    with torch.no_grad():
        raw = model(x).squeeze(-1)
        calibrated = model.predict(x).squeeze(-1)

    assert torch.equal(torch.argsort(raw), torch.argsort(calibrated))


def test_predict_is_bounded_to_unit_interval():
    torch.manual_seed(0)
    model = MLPRanker(input_dim=6, hidden_dims=[8], dropout=0.0)
    model.eval()

    with torch.no_grad():
        out = model.predict(torch.randn(64, 6) * 50)

    assert out.min() >= 0.0 and out.max() <= 1.0
