import math

import numpy as np
import pandas as pd
import pytest
import torch

from training.train_retrieval import InteractionDataset, bpr_loss, get_device


def make_df(rows):
    """rows: list of (user_id, item_id, watch_ratio)."""
    return pd.DataFrame(rows, columns=["user_id", "item_id", "watch_ratio"])


@pytest.fixture
def toy_df():
    # user 0: items 0,1 watched through; items 2,3 abandoned
    # user 1: item 4 watched through; item 5 abandoned
    return make_df([
        (0, 0, 0.95), (0, 1, 0.80), (0, 2, 0.10), (0, 3, 0.20),
        (1, 4, 0.90), (1, 5, 0.05),
        (0, 6, 0.50),   # ambiguous middle band — belongs to neither class
    ])


def test_positives_exclude_low_and_ambiguous_watch_ratio(toy_df):
    ds = InteractionDataset(toy_df, num_neg=2, pos_threshold=0.7, neg_threshold=0.3)

    assert len(ds) == 3                      # (0,0), (0,1), (1,4)
    assert set(ds.pos_items.tolist()) == {0, 1, 4}
    assert 6 not in ds.pos_items.tolist()    # 0.50 is neither positive nor negative


def test_negatives_are_drawn_from_the_same_users_low_engagement_items(toy_df):
    """
    The bug this guards against: sampling negatives uniformly from the whole
    item vocabulary. On a fully observed matrix that draws items the user
    actually enjoyed, erasing the ranking signal.
    """
    ds = InteractionDataset(toy_df, num_neg=8, pos_threshold=0.7, neg_threshold=0.3)

    for idx in range(len(ds)):
        sample = ds[idx]
        uid = int(sample["user_id"])
        allowed = {0: {2, 3}, 1: {5}}[uid]
        assert set(sample["neg_items"].tolist()) <= allowed


def test_negative_is_never_the_positive_item(toy_df):
    ds = InteractionDataset(toy_df, num_neg=8, pos_threshold=0.7, neg_threshold=0.3)
    for idx in range(len(ds)):
        sample = ds[idx]
        assert int(sample["pos_item"]) not in sample["neg_items"].tolist()


def test_user_without_own_negatives_falls_back_to_global_pool():
    df = make_df([
        (0, 0, 0.95),           # user 0 has no low-engagement item
        (1, 1, 0.90), (1, 2, 0.05),
    ])
    ds = InteractionDataset(df, num_neg=3, pos_threshold=0.7, neg_threshold=0.3)

    user0 = next(ds[i] for i in range(len(ds)) if int(ds[i]["user_id"]) == 0)
    assert set(user0["neg_items"].tolist()) <= {2}


def test_sample_shapes_and_dtypes(toy_df):
    ds = InteractionDataset(toy_df, num_neg=4, pos_threshold=0.7, neg_threshold=0.3)
    sample = ds[0]

    assert sample["user_id"].dtype == torch.int64
    assert sample["pos_item"].dtype == torch.int64
    assert sample["neg_items"].shape == torch.Size([4])


def test_raises_when_no_positives_exist():
    df = make_df([(0, 0, 0.1), (0, 1, 0.2)])
    with pytest.raises(ValueError, match="watch_ratio >="):
        InteractionDataset(df, num_neg=2, pos_threshold=0.7, neg_threshold=0.3)


def test_raises_when_no_negatives_exist():
    df = make_df([(0, 0, 0.9), (0, 1, 0.95)])
    with pytest.raises(ValueError, match="watch_ratio <="):
        InteractionDataset(df, num_neg=2, pos_threshold=0.7, neg_threshold=0.3)


def test_bpr_loss_positive_margin():
    user_emb = torch.tensor([[1.0]])
    pos_item_emb = torch.tensor([[1.0]])
    neg_item_emb = torch.tensor([[[0.0]]])
    expected = -math.log(1 / (1 + math.exp(-1.0)))
    assert abs(bpr_loss(user_emb, pos_item_emb, neg_item_emb).item() - expected) < 1e-4


def test_bpr_loss_zero_margin_equals_log2():
    user_emb = torch.tensor([[1.0]])
    pos_item_emb = torch.tensor([[0.5]])
    neg_item_emb = torch.tensor([[[0.5]]])
    assert abs(bpr_loss(user_emb, pos_item_emb, neg_item_emb).item() - math.log(2)) < 1e-5


def test_bpr_loss_negative_margin():
    user_emb = torch.tensor([[1.0]])
    pos_item_emb = torch.tensor([[0.0]])
    neg_item_emb = torch.tensor([[[1.0]]])
    assert bpr_loss(user_emb, pos_item_emb, neg_item_emb).item() > math.log(2)


def test_bpr_loss_batch_scalar():
    B, D, num_neg = 4, 8, 2
    user_emb = torch.randn(B, D)
    pos_item_emb = torch.randn(B, D)
    neg_item_emb = torch.randn(B, num_neg, D)
    loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
    assert loss.shape == torch.Size([])
    assert loss.item() > 0


def test_bpr_loss_gradients_flow():
    B, D, num_neg = 4, 8, 2
    user_emb = torch.randn(B, D, requires_grad=True)
    pos_item_emb = torch.randn(B, D)
    neg_item_emb = torch.randn(B, num_neg, D)
    loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
    loss.backward()
    assert user_emb.grad is not None


def test_get_device_cpu_explicit():
    cfg = {"training": {"retrieval": {"device": "cpu"}}}
    assert get_device(cfg) == torch.device("cpu")


def test_get_device_auto_returns_valid_device():
    cfg = {"training": {"retrieval": {"device": "auto"}}}
    result = get_device(cfg)
    assert isinstance(result, torch.device)
    assert result.type in ("cpu", "cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_get_device_cuda_skipped_on_cpu():
    cfg = {"training": {"retrieval": {"device": "cuda"}}}
    assert get_device(cfg) == torch.device("cuda")
