import torch
import torch.nn as nn
import pytest
from models.ranker import MLPRanker, build_ranker


@pytest.fixture
def cfg():
    return {
        "two_tower": {"embedding_dim": 64},
        "features": {"user_dense_dim": 16, "item_dense_dim": 16},
        "ranking": {"hidden_dims": [256, 128], "dropout": 0.0},
    }


@pytest.fixture
def ranker(cfg):
    r = build_ranker(cfg)
    r.eval()
    return r


def test_build_ranker_returns_mlp_ranker(ranker):
    assert isinstance(ranker, MLPRanker)


def test_input_dim_first_linear(ranker):
    assert ranker.net[0].in_features == 160


def test_output_shape(ranker):
    x = torch.randn(32, 160)
    with torch.no_grad():
        assert ranker(x).shape == (32, 1)


def test_predict_output_range(ranker):
    """forward() returns unbounded scores; predict() is the calibrated head."""
    x = torch.randn(1000, 160)
    with torch.no_grad():
        out = ranker.predict(x)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_output_is_differentiable(cfg):
    r = build_ranker(cfg)
    out = r(torch.randn(4, 160))
    out.sum().backward()


def test_network_ends_in_a_linear_score(ranker):
    """
    The sigmoid was moved out of the network so the pairwise objective can work
    on unbounded scores — squashing compresses the positive/negative margin and
    flattens the gradient where the ranking loss needs it. predict() reapplies
    it for the serving contract.
    """
    assert isinstance(ranker.net[-1], nn.Linear)
    assert ranker.net[-1].out_features == 1


def test_hidden_output_dims(ranker):
    assert ranker.net[0].out_features == 256
    assert ranker.net[3].out_features == 128
    assert ranker.net[6].out_features == 1
