import torch
import pytest
from models.two_tower import UserTower, ItemTower, TwoTowerModel, build_model


@pytest.fixture
def cfg():
    return {
        "features": {
            "user_emb_dim": 64,
            "item_emb_dim": 64,
            "user_dense_dim": 16,
            "item_dense_dim": 16,
        },
        "two_tower": {"tower_hidden": [256, 128], "embedding_dim": 64, "dropout": 0.0},
    }


@pytest.fixture
def model(cfg):
    return build_model(cfg, n_users=100, n_items=200)


def test_build_model_returns_two_tower_model(model):
    assert isinstance(model, TwoTowerModel)


def test_user_tower_output_shape(model):
    user_ids = torch.arange(8)
    user_features = torch.randn(8, 16)
    model.eval()
    with torch.no_grad():
        output = model.user_tower(user_ids, user_features)
    assert output.shape == (8, 64)


def test_item_tower_output_shape(model):
    item_ids = torch.arange(8) % 200
    item_features = torch.randn(8, 16)
    model.eval()
    with torch.no_grad():
        output = model.item_tower(item_ids, item_features)
    assert output.shape == (8, 64)


def test_user_tower_l2_normalised(model):
    user_ids = torch.arange(8)
    user_features = torch.randn(8, 16)
    model.eval()
    with torch.no_grad():
        output = model.user_tower(user_ids, user_features)
    norms = torch.norm(output, dim=1)
    assert torch.allclose(norms, torch.ones(8), atol=1e-5)


def test_item_tower_l2_normalised(model):
    item_ids = torch.arange(8) % 200
    item_features = torch.randn(8, 16)
    model.eval()
    with torch.no_grad():
        output = model.item_tower(item_ids, item_features)
    norms = torch.norm(output, dim=1)
    assert torch.allclose(norms, torch.ones(8), atol=1e-5)


def test_two_tower_forward_returns_pair(model):
    model.eval()
    with torch.no_grad():
        result = model(
            torch.arange(8),
            torch.randn(8, 16),
            torch.arange(8) % 200,
            torch.randn(8, 16),
        )
    assert isinstance(result, tuple)
    assert len(result) == 2
    user_emb, item_emb = result
    assert user_emb.shape == (8, 64)
    assert item_emb.shape == (8, 64)
    assert torch.allclose(torch.norm(user_emb, dim=1), torch.ones(8), atol=1e-5)
    assert torch.allclose(torch.norm(item_emb, dim=1), torch.ones(8), atol=1e-5)


def test_cosine_similarity_range(model):
    model.eval()
    with torch.no_grad():
        user_emb, item_emb = model(
            torch.arange(8),
            torch.randn(8, 16),
            torch.arange(8) % 200,
            torch.randn(8, 16),
        )
    sim = (user_emb * item_emb).sum(dim=1)
    assert sim.min() >= -1.0 - 1e-5
    assert sim.max() <= 1.0 + 1e-5


def test_build_model_respects_n_users_n_items(cfg):
    m = build_model(cfg, n_users=50, n_items=300)
    assert m.user_tower.id_emb.num_embeddings == 50
    assert m.item_tower.id_emb.num_embeddings == 300
