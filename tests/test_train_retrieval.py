import math
import torch
import pytest
from training.train_retrieval import bpr_loss, get_device


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
