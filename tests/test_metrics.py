import math
import numpy as np
import pytest
from evaluation.metrics import recall_at_k, ndcg_at_k, watch_time_auc, evaluate_at_k_values


def test_recall_perfect_match():
    assert recall_at_k([1, 2, 3], [1, 2, 3], 3) == pytest.approx(1.0)


def test_recall_partial_match():
    assert recall_at_k([1, 2, 3], [1, 4], 3) == pytest.approx(0.5)


def test_recall_truncates_at_k():
    assert recall_at_k([1, 2, 3, 4], [4], 3) == pytest.approx(0.0)


def test_recall_empty_ground_truth():
    assert recall_at_k([1, 2, 3], [], 5) == pytest.approx(0.0)


def test_ndcg_perfect():
    assert ndcg_at_k([1, 2], [1, 2], 2) == pytest.approx(1.0)


def test_ndcg_partial():
    result = ndcg_at_k([1, 2, 3], [1, 3], 3)
    dcg = 1.0 / math.log2(2) + 0 + 1.0 / math.log2(4)
    idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
    expected = dcg / idcg
    assert abs(result - expected) < 1e-4


def test_ndcg_no_hits():
    assert ndcg_at_k([5, 6], [1, 2], 2) == pytest.approx(0.0)


def test_ndcg_empty_ground_truth():
    assert ndcg_at_k([1], [], 5) == pytest.approx(0.0)


def test_watch_time_auc_perfect():
    y_true = np.array([0.0, 0.0, 1.0, 1.0])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    assert abs(watch_time_auc(y_true, y_score) - 1.0) < 1e-6


def test_watch_time_auc_random_guessing():
    y_true = np.array([0.0, 1.0, 0.0, 1.0])
    y_score = np.array([0.5, 0.5, 0.5, 0.5])
    assert watch_time_auc(y_true, y_score) == pytest.approx(0.5)


def test_watch_time_auc_all_positive_degenerate():
    y_true = np.array([1.0, 1.0, 1.0])
    y_score = np.array([0.3, 0.5, 0.9])
    assert watch_time_auc(y_true, y_score) == pytest.approx(0.5)


def test_evaluate_at_k_values_keys():
    result = evaluate_at_k_values([[1, 2], [3, 4]], [[1], [3]], [5, 10])
    for key in {"recall@5", "recall@10", "ndcg@5", "ndcg@10"}:
        assert key in result
    for v in result.values():
        assert isinstance(v, float)
        assert 0.0 <= v <= 1.0


def test_evaluate_at_k_values_correct_recall():
    result = evaluate_at_k_values([[1, 2, 3]], [[1]], [3])
    assert abs(result["recall@3"] - 1.0) < 1e-6
