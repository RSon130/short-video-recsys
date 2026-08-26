"""
Offline evaluation metrics for the recommender system.

All metrics operate on ranked lists of item IDs and are computed at
cutoff K (e.g. Recall@10, NDCG@20).  These are the standard metrics
used in the recommender systems literature and industry.
"""
import math

import numpy as np
from sklearn.metrics import roc_auc_score


def recall_at_k(recommended: list, ground_truth: list, k: int) -> float:
    """
    Fraction of relevant items that appear in the top-K recommendations.

    Formula:  Recall@K = |relevant ∩ recommended[:K]| / min(K, |relevant|)

    Normalises by min(K, |relevant|) rather than |relevant| so that a system
    cannot be penalised for having fewer relevant items than K.

    Args:
        recommended:  Ordered list of recommended item IDs (best first).
        ground_truth: Set of item IDs the user actually interacted with.
        k:            Cutoff — only the first k recommendations are considered.

    Returns:
        Float in [0, 1].  Returns 0.0 if ground_truth is empty.
    """
    if not ground_truth:
        return 0.0
    hits = len(set(recommended[:k]) & set(ground_truth))
    return hits / min(k, len(ground_truth))


def ndcg_at_k(recommended: list, ground_truth: list, k: int) -> float:
    """
    Normalised Discounted Cumulative Gain at cutoff K.

    Measures ranking quality: relevant items ranked higher contribute more
    than those ranked lower.  Normalised by the Ideal DCG (IDCG) — the score
    achieved if all relevant items were ranked first.

    Formula:
        DCG@K  = Σ  1 / log₂(rank + 2)   for each relevant item at rank i (0-indexed)
        IDCG@K = Σ  1 / log₂(i + 2)       for i in 0..min(K, |relevant|)-1
        NDCG@K = DCG@K / IDCG@K

    Args:
        recommended:  Ordered list of recommended item IDs.
        ground_truth: Set of ground-truth relevant item IDs.
        k:            Cutoff rank.

    Returns:
        Float in [0, 1].  Returns 0.0 if ground_truth is empty.
    """
    gt_set = set(ground_truth)
    dcg = sum(1.0 / math.log2(i + 2) for i, item in enumerate(recommended[:k]) if item in gt_set)
    ideal_len = min(k, len(ground_truth))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_len))
    return dcg / idcg if idcg > 0 else 0.0


def watch_time_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Area Under the ROC Curve computed on watch-time labels.

    Converts continuous watch_ratio into binary labels (≥ 0.5 → positive)
    then computes sklearn's roc_auc_score.  AUC measures the probability
    that the model ranks a randomly chosen positive above a randomly chosen
    negative — 1.0 is perfect, 0.5 is random guessing.

    Degenerate guard: if all labels are the same class, AUC is undefined;
    returns 0.5 (chance level) to avoid exceptions.

    Args:
        y_true:  Observed watch_ratio values, shape (n,). Binarised at 0.5.
        y_score: Predicted scores from the ranker, shape (n,).

    Returns:
        AUC float in [0, 1].
    """
    y_binary = (y_true >= 0.5).astype(int)
    if y_binary.sum() == 0 or y_binary.sum() == len(y_binary):
        return 0.5
    return float(roc_auc_score(y_binary, y_score))


def evaluate_at_k_values(recommended_lists: list, ground_truth_lists: list, k_values: list) -> dict:
    """
    Compute mean Recall and NDCG across all users for multiple cutoffs.

    Iterates over every (recommended, ground_truth) pair and averages the
    per-user metric scores.  This is the standard offline evaluation loop
    run on the test split after training.

    Args:
        recommended_lists:  List of per-user recommended item lists.
        ground_truth_lists: List of per-user ground-truth item lists.
                            Must be the same length and order as recommended_lists.
        k_values:           List of cutoffs to evaluate, e.g. [5, 10, 20].

    Returns:
        Dict with keys like 'recall@5', 'ndcg@10', values are mean floats in [0, 1].
    """
    results = {}
    for k in k_values:
        recalls = [recall_at_k(r, g, k) for r, g in zip(recommended_lists, ground_truth_lists)]
        ndcgs = [ndcg_at_k(r, g, k) for r, g in zip(recommended_lists, ground_truth_lists)]
        results[f"recall@{k}"] = float(np.mean(recalls))
        results[f"ndcg@{k}"] = float(np.mean(ndcgs))
    return results


def per_user_metric(recommended_lists: list, ground_truth_lists: list,
                    k: int, metric: str = "recall") -> "np.ndarray":
    """
    Per-user metric values, unaggregated.

    evaluate_at_k_values() returns the mean, which is what goes in a results
    table. Significance testing needs the underlying distribution: the paired
    bootstrap in evaluation/ab_test.py differences these user by user, and the
    power calculation needs their standard deviation.

    Args:
        recommended_lists:  per-user recommended item lists.
        ground_truth_lists: per-user relevant item lists, same order.
        k:                  cutoff.
        metric:             "recall" or "ndcg".

    Returns:
        Array of per-user scores, aligned with the input order.
    """
    fn = {"recall": recall_at_k, "ndcg": ndcg_at_k}[metric]
    return np.array([fn(r, g, k) for r, g in zip(recommended_lists, ground_truth_lists)])
