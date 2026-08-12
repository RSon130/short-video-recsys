"""
Deterministic user-level A/B group assignment and metric comparison.

Simulates an A/B test by splitting users into two groups and comparing
a per-group metric (watch-time AUC) between two model variants.
"""
import hashlib

import numpy as np
import pandas as pd

from evaluation.metrics import watch_time_auc
from data.schema import Cols


def assign_group(user_id: int, traffic_split: float = 0.5, seed: int = 42) -> str:
    """
    Deterministically assign a user to group 'A' or 'B'.

    Uses an MD5 hash of (seed + user_id) modulo 1000 to produce a stable,
    pseudo-random assignment.  The same user always gets the same group for
    a given seed — this is essential for consistency across service restarts
    and for reproducible experiment replay.

    Why MD5 here?
        We're using it as a fast, uniform hash function for bucketing —
        not for any security purpose.  The output distribution is uniform
        enough to guarantee approximately `traffic_split * 100`% of users
        land in group A.

    Args:
        user_id:       The user's original (raw) ID.
        traffic_split: Fraction of users assigned to group A, e.g. 0.5.
        seed:          Experiment seed — change to run a new independent test.

    Returns:
        'A' or 'B'.
    """
    h = int(hashlib.md5(f"{seed}{user_id}".encode()).hexdigest(), 16)
    return "A" if (h % 1000) < int(traffic_split * 1000) else "B"


def run_ab_test(
    interactions: pd.DataFrame,
    model_a_scores: np.ndarray,
    model_b_scores: np.ndarray,
    cfg: dict,
) -> dict:
    """
    Run a simulated A/B test and report per-group watch-time AUC.

    Users are split by assign_group() into group A (evaluated with
    model_a_scores) and group B (evaluated with model_b_scores).  The
    row masks on `interactions` ensure each group only evaluates the
    scores for their own users' interactions.

    Typical use case:
        model_a_scores = two-tower cosine similarity scores (retrieval only)
        model_b_scores = MLP ranker predicted watch_ratio
        → compare whether adding a ranker improves AUC over retrieval alone.

    Args:
        interactions:    Interaction DataFrame with user_id and watch_ratio columns.
        model_a_scores:  Score array aligned row-for-row with `interactions`.
        model_b_scores:  Score array aligned row-for-row with `interactions`.
        cfg:             Config dict; reads evaluation.ab_test_traffic_split
                         and project.seed.

    Returns:
        Dict with keys 'group_A' and 'group_B', each containing:
            'n_users':        int — number of unique users in the group.
            'watch_time_auc': float — mean AUC for that group.
    """
    traffic_split = cfg["evaluation"]["ab_test_traffic_split"]
    seed = cfg["project"]["seed"]

    group_users_a: set = set()
    group_users_b: set = set()
    for uid in interactions[Cols.USER_ID].unique():
        group = assign_group(int(uid), traffic_split, seed)
        if group == "A":
            group_users_a.add(int(uid))
        else:
            group_users_b.add(int(uid))

    mask_a = interactions[Cols.USER_ID].isin(group_users_a)
    y_true_a = interactions.loc[mask_a, Cols.WATCH_RATIO].values
    y_score_a = model_a_scores[mask_a.values]

    mask_b = interactions[Cols.USER_ID].isin(group_users_b)
    y_true_b = interactions.loc[mask_b, Cols.WATCH_RATIO].values
    y_score_b = model_b_scores[mask_b.values]

    return {
        "group_A": {
            "n_users": len(group_users_a),
            "watch_time_auc": watch_time_auc(y_true_a, y_score_a),
        },
        "group_B": {
            "n_users": len(group_users_b),
            "watch_time_auc": watch_time_auc(y_true_b, y_score_b),
        },
    }
