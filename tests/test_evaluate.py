import pandas as pd

from scripts.evaluate import build_ground_truth


def make_df(rows):
    return pd.DataFrame(rows, columns=["user_id", "item_id", "watch_ratio"])


def test_ground_truth_uses_engagement_not_presence():
    """
    On a fully-observed matrix nearly every (user, item) pair has a row, so
    treating "appears in the test split" as relevant measures exposure and
    inflates every metric. Relevance must come from watch_ratio.
    """
    df = make_df([
        (0, 1, 0.95),
        (0, 2, 0.05),   # shown, abandoned — not relevant
        (0, 3, 0.50),   # ambiguous — below threshold, not relevant
    ])

    gt = build_ground_truth(df, positive_threshold=0.7)

    assert gt == {0: [1]}


def test_ground_truth_groups_multiple_items_per_user():
    df = make_df([
        (0, 1, 0.9), (0, 2, 0.8),
        (1, 3, 0.95),
    ])

    gt = build_ground_truth(df, positive_threshold=0.7)

    assert sorted(gt[0]) == [1, 2]
    assert gt[1] == [3]


def test_users_without_relevant_items_are_omitted():
    df = make_df([(0, 1, 0.9), (1, 2, 0.1)])

    gt = build_ground_truth(df, positive_threshold=0.7)

    assert 1 not in gt


def test_threshold_is_inclusive():
    df = make_df([(0, 1, 0.7)])

    assert build_ground_truth(df, positive_threshold=0.7) == {0: [1]}
