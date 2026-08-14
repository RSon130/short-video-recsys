import pandas as pd
import pytest

from evaluation.baselines import popularity_ranking, popularity_recommendations


def make_df(rows):
    return pd.DataFrame(rows, columns=["user_id", "item_id", "watch_ratio"])


def test_popularity_counts_engagement_not_exposure():
    """
    On a fully-observed matrix every item is shown to nearly every user, so raw
    interaction counts measure exposure. Item 1 is seen more often but engaged
    with less; item 2 should rank higher.
    """
    df = make_df([
        (0, 1, 0.1), (1, 1, 0.1), (2, 1, 0.1), (3, 1, 0.9),
        (0, 2, 0.9), (1, 2, 0.9),
    ])

    ranking = popularity_ranking(df, positive_threshold=0.7)

    assert list(ranking)[0] == 2


def test_popularity_ranking_orders_by_positive_count():
    df = make_df([
        (0, 10, 0.9), (1, 10, 0.9), (2, 10, 0.9),
        (0, 20, 0.9), (1, 20, 0.9),
        (0, 30, 0.9),
    ])

    assert list(popularity_ranking(df, 0.7)) == [10, 20, 30]


def test_popularity_ranking_excludes_items_with_no_engagement():
    df = make_df([(0, 10, 0.9), (0, 20, 0.2)])

    assert list(popularity_ranking(df, 0.7)) == [10]


def test_recommendations_are_identical_across_users_without_seen_filter():
    ranking = pd.array([1, 2, 3]).to_numpy()

    recs = popularity_recommendations(ranking, users=[0, 1], top_k=2)

    assert recs == [[1, 2], [1, 2]]


def test_recommendations_exclude_items_already_seen():
    ranking = pd.array([1, 2, 3, 4]).to_numpy()

    recs = popularity_recommendations(
        ranking, users=[0, 1], top_k=2, seen={0: {1, 2}}
    )

    assert recs[0] == [3, 4]     # user 0 already consumed 1 and 2
    assert recs[1] == [1, 2]


def test_recommendations_respect_top_k():
    ranking = pd.array([1, 2, 3, 4, 5]).to_numpy()

    recs = popularity_recommendations(ranking, users=[0], top_k=3)

    assert len(recs[0]) == 3
