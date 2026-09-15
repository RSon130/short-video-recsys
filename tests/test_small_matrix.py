import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from data.kuairec import KuaiRecLoader
from data.schema import Cols
from evaluation.small_matrix import (Baselines, DurationModel, auc, bootstrap_diff, holm,
                                     ndcg_at, split_users, top_candidates, two_stage_scores,
                                     within_user_top)


def test_exact_duplicates_are_dropped_but_rewatches_kept():
    df = pd.DataFrame({Cols.USER_ID: [1, 1, 1], Cols.ITEM_ID: [5, 5, 5],
                       "timestamp": [10.0, 10.0, 99.0], Cols.WATCH_RATIO: [0.5, 0.5, 0.9]})
    out = KuaiRecLoader.drop_exact_duplicates(df)
    assert len(out) == 2 and sorted(out["timestamp"]) == [10.0, 99.0]


def _toy_train():
    # Two items per duration level; short items have inflated watch_ratio.
    rng = np.random.default_rng(0)
    rows = []
    for item in range(100):
        dur = 1000 * (1 + item // 2)
        for user in range(30):
            rows.append((user, item, rng.random() * (3.0 if dur < 20000 else 1.0), dur))
    return pd.DataFrame(rows, columns=[Cols.USER_ID, Cols.ITEM_ID, Cols.WATCH_RATIO, "video_duration"])


def test_percentile_is_mid_rank_within_bucket():
    train = _toy_train()
    model = DurationModel.fit(train, n_items=100)
    b = model.bucket(np.array([0]))[0]
    ref = model.sorted_wr[b]
    p = model.percentile(np.array([0, 0]), np.array([ref.min() - 1, ref.max() + 1]))
    assert p[0] == 0.0 and p[1] == 1.0
    # ties sit at the middle of the tied block
    mid = model.percentile(np.array([0]), np.array([ref[len(ref) // 2]]))[0]
    assert 0.0 < mid < 1.0


def test_percentile_label_removes_duration_from_positive_rates():
    train = _toy_train()
    model = DurationModel.fit(train, n_items=100)
    p = model.percentile(train[Cols.ITEM_ID].to_numpy(), train[Cols.WATCH_RATIO].to_numpy())
    rel = within_user_top(train[Cols.USER_ID].to_numpy(), p, 0.3)
    short = train["video_duration"].to_numpy() < 20000
    assert abs(rel[short].mean() - rel[~short].mean()) < 0.08
    raw = (train[Cols.WATCH_RATIO] >= 0.7).to_numpy()
    assert raw[short].mean() - raw[~short].mean() > 0.3   # the raw label is confounded


def test_within_user_top_fraction_per_user():
    users = np.repeat([0, 1], 100)
    values = np.concatenate([np.arange(100), np.arange(100) * 10.0])
    rel = within_user_top(users, values, 0.3)
    assert rel[:100].sum() == 30 and rel[100:].sum() == 30


def test_auc_matches_sklearn_with_ties():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    s = rng.integers(0, 5, 500).astype(float)
    assert auc(y, s) == pytest.approx(roc_auc_score(y, s))
    assert np.isnan(auc(np.ones(5), np.arange(5.0)))


def test_ndcg_perfect_and_empty():
    assert ndcg_at(np.array([1, 1, 0, 0]), 2) == pytest.approx(1.0)
    assert ndcg_at(np.array([0, 0, 1, 1]), 2) < 1.0
    assert np.isnan(ndcg_at(np.zeros(4), 0))


def test_two_stage_puts_candidates_first_in_ranker_order():
    ret = np.array([0.9, 0.1, 0.8, 0.5, 0.7])
    cand = top_candidates(ret, 3)                     # items 0, 2, 4
    assert set(cand.tolist()) == {0, 2, 4}
    rk = np.array([0.1, 0.9, 0.5])[np.argsort(np.argsort(cand))]  # arbitrary ranker scores per cand
    s = two_stage_scores(ret, rk, cand)
    order = np.argsort(-s)
    assert set(order[:3].tolist()) == {0, 2, 4}
    assert order[3:].tolist() == [3, 1]               # rest keep retrieval order
    assert order[0] == cand[np.argmax(rk)]


def test_holm_step_down():
    # thresholds 0.05/3, 0.05/2, 0.05/1
    assert holm({"a": 0.001, "b": 0.02, "c": 0.04}) == {"a": True, "b": True, "c": True}
    assert holm({"a": 0.001, "b": 0.03, "c": 0.04}) == {"a": True, "b": False, "c": False}
    assert holm({"a": 0.06, "b": 0.001}) == {"b": True, "a": False}


def test_bootstrap_detects_consistent_difference():
    a = np.full(200, 0.5)
    b = a + 0.01 + np.random.default_rng(0).normal(0, 0.001, 200)
    r = bootstrap_diff(a, b, n_boot=2000)
    assert r["ci_low"] > 0 and r["p"] < 0.01
    r0 = bootstrap_diff(a, a + np.random.default_rng(1).normal(0, 0.01, 200), n_boot=2000)
    assert r0["p"] > 0.01


def test_user_split_is_deterministic_and_disjoint():
    v1, t1 = split_users(range(1000), seed=42)
    v2, t2 = split_users(range(1000), seed=42)
    assert np.array_equal(v1, v2) and not set(v1) & set(t1)
    assert 0.25 < len(v1) / 1000 < 0.35


def test_item_knn_scores_items_similar_to_what_the_user_liked():
    # Users 0-9 like items 0,1 and dislike 2,3; item 4 co-varies with 0/1.
    rows = []
    for u in range(10):
        for i, wr in ((0, 2.0), (1, 2.0), (2, 0.1), (3, 0.1), (4, 2.0 if u < 8 else 0.1)):
            rows.append((u, i, wr, 5000))
    rows += [(10, 0, 2.0, 5000), (10, 2, 0.1, 5000)]          # eval user: liked 0, disliked 2
    train = pd.DataFrame(rows, columns=[Cols.USER_ID, Cols.ITEM_ID, Cols.WATCH_RATIO, "video_duration"])
    model = DurationModel.fit(train, n_items=5)
    base = Baselines(train, model, n_users=11, n_items=5, item_play_progress=np.full(5, np.nan),
                     eval_users=np.array([10]), target_items=np.array([1, 3, 4]))
    s = base.scores("B4 item-kNN", 10, np.array([1, 3, 4]))
    assert s[0] > s[1] and s[2] > s[1]
