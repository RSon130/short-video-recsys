import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from data.kuairand import long_view_rule, valid_play
from evaluation.gauc import bootstrap_mean, paired_bootstrap, per_user_auc, users_with_both_classes


def test_valid_play_and_long_view_rules_match_readme():
    play = np.array([5_000, 4_999, 7_001, 7_000, 17_000, 18_000])
    dur = np.array([5_000, 5_000, 60_000, 60_000, 17_000, 60_000])
    assert valid_play(play, dur).tolist() == [True, False, True, False, True, True]
    assert long_view_rule(play, dur).tolist() == [True, False, False, False, True, True]


def test_per_user_auc_matches_sklearn_and_skips_one_class_users():
    rng = np.random.default_rng(0)
    users = np.repeat([0, 1, 2], 50)
    labels = rng.random(150) < 0.3
    labels[100:] = False                      # user 2 has no positives
    scores = rng.integers(0, 5, 150).astype(float)   # many ties
    got = per_user_auc(users, labels, scores)
    assert list(got.index) == [0, 1]
    for u in (0, 1):
        m = users == u
        assert np.isclose(got[u], roc_auc_score(labels[m], scores[m]))
    assert users_with_both_classes(users, labels) == 2


def test_constant_score_gives_half():
    users = np.repeat([0, 1], 4)
    labels = np.array([1, 0, 0, 1, 0, 1, 1, 0])
    assert np.allclose(per_user_auc(users, labels, np.zeros(8)), 0.5)


def test_bootstraps_bracket_the_mean():
    rng = np.random.default_rng(1)
    a = pd.Series(rng.normal(0.5, 0.05, 500))
    b = a + 0.02
    r = paired_bootstrap(a, b, n_boot=500)
    assert np.isclose(r["diff"], 0.02) and r["ci_low"] - 1e-9 <= 0.02 <= r["ci_high"] + 1e-9 and r["p"] <= 0.01
    m = bootstrap_mean(a, n_boot=500)
    assert m["ci_low"] < m["mean"] < m["ci_high"]


def test_per_user_auc_rejects_nan_scores():
    import pytest
    with pytest.raises(ValueError):
        per_user_auc(np.array([0, 0]), np.array([1, 0]), np.array([np.nan, 1.0]))
