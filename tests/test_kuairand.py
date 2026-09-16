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


def test_ips_weights_are_normalised_clipped_and_favour_rare_items():
    from scripts.exposure_bias_kuairand import ips_weights
    hist = pd.DataFrame({"video_id": [1] * 500 + [2] * 50 + [3] * 5})
    rows = pd.DataFrame({"video_id": [1, 2, 3, 99]})
    w, clipped = ips_weights(hist, rows)
    assert np.isfinite(w).all() and np.isclose(w.mean(), 1.0)
    assert w[0] < w[1] < w[2]                      # rarer item, larger weight
    assert w[3] >= w[2]                            # unseen item gets the largest
    assert clipped.dtype == bool and len(clipped) == len(w)


def test_band_demean_removes_within_band_level():
    from scripts.exposure_bias_kuairand import band_demean
    out = band_demean(np.array([1.0, 3.0, 10.0, 12.0]), np.array([0, 0, 1, 1]))
    assert np.allclose(out, [-1, 1, -1, 1])


def test_duration_terciles_put_unknown_duration_in_its_own_group():
    from scripts.exposure_bias_kuairand import duration_terciles
    t = duration_terciles(pd.Series([5.0, 50.0, 500.0, np.nan, 6.0, 60.0]))
    assert t[3] == -1 and sorted(set(t.tolist())) == [-1, 0, 1, 2]


def test_a2_drops_exactly_the_pre_registered_exposure_features():
    from scripts.exposure_bias_kuairand import A2_DROP
    assert A2_DROP == ("item_impr_share", "author_impr_share", "utag_share",
                       "uauth_share", "uband_share", "user_single_col_share")
