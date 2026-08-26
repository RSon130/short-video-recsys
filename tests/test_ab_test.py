import numpy as np
import pytest

from evaluation.ab_test import (
    assign_group,
    check_sample_ratio,
    minimum_detectable_effect,
    paired_bootstrap,
)


# ---------------------------------------------------------------- bucketing


def test_assignment_is_stable_for_the_same_user():
    """A user must not flip variants between calls, restarts, or deploys."""
    assert all(assign_group(123) == assign_group(123) for _ in range(50))


def test_assignment_changes_with_the_experiment_seed():
    """A new seed starts an independent experiment over the same population."""
    a = [assign_group(u, seed=1) for u in range(500)]
    b = [assign_group(u, seed=2) for u in range(500)]
    assert a != b


def test_split_is_approximately_balanced():
    groups = [assign_group(u) for u in range(5000)]
    share_a = sum(1 for g in groups if g == "A") / len(groups)
    assert 0.47 < share_a < 0.53


def test_traffic_split_is_respected():
    groups = [assign_group(u, traffic_split=0.2) for u in range(5000)]
    share_a = sum(1 for g in groups if g == "A") / len(groups)
    assert 0.17 < share_a < 0.23


def test_sample_ratio_check_passes_on_a_healthy_split():
    result = check_sample_ratio(range(5000))
    assert not result["srm_suspected"]
    assert abs(result["z"]) < 3
    assert result["n_a"] + result["n_b"] == 5000


def test_sample_ratio_check_flags_a_skewed_population():
    """
    Feeding it a population that is entirely group A is what a broken
    assignment looks like from the outside: observed share far from intended.
    """
    only_a = [u for u in range(5000) if assign_group(u) == "A"]

    assert check_sample_ratio(only_a, traffic_split=0.5)["srm_suspected"]


# ---------------------------------------------------------- paired bootstrap


def test_detects_a_real_and_consistent_improvement():
    rng = np.random.default_rng(0)
    a = rng.random(500)
    b = a + 0.05                      # same users, uniformly better

    result = paired_bootstrap(a, b)

    assert result["significant"]
    assert result["ci_low"] > 0
    assert result["mean_diff"] == pytest.approx(0.05, abs=1e-6)


def test_reports_no_significance_when_systems_are_equivalent():
    rng = np.random.default_rng(0)
    a = rng.random(500)
    b = rng.random(500)

    result = paired_bootstrap(a, b)

    assert not result["significant"]
    assert result["ci_low"] < 0 < result["ci_high"]


def test_tiny_sample_with_a_small_effect_is_not_significant():
    """
    The incident this guards: a 150-user sample showed +0.7% and the full set
    reversed it. A CI on the small sample must decline to call it.
    """
    rng = np.random.default_rng(7)
    a = rng.random(150)
    b = a + rng.normal(0.001, 0.3, 150)

    assert not paired_bootstrap(a, b)["significant"]


def test_pairing_beats_pooling_on_the_same_data():
    """
    Between-user variance dwarfs the effect. Differencing within user removes
    it, which is the whole reason to pair.
    """
    rng = np.random.default_rng(3)
    user_skill = rng.normal(0.5, 0.3, 400)     # huge between-user spread
    a = user_skill
    b = user_skill + 0.02                      # small consistent lift

    paired = paired_bootstrap(a, b)
    pooled_gap = abs(b.mean() - a.mean())
    pooled_noise = np.sqrt(a.var() / len(a) + b.var() / len(b)) * 1.96

    assert paired["significant"]
    assert pooled_gap < pooled_noise           # an unpaired test would miss it


def test_detects_a_regression_as_significant():
    a = np.full(300, 0.5)
    b = np.full(300, 0.4)

    result = paired_bootstrap(a, b)

    assert result["significant"]
    assert result["ci_high"] < 0
    assert result["relative_lift"] < 0


def test_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="equal shapes"):
        paired_bootstrap(np.zeros(10), np.zeros(11))


def test_rejects_empty_input():
    with pytest.raises(ValueError, match="no observations"):
        paired_bootstrap(np.array([]), np.array([]))


# ------------------------------------------------------------------- power


def test_more_users_detect_smaller_effects():
    rng = np.random.default_rng(0)
    values = rng.random(1000)

    small = minimum_detectable_effect(values, n_users=100)
    large = minimum_detectable_effect(values, n_users=10_000)

    assert large["mde_absolute"] < small["mde_absolute"]


def test_noisier_metrics_need_more_users():
    quiet = minimum_detectable_effect(np.random.default_rng(0).normal(0.5, 0.01, 1000))
    noisy = minimum_detectable_effect(np.random.default_rng(0).normal(0.5, 0.30, 1000))

    assert noisy["users_for_5pct_lift"] > quiet["users_for_5pct_lift"]


def test_smaller_target_lift_needs_more_users():
    values = np.random.default_rng(0).random(1000)
    result = minimum_detectable_effect(values)

    assert result["users_for_5pct_lift"] > result["users_for_10pct_lift"]
