import numpy as np
import pandas as pd
import pytest
from evaluation.ab_test import assign_group, run_ab_test
from data.schema import Cols


@pytest.fixture
def ab_cfg():
    return {"evaluation": {"ab_test_traffic_split": 0.5}, "project": {"seed": 42}}


@pytest.fixture
def interactions_df():
    rows = []
    for uid in [1, 2, 3, 4]:
        for i in range(5):
            rows.append(
                {Cols.USER_ID: uid, Cols.WATCH_RATIO: 0.8 if i % 2 == 0 else 0.2}
            )
    return pd.DataFrame(rows)


def test_assign_group_deterministic():
    assert assign_group(42, 0.5, 42) == assign_group(42, 0.5, 42)


def test_assign_group_returns_a_or_b():
    for uid in range(20):
        assert assign_group(uid, 0.5, 42) in {"A", "B"}


def test_assign_group_seed_changes_assignment():
    assert any(
        assign_group(uid, 0.5, 42) != assign_group(uid, 0.5, 99) for uid in range(50)
    )


def test_assign_group_balance():
    groups = [assign_group(uid, 0.5, 42) for uid in range(1000)]
    count_a = groups.count("A")
    assert 450 <= count_a <= 550


def test_assign_group_all_b_at_zero_split():
    for uid in range(100):
        assert assign_group(uid, 0.0, 42) == "B"


def test_assign_group_all_a_at_full_split():
    for uid in range(100):
        assert assign_group(uid, 1.0, 42) == "A"


def test_run_ab_test_output_structure(ab_cfg, interactions_df):
    result = run_ab_test(
        interactions_df, np.ones(20) * 0.7, np.ones(20) * 0.3, ab_cfg
    )
    assert "group_A" in result
    assert "group_B" in result
    assert "n_users" in result["group_A"]
    assert "watch_time_auc" in result["group_A"]
    assert "n_users" in result["group_B"]
    assert "watch_time_auc" in result["group_B"]


def test_run_ab_test_n_users_partition(ab_cfg, interactions_df):
    result = run_ab_test(
        interactions_df, np.ones(20) * 0.7, np.ones(20) * 0.3, ab_cfg
    )
    assert result["group_A"]["n_users"] + result["group_B"]["n_users"] == 4


def test_run_ab_test_auc_in_range(ab_cfg, interactions_df):
    result = run_ab_test(
        interactions_df, np.ones(20) * 0.7, np.ones(20) * 0.3, ab_cfg
    )
    assert 0 <= result["group_A"]["watch_time_auc"] <= 1
    assert 0 <= result["group_B"]["watch_time_auc"] <= 1
