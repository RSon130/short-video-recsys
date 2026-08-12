import pytest
import pandas as pd


@pytest.fixture
def minimal_cfg(tmp_path):
    return {
        "data": {
            "source": "kuairec",
            "kuairec": {
                "raw_dir": str(tmp_path),
                "interaction_file": "small_matrix.csv",
                "clip_watch_ratio": True,
                "min_interactions_per_user": 2,
                "min_interactions_per_item": 2,
                "column_map": {},
            },
        }
    }


@pytest.fixture
def interaction_df():
    data = []
    ts = 0
    for user_id in [1, 2, 3, 4]:
        for item_id in [10, 11, 12, 13]:
            data.append(
                {
                    "user_id": user_id,
                    "item_id": item_id,
                    "timestamp": ts,
                    "watch_ratio": 0.5,
                    "like": 0,
                    "comment": 0,
                    "share": 0,
                    "follow": 0,
                }
            )
            ts += 1
    return pd.DataFrame(data)


@pytest.fixture
def user_feature_df():
    return pd.DataFrame({"user_id": [1, 2, 3, 4], "age": [20, 25, 30, 35]})


@pytest.fixture
def item_feature_df():
    return pd.DataFrame(
        {"item_id": [10, 11, 12, 13], "category_id": [1, 2, 3, 4]}
    )
