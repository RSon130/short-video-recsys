import pytest
import pandas as pd

from data.base import BaseDataLoader
from data.schema import Cols


class MinimalLoader(BaseDataLoader):
    def __init__(self, interactions, user_features, item_features):
        super().__init__(cfg={})
        self._i = interactions
        self._u = user_features
        self._it = item_features

    def load_interactions(self):
        return self._i.copy()

    def load_user_features(self):
        return self._u.copy()

    def load_item_features(self):
        return self._it.copy()


def test_rename_maps_present_columns():
    df = pd.DataFrame({"video_id": [1, 2]})
    result = BaseDataLoader._rename(df, {"video_id": "item_id"})
    assert "item_id" in result.columns
    assert "video_id" not in result.columns


def test_rename_ignores_missing_columns():
    df = pd.DataFrame({"a": [1, 2]})
    result = BaseDataLoader._rename(df, {"x": "y"})
    assert list(result.columns) == ["a"]


def test_rename_leaves_other_columns_intact():
    df = pd.DataFrame({"a": [1], "b": [2]})
    result = BaseDataLoader._rename(df, {"a": "c"})
    assert set(result.columns) == {"c", "b"}


def test_require_cols_passes_with_all_present():
    df = pd.DataFrame({"a": [1], "b": [2]})
    BaseDataLoader._require_cols(df, ["a", "b"], "test")


def test_require_cols_raises_on_missing():
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError) as exc:
        BaseDataLoader._require_cols(df, ["a", "b"], "src")
    assert "b" in str(exc.value)


def test_temporal_split_sizes():
    df = pd.DataFrame({"timestamp": range(100), "val": range(100)})
    loader = MinimalLoader(df, df, df)
    split = loader.temporal_split(df, train_ratio=0.8, val_ratio=0.1)
    assert len(split.train) == 80
    assert len(split.val) == 10
    assert len(split.test) == 10


def test_temporal_split_sorted_by_timestamp():
    df = pd.DataFrame({"timestamp": range(100), "val": range(100)})
    loader = MinimalLoader(df, df, df)
    split = loader.temporal_split(df, train_ratio=0.8, val_ratio=0.1)
    assert split.train["timestamp"].max() < split.val["timestamp"].min()
    assert split.val["timestamp"].max() < split.test["timestamp"].min()


def test_temporal_split_no_index_overlap():
    df = pd.DataFrame({"timestamp": range(100), "val": range(100)})
    loader = MinimalLoader(df, df, df)
    split = loader.temporal_split(df, train_ratio=0.8, val_ratio=0.1)
    combined = pd.concat([split.train, split.val, split.test])
    assert combined.index.duplicated().sum() == 0


def test_build_feature_store_ids_are_contiguous(
    interaction_df, user_feature_df, item_feature_df
):
    loader = MinimalLoader(interaction_df, user_feature_df, item_feature_df)
    fs = loader.build_feature_store()
    assert sorted(fs.interactions["user_id"].unique()) == list(range(fs.n_users))
    assert sorted(fs.interactions["item_id"].unique()) == list(range(fs.n_items))


def test_build_feature_store_id_map_keys_are_originals(
    interaction_df, user_feature_df, item_feature_df
):
    loader = MinimalLoader(interaction_df, user_feature_df, item_feature_df)
    fs = loader.build_feature_store()
    assert set(fs.user_id_map.keys()) == {1, 2, 3, 4}


def test_build_feature_store_drops_features_outside_interactions(
    interaction_df, user_feature_df, item_feature_df
):
    extra_row = pd.DataFrame({"user_id": [99], "age": [99]})
    extended_user_features = pd.concat(
        [user_feature_df, extra_row], ignore_index=True
    )
    loader = MinimalLoader(interaction_df, extended_user_features, item_feature_df)
    fs = loader.build_feature_store()
    assert 99 not in fs.user_features["user_id"].values


def test_build_feature_store_n_users_n_items(
    interaction_df, user_feature_df, item_feature_df
):
    loader = MinimalLoader(interaction_df, user_feature_df, item_feature_df)
    fs = loader.build_feature_store()
    assert fs.n_users == 4
    assert fs.n_items == 4
