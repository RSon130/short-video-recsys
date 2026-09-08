import pytest
import pandas as pd

from data.kuairec import KuaiRecLoader


def make_cfg(tmp_path, **overrides):
    cfg = {
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
    cfg["data"]["kuairec"].update(overrides)
    return cfg


def _make_3x3_rows(watch_ratio=0.5):
    rows = []
    for uid in [1, 2, 3]:
        for iid in [10, 11, 12]:
            rows.append(
                {
                    "video_id": iid,
                    "user_id": uid,
                    "watch_ratio": watch_ratio,
                    "like": 0,
                    "comment": 0,
                    "share": 0,
                    "follow": 0,
                }
            )
    return rows


def test_load_interactions_renames_video_id(tmp_path):
    rows = []
    for uid in [1, 2, 3, 4]:
        for iid in [10, 11, 12]:
            rows.append(
                {
                    "video_id": iid,
                    "user_id": uid,
                    "watch_ratio": 0.5,
                    "like": 0,
                    "comment": 0,
                    "share": 0,
                    "follow": 0,
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(tmp_path / "small_matrix.csv", index=False)
    loader = KuaiRecLoader(make_cfg(tmp_path))
    result = loader.load_interactions()
    assert "item_id" in result.columns
    assert "video_id" not in result.columns


def test_load_interactions_clips_watch_ratio_above_1(tmp_path):
    rows = _make_3x3_rows()
    df = pd.DataFrame(rows)
    df.loc[0, "watch_ratio"] = 1.8
    df.to_csv(tmp_path / "small_matrix.csv", index=False)
    loader = KuaiRecLoader(make_cfg(tmp_path))
    result = loader.load_interactions()
    assert (result["watch_ratio"] <= 1.0).all()


def test_load_interactions_no_clip_when_disabled(tmp_path):
    rows = _make_3x3_rows()
    df = pd.DataFrame(rows)
    df.loc[0, "watch_ratio"] = 1.8
    df.to_csv(tmp_path / "small_matrix.csv", index=False)
    loader = KuaiRecLoader(make_cfg(tmp_path, clip_watch_ratio=False))
    result = loader.load_interactions()
    assert (result["watch_ratio"] > 1.0).any()


def test_load_interactions_timestamp_from_date_column(tmp_path):
    rows = []
    for uid in [1, 2, 3]:
        for iid in [10, 11, 12]:
            rows.append(
                {
                    "video_id": iid,
                    "user_id": uid,
                    "watch_ratio": 0.5,
                    "date": 20210101,
                    "like": 0,
                    "comment": 0,
                    "share": 0,
                    "follow": 0,
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(tmp_path / "small_matrix.csv", index=False)
    loader = KuaiRecLoader(make_cfg(tmp_path))
    result = loader.load_interactions()
    assert (result["timestamp"] > 0).all()


def test_load_interactions_timestamp_zero_when_no_date(tmp_path):
    df = pd.DataFrame(_make_3x3_rows())
    df.to_csv(tmp_path / "small_matrix.csv", index=False)
    loader = KuaiRecLoader(make_cfg(tmp_path))
    result = loader.load_interactions()
    assert (result["timestamp"] == 0).all()


def test_load_interactions_missing_watch_ratio_raises(tmp_path):
    df = pd.DataFrame({"video_id": [10], "user_id": [1], "like": [0]})
    df.to_csv(tmp_path / "small_matrix.csv", index=False)
    loader = KuaiRecLoader(make_cfg(tmp_path))
    with pytest.raises(ValueError):
        loader.load_interactions()


def test_load_interactions_file_not_found_raises(tmp_path):
    loader = KuaiRecLoader(make_cfg(tmp_path))
    with pytest.raises(FileNotFoundError):
        loader.load_interactions()


def test_load_interactions_cold_start_removes_sparse_users(tmp_path):
    rows = _make_3x3_rows()
    # user 99 has only 1 interaction → sparse
    rows.append(
        {
            "video_id": 10,
            "user_id": 99,
            "watch_ratio": 0.5,
            "like": 0,
            "comment": 0,
            "share": 0,
            "follow": 0,
        }
    )
    df = pd.DataFrame(rows)
    df.to_csv(tmp_path / "small_matrix.csv", index=False)
    loader = KuaiRecLoader(make_cfg(tmp_path))
    result = loader.load_interactions()
    assert 99 not in result["user_id"].values


def test_load_interactions_cold_start_cascade(tmp_path):
    # user 1: items 10, 11 (2 interactions — at threshold)
    # user 2: items 11, 12 (2 interactions — at threshold)
    # user 3: items 10, 11 (2 interactions — stabilises items 10 and 11)
    # user 99: item 12 only (sparse — triggers cascade)
    # Cascade: remove user 99 → item 12 becomes sparse → remove item 12
    #          → user 2 now has 1 interaction → remove user 2
    rows = [
        {"video_id": 10, "user_id": 1, "watch_ratio": 0.5, "like": 0, "comment": 0, "share": 0, "follow": 0},
        {"video_id": 11, "user_id": 1, "watch_ratio": 0.5, "like": 0, "comment": 0, "share": 0, "follow": 0},
        {"video_id": 11, "user_id": 2, "watch_ratio": 0.5, "like": 0, "comment": 0, "share": 0, "follow": 0},
        {"video_id": 12, "user_id": 2, "watch_ratio": 0.5, "like": 0, "comment": 0, "share": 0, "follow": 0},
        {"video_id": 10, "user_id": 3, "watch_ratio": 0.5, "like": 0, "comment": 0, "share": 0, "follow": 0},
        {"video_id": 11, "user_id": 3, "watch_ratio": 0.5, "like": 0, "comment": 0, "share": 0, "follow": 0},
        {"video_id": 12, "user_id": 99, "watch_ratio": 0.5, "like": 0, "comment": 0, "share": 0, "follow": 0},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(tmp_path / "small_matrix.csv", index=False)
    loader = KuaiRecLoader(make_cfg(tmp_path))
    result = loader.load_interactions()
    assert 99 not in result["user_id"].values
    assert 2 not in result["user_id"].values


def test_load_interactions_required_cols_present(tmp_path):
    df = pd.DataFrame(_make_3x3_rows())
    df.to_csv(tmp_path / "small_matrix.csv", index=False)
    loader = KuaiRecLoader(make_cfg(tmp_path))
    result = loader.load_interactions()
    for col in ["user_id", "item_id", "timestamp", "watch_ratio"]:
        assert col in result.columns


def test_load_user_features_stub_when_file_missing(tmp_path):
    loader = KuaiRecLoader(make_cfg(tmp_path))
    result = loader.load_user_features()
    assert "user_id" in result.columns
    assert len(result) == 0


def test_load_user_features_encodes_object_columns(tmp_path):
    df = pd.DataFrame(
        {"user_id": [1, 2, 3], "age_group": ["young", "middle", "old"]}
    )
    df.to_csv(tmp_path / "user_features.csv", index=False)
    loader = KuaiRecLoader(make_cfg(tmp_path))
    result = loader.load_user_features()
    assert pd.api.types.is_integer_dtype(result["age_group"])


def test_load_user_features_fillna_zero(tmp_path):
    df = pd.DataFrame({"user_id": [1, 2], "score": [1.0, None]})
    df.to_csv(tmp_path / "user_features.csv", index=False)
    loader = KuaiRecLoader(make_cfg(tmp_path))
    result = loader.load_user_features()
    assert not result.isnull().any().any()


def test_load_item_features_file_not_found_raises(tmp_path):
    loader = KuaiRecLoader(make_cfg(tmp_path))
    with pytest.raises(FileNotFoundError):
        loader.load_item_features()


def test_load_item_features_renames_video_id(tmp_path):
    df = pd.DataFrame({"video_id": [10, 11], "category_id": [1, 2]})
    df.to_csv(tmp_path / "item_categories.csv", index=False)
    loader = KuaiRecLoader(make_cfg(tmp_path))
    result = loader.load_item_features()
    assert "item_id" in result.columns
    assert "video_id" not in result.columns


def test_expand_category_feat_creates_cat_columns(tmp_path):
    df = pd.DataFrame({"video_id": [10, 11], "feat": ["[3, 17]", "[5]"]})
    df.to_csv(tmp_path / "item_categories.csv", index=False)
    loader = KuaiRecLoader(make_cfg(tmp_path))
    result = loader.load_item_features()
    row = result[result["item_id"] == 10].iloc[0]
    assert row["cat_3"] == 1
    assert row["cat_17"] == 1
    for col in result.columns:
        if col.startswith("cat_") and col not in ("cat_3", "cat_17"):
            assert row[col] == 0


def test_expand_category_feat_drops_original_feat_column(tmp_path):
    df = pd.DataFrame({"video_id": [10, 11], "feat": ["[3, 17]", "[5]"]})
    df.to_csv(tmp_path / "item_categories.csv", index=False)
    loader = KuaiRecLoader(make_cfg(tmp_path))
    result = loader.load_item_features()
    assert "feat" not in result.columns


def test_load_item_features_merges_daily_features(tmp_path):
    cat_df = pd.DataFrame({"video_id": [10, 11], "category_id": [1, 2]})
    cat_df.to_csv(tmp_path / "item_categories.csv", index=False)
    daily_df = pd.DataFrame({"video_id": [10, 11], "play_cnt": [100, 200]})
    daily_df.to_csv(tmp_path / "item_daily_features.csv", index=False)
    loader = KuaiRecLoader(make_cfg(tmp_path))
    result = loader.load_item_features()
    assert "play_cnt" in result.columns


def test_load_item_features_mean_aggregation_of_daily(tmp_path):
    cat_df = pd.DataFrame({"video_id": [10], "category_id": [1]})
    cat_df.to_csv(tmp_path / "item_categories.csv", index=False)
    daily_df = pd.DataFrame({"video_id": [10, 10], "play_cnt": [10, 20]})
    daily_df.to_csv(tmp_path / "item_daily_features.csv", index=False)
    loader = KuaiRecLoader(make_cfg(tmp_path))
    result = loader.load_item_features()
    assert result.loc[result["item_id"] == 10, "play_cnt"].iloc[0] == 15.0


def test_absent_engagement_columns_are_not_zero_filled(tmp_path):
    """
    KuaiRec's matrices carry no per-interaction like/comment/share/follow.
    Defaulting them to 0 created four all-zero columns that flowed through the
    whole pipeline — training on one would have silently learned nothing.
    Absent must stay absent so a consumer gets KeyError instead.
    """
    raw = tmp_path / "kuairec"
    raw.mkdir()
    pd.DataFrame({
        "user_id": [0, 0, 1, 1],
        "video_id": [0, 1, 0, 1],
        "watch_ratio": [0.9, 0.1, 0.8, 0.2],
        "timestamp": [1, 2, 3, 4],
    }).to_csv(raw / "small_matrix.csv", index=False)
    pd.DataFrame({"video_id": [0, 1], "feat": ["[1]", "[2]"]}).to_csv(
        raw / "item_categories.csv", index=False)

    cfg = {"data": {"source": "kuairec", "kuairec": {
        "raw_dir": str(raw), "interaction_file": "small_matrix.csv",
        "clip_watch_ratio": True, "min_interactions_per_user": 1,
        "min_interactions_per_item": 1, "column_map": {}}}}

    from data.kuairec import KuaiRecLoader
    df = KuaiRecLoader(cfg).load_interactions()

    for absent in ("like", "comment", "share", "follow"):
        assert absent not in df.columns, (
            f"{absent} was fabricated; it does not exist in KuaiRec matrices"
        )
    assert "watch_ratio" in df.columns
