import pickle
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from features.engineer import load_config, save_parquet


def test_load_config_returns_dict_with_base_keys():
    result = load_config("config/kuairec.yaml")
    assert isinstance(result, dict)
    for key in ["project", "data", "features", "two_tower", "ranking", "training"]:
        assert key in result
    assert result["project"]["seed"] == 42


def test_load_config_override_merges():
    cfg = load_config("config/kuairec.yaml")
    assert cfg["data"]["source"] == "kuairec"
    assert cfg["features"]["user_emb_dim"] == 64
    assert "kuairec" in cfg["data"]


def test_save_parquet_creates_file(tmp_path):
    dest = tmp_path / "sub" / "out.parquet"
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    save_parquet(df, dest)
    assert dest.exists()
    loaded = pd.read_parquet(dest)
    assert len(loaded) == 2
    assert list(loaded.columns) == ["a", "b"]


def test_save_parquet_creates_nested_dirs(tmp_path):
    path = tmp_path / "a" / "b" / "c.parquet"
    save_parquet(pd.DataFrame({"x": [1]}), path)
    assert path.exists()


def test_run_calls_save_parquet_five_times(tmp_path):
    mock_fs = MagicMock()
    mock_fs.interactions = pd.DataFrame(
        {"user_id": [0], "item_id": [0], "timestamp": [0], "watch_ratio": [0.5]}
    )
    mock_fs.user_features = pd.DataFrame({"user_id": [0]})
    mock_fs.item_features = pd.DataFrame({"item_id": [0]})
    mock_fs.user_id_map = {0: 0}
    mock_fs.item_id_map = {0: 0}
    mock_fs.n_users = 1
    mock_fs.n_items = 1

    mock_split = MagicMock()
    mock_split.train = pd.DataFrame(
        {"user_id": [0], "item_id": [0], "timestamp": [0], "watch_ratio": [0.5]}
    )
    mock_split.val = pd.DataFrame(
        {"user_id": [0], "item_id": [0], "timestamp": [0], "watch_ratio": [0.5]}
    )
    mock_split.test = pd.DataFrame(
        {"user_id": [0], "item_id": [0], "timestamp": [0], "watch_ratio": [0.5]}
    )

    mock_loader = MagicMock()
    mock_loader.build_feature_store.return_value = mock_fs
    mock_loader.temporal_split.return_value = mock_split

    cfg = {
        "data": {
            "source": "kuairec",
            "train_ratio": 0.8,
            "val_ratio": 0.1,
            "kuairec": {
                "raw_dir": str(tmp_path),
                "interaction_file": "x.csv",
                "clip_watch_ratio": True,
                "min_interactions_per_user": 1,
                "min_interactions_per_item": 1,
                "column_map": {},
            },
        },
    }

    with patch("features.engineer.get_loader", return_value=mock_loader), \
         patch("features.engineer.save_parquet") as mock_save_parquet, \
         patch("features.engineer.pickle.dump") as mock_pickle_dump:
        from features.engineer import run
        run(cfg)

    assert mock_save_parquet.call_count == 5
    assert mock_pickle_dump.call_count == 1
