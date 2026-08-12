import pytest

from data import get_loader
from data.kuairec import KuaiRecLoader


def test_get_loader_kuairec_returns_kuairec_loader(tmp_path):
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
    assert isinstance(get_loader(cfg), KuaiRecLoader)


def test_get_loader_unknown_source_raises_value_error():
    with pytest.raises(ValueError) as exc:
        get_loader({"data": {"source": "unknown_source"}})
    assert "unknown_source" in str(exc.value)


def test_get_loader_error_message_lists_valid_sources():
    with pytest.raises(ValueError) as exc:
        get_loader({"data": {"source": "unknown_source"}})
    assert "kuairec" in str(exc.value)
