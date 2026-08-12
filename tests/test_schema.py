import pytest
import pandas as pd
from pydantic import ValidationError

from data.schema import (
    Cols,
    RecommendRequest,
    ItemScore,
    RecommendResponse,
    InteractionEvent,
    DataSplit,
    FeatureStore,
)


def test_cols_constants():
    assert Cols.USER_ID == "user_id"
    assert Cols.ITEM_ID == "item_id"
    assert Cols.WATCH_RATIO == "watch_ratio"
    assert Cols.LIKE == "like"


def test_recommend_request_defaults():
    r = RecommendRequest(user_id=1)
    assert r.top_k == 20
    assert r.strategy == "full"
    assert r.session_items == []


def test_recommend_request_invalid_strategy():
    with pytest.raises(ValidationError):
        RecommendRequest(user_id=1, strategy="invalid")


@pytest.mark.parametrize("strategy", ["full", "recall_only", "rank_only"])
def test_recommend_request_strategy_valid_values(strategy):
    RecommendRequest(user_id=1, strategy=strategy)


def test_recommend_request_session_too_long():
    with pytest.raises(ValidationError):
        RecommendRequest(user_id=1, session_items=list(range(51)))


def test_recommend_request_top_k_bounds():
    with pytest.raises(ValidationError):
        RecommendRequest(user_id=1, top_k=0)
    with pytest.raises(ValidationError):
        RecommendRequest(user_id=1, top_k=101)
    RecommendRequest(user_id=1, top_k=100)


def test_interaction_event_clip_above_1():
    # In Pydantic v2, Field(ge=0.0, le=1.0) enforces constraints before the
    # @field_validator runs (mode='after'), so out-of-range values raise ValidationError.
    with pytest.raises(ValidationError):
        InteractionEvent(user_id=1, item_id=1, watch_ratio=1.5)


def test_interaction_event_clip_below_0():
    with pytest.raises(ValidationError):
        InteractionEvent(user_id=1, item_id=1, watch_ratio=-0.5)


def test_interaction_event_valid_watch_ratio():
    assert InteractionEvent(user_id=1, item_id=1, watch_ratio=0.75).watch_ratio == 0.75


def test_interaction_event_binary_fields():
    with pytest.raises(ValidationError):
        InteractionEvent(user_id=1, item_id=1, watch_ratio=0.5, like=2)
    with pytest.raises(ValidationError):
        InteractionEvent(user_id=1, item_id=1, watch_ratio=0.5, like=-1)


def test_data_split_sizes():
    train = pd.DataFrame(range(80))
    val = pd.DataFrame(range(10))
    test = pd.DataFrame(range(10))
    assert DataSplit(train=train, val=val, test=test).sizes == {
        "train": 80,
        "val": 10,
        "test": 10,
    }


def test_feature_store_defaults():
    df = pd.DataFrame()
    fs = FeatureStore(interactions=df, user_features=df, item_features=df)
    assert fs.n_users == 0
    assert fs.n_items == 0
    assert fs.user_id_map == {}
    assert fs.item_id_map == {}
