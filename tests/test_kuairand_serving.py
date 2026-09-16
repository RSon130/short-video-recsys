import json

import numpy as np
import pandas as pd
import pytest

from serving.kuairand_scorer import KuaiRandScorer


@pytest.fixture
def scorer(tmp_path):
    """A four-item catalogue: user 7 has seen items 10 and 30."""
    pd.DataFrame({"item_id": [10, 20, 30, 40], "impressions": [100, 50, 20, 20],
                  "duration_s": [5.0, 10.0, np.nan, 7.0], "rank": [1, 2, 3, 4]}
                 ).to_parquet(tmp_path / "kuairand_items.parquet", index=False)
    np.savez_compressed(tmp_path / "kuairand_seen.npz",
                        users=np.array([3, 7], np.int32),
                        offsets=np.array([0, 1, 3], np.int64),
                        items=np.array([20, 10, 30], np.int32))
    (tmp_path / "kuairand_meta.json").write_text(json.dumps({"scorer": "item_impressions",
                                                             "why_this_scorer": "baseline won"}))
    return KuaiRandScorer(tmp_path)


def test_ranks_by_impressions_with_stable_tie_break(scorer):
    items, excluded = scorer.top_k(user_id=999, k=4, exclude_seen=False)
    assert [i for i, _ in items] == [10, 20, 30, 40] and excluded == 0


def test_excludes_items_the_user_has_already_seen(scorer):
    items, excluded = scorer.top_k(user_id=7, k=4)
    assert [i for i, _ in items] == [20, 40] and excluded == 2


def test_unknown_user_gets_the_full_catalogue(scorer):
    assert len(scorer.seen_items(12345)) == 0
    items, excluded = scorer.top_k(user_id=12345, k=3)
    assert [i for i, _ in items] == [10, 20, 30] and excluded == 0


def test_top_k_is_capped_by_the_catalogue(scorer):
    items, _ = scorer.top_k(user_id=7, k=50)
    assert len(items) == 2 and scorer.catalogue_size == 4
