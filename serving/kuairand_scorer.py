"""
The KuaiRand scorer that phase 2 deploys: item impression count.

It is non-personal by result, not by accident. Under the pre-registered
protocol (docs/phase2_exposure_bias_plan.md) no model arm beat this count by a
material margin on held-out users, and the rule was to deploy whichever scorer
won, baseline or not. The service reports the scorer's identity and the numbers
behind that choice, so nobody has to guess what is running.

Personalisation here is limited to filtering out what the user has already been
shown; the ordering itself is the same for every user.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SERVING_DIR = Path("datastore/serving")


class KuaiRandScorer:
    def __init__(self, serving_dir: Path = SERVING_DIR):
        d = Path(serving_dir)
        items = pd.read_parquet(d / "kuairand_items.parquet")
        self.meta = json.loads((d / "kuairand_meta.json").read_text())
        # Pre-sorted at export time; ranking is a slice, not a sort, per request.
        self.item_ids = items["item_id"].to_numpy(np.int64)
        self.scores = items["impressions"].to_numpy(np.float64)
        seen = np.load(d / "kuairand_seen.npz")
        self._seen_users, self._seen_offsets, self._seen_items = (
            seen["users"], seen["offsets"], seen["items"])

    @property
    def catalogue_size(self) -> int:
        return len(self.item_ids)

    def seen_items(self, user_id: int) -> np.ndarray:
        """Items already shown to this user in the training window; empty if unknown."""
        i = np.searchsorted(self._seen_users, user_id)
        if i >= len(self._seen_users) or self._seen_users[i] != user_id:
            return np.empty(0, np.int32)
        return self._seen_items[self._seen_offsets[i]:self._seen_offsets[i + 1]]

    def top_k(self, user_id: int, k: int, exclude_seen: bool = True) -> tuple[list, int]:
        """Top-k (item_id, score) pairs and the number of items filtered out."""
        ids, scores = self.item_ids, self.scores
        n_excluded = 0
        if exclude_seen:
            seen = self.seen_items(user_id)
            if len(seen):
                keep = ~np.isin(ids, seen)
                n_excluded = int((~keep).sum())
                ids, scores = ids[keep], scores[keep]
        return [(int(i), float(s)) for i, s in zip(ids[:k], scores[:k])], n_excluded
