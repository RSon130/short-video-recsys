"""
Shared dense-feature assembly for the ranker.

Why this module exists
----------------------
The ranker input vector is the concatenation

    [user_emb || item_emb || user_dense || item_dense]

and it must be built *identically* in three places: ranker training, offline
evaluation, and online serving. It originally was not. Training looked up real
dense features from the parquet files while evaluation and serving passed zero
vectors, so the model was scored on inputs it had never been trained on —
textbook train/serve skew. The symptom is subtle and misleading: the ranker
simply underperforms, which reads as "the ranker isn't adding value" rather than
as a bug, and the retrieval-vs-ranker A/B comparison silently becomes worthless.

Every caller now goes through `DenseFeatureStore.build_input()`. There is no
second implementation to drift from.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from data.schema import Cols

DEFAULT_PROCESSED_DIR = Path("datastore/processed")


class DenseFeatureStore:
    """
    Lookup table for per-user and per-item dense side features.

    Users or items absent from the feature tables fall back to a zero vector —
    the same fallback used during training, so an unseen ID degrades to the
    embedding-only case rather than raising at request time.
    """

    def __init__(self, user_features: pd.DataFrame, item_features: pd.DataFrame):
        self.user_dense_dim = user_features.shape[1] - 1  # exclude the id column
        self.item_dense_dim = item_features.shape[1] - 1

        self._user_map = {
            int(row[Cols.USER_ID]): row.drop(Cols.USER_ID).values.astype(np.float32)
            for _, row in user_features.iterrows()
        }
        self._item_map = {
            int(row[Cols.ITEM_ID]): row.drop(Cols.ITEM_ID).values.astype(np.float32)
            for _, row in item_features.iterrows()
        }

        self._user_zero = np.zeros(self.user_dense_dim, dtype=np.float32)
        self._item_zero = np.zeros(self.item_dense_dim, dtype=np.float32)

    @classmethod
    def load(cls, processed_dir=DEFAULT_PROCESSED_DIR) -> "DenseFeatureStore":
        """Load the store from the parquet files written by features/engineer.py."""
        processed_dir = Path(processed_dir)
        return cls(
            pd.read_parquet(processed_dir / "user_features.parquet"),
            pd.read_parquet(processed_dir / "item_features.parquet"),
        )

    def user(self, uid: int) -> np.ndarray:
        return self._user_map.get(int(uid), self._user_zero)

    def item(self, iid: int) -> np.ndarray:
        return self._item_map.get(int(iid), self._item_zero)

    def build_input(
        self,
        user_emb: np.ndarray,
        item_emb: np.ndarray,
        uid: int,
        iid: int,
    ) -> np.ndarray:
        """
        Assemble one ranker input row.

        This is the single definition of the ranker's input layout. Training,
        evaluation, and serving all call it.
        """
        return np.concatenate(
            [user_emb, item_emb, self.user(uid), self.item(iid)]
        ).astype(np.float32)

    def build_batch(
        self,
        user_emb: np.ndarray,
        item_embs: np.ndarray,
        uid: int,
        iids,
    ) -> np.ndarray:
        """
        Assemble a batch of rows for one user against many candidate items.

        Scoring candidates in a single batched forward pass rather than one call
        per item is what keeps ranking inside the latency budget: the per-call
        PyTorch overhead dominates at 200 candidates.
        """
        iids = [int(i) for i in iids]
        return np.stack(
            [self.build_input(user_emb, item_embs[i], uid, i) for i in iids]
        ).astype(np.float32)
