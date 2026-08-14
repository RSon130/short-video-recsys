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

        # Features are held as id-indexed matrices rather than dicts of rows so
        # a whole batch can be gathered with one fancy-index instead of a Python
        # loop. Rows for ids absent from the feature tables stay zero, which is
        # the same fallback the per-sample path uses.
        self._user_mat = self._to_matrix(user_features, Cols.USER_ID, self.user_dense_dim)
        self._item_mat = self._to_matrix(item_features, Cols.ITEM_ID, self.item_dense_dim)

        self._user_zero = np.zeros(self.user_dense_dim, dtype=np.float32)
        self._item_zero = np.zeros(self.item_dense_dim, dtype=np.float32)

    @staticmethod
    def _to_matrix(features: pd.DataFrame, id_col: str, dim: int) -> np.ndarray:
        ids = features[id_col].to_numpy(dtype=np.int64)
        values = features.drop(columns=[id_col]).to_numpy(dtype=np.float32)
        matrix = np.zeros((int(ids.max()) + 1 if len(ids) else 0, dim), dtype=np.float32)
        matrix[ids] = values
        return matrix

    def _lookup(self, matrix: np.ndarray, ids: np.ndarray, dim: int) -> np.ndarray:
        """Gather rows for `ids`, returning zeros for ids outside the table."""
        ids = np.asarray(ids, dtype=np.int64)
        out = np.zeros((len(ids), dim), dtype=np.float32)
        known = ids < len(matrix)
        out[known] = matrix[ids[known]]
        return out

    @classmethod
    def load(cls, processed_dir=DEFAULT_PROCESSED_DIR) -> "DenseFeatureStore":
        """Load the store from the parquet files written by features/engineer.py."""
        processed_dir = Path(processed_dir)
        return cls(
            pd.read_parquet(processed_dir / "user_features.parquet"),
            pd.read_parquet(processed_dir / "item_features.parquet"),
        )

    def user(self, uid: int) -> np.ndarray:
        uid = int(uid)
        return self._user_mat[uid] if uid < len(self._user_mat) else self._user_zero

    def item(self, iid: int) -> np.ndarray:
        iid = int(iid)
        return self._item_mat[iid] if iid < len(self._item_mat) else self._item_zero

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
        iids = np.asarray(iids, dtype=np.int64)
        uids = np.full(len(iids), int(uid), dtype=np.int64)
        user_embs = np.broadcast_to(user_emb, (len(iids), len(user_emb)))
        return self.build_matrix(user_embs, item_embs[iids], uids, iids)

    def build_matrix(
        self,
        user_embs: np.ndarray,
        item_embs: np.ndarray,
        uids: np.ndarray,
        iids: np.ndarray,
    ) -> np.ndarray:
        """
        Assemble many rows at once, one row per (uid, iid) pair.

        Vectorised counterpart to build_input, and the same column order. Ranker
        training assembles millions of rows per epoch; doing that with a Python
        loop and a per-row np.concatenate dominated the epoch time.

        Args:
            user_embs: (n, emb_dim) user embeddings, already gathered.
            item_embs: (n, emb_dim) item embeddings, already gathered.
            uids:      (n,) user ids, for dense-feature lookup.
            iids:      (n,) item ids, for dense-feature lookup.

        Returns:
            (n, 2*emb_dim + user_dense_dim + item_dense_dim) float32 matrix.
        """
        return np.hstack([
            np.asarray(user_embs, dtype=np.float32),
            np.asarray(item_embs, dtype=np.float32),
            self._lookup(self._user_mat, uids, self.user_dense_dim),
            self._lookup(self._item_mat, iids, self.item_dense_dim),
        ])
