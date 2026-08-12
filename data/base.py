"""
Abstract base class for all data loaders.

To plug in a new data source:
1. Subclass BaseDataLoader
2. Implement the three abstract methods
3. Register it in data/__init__.py::get_loader()

The returned DataFrames must use the canonical column names from schema.Cols.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple

import pandas as pd

from data.schema import Cols, DataSplit, FeatureStore


class BaseDataLoader(ABC):
    """
    Contract every data source must fulfill.

    All returned DataFrames use canonical column names (schema.Cols).
    Implementing classes handle source-specific column renaming internally.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Required interface
    # ------------------------------------------------------------------

    @abstractmethod
    def load_interactions(self) -> pd.DataFrame:
        """
        Returns a DataFrame with at minimum:
            user_id, item_id, timestamp, watch_ratio, like
        Optional: comment, share, follow
        """

    @abstractmethod
    def load_user_features(self) -> pd.DataFrame:
        """
        Returns a DataFrame indexed by user_id with numeric feature columns.
        Must contain at least the column user_id.
        """

    @abstractmethod
    def load_item_features(self) -> pd.DataFrame:
        """
        Returns a DataFrame indexed by item_id with numeric feature columns.
        Must contain at least item_id and category_id.
        """

    # ------------------------------------------------------------------
    # Shared logic (override if needed)
    # ------------------------------------------------------------------

    def build_feature_store(self) -> FeatureStore:
        """Load all tables, build integer ID maps, return a FeatureStore."""
        interactions = self.load_interactions()
        user_features = self.load_user_features()
        item_features = self.load_item_features()

        # Build contiguous integer ID maps for embedding lookup
        user_ids = sorted(interactions[Cols.USER_ID].unique())
        item_ids = sorted(interactions[Cols.ITEM_ID].unique())
        user_id_map = {uid: idx for idx, uid in enumerate(user_ids)}
        item_id_map = {iid: idx for idx, iid in enumerate(item_ids)}

        interactions[Cols.USER_ID] = interactions[Cols.USER_ID].map(user_id_map)
        interactions[Cols.ITEM_ID] = interactions[Cols.ITEM_ID].map(item_id_map)

        user_features[Cols.USER_ID] = user_features[Cols.USER_ID].map(user_id_map)
        item_features[Cols.ITEM_ID] = item_features[Cols.ITEM_ID].map(item_id_map)

        # Drop rows whose IDs didn't survive the map (outside interaction set)
        user_features = user_features.dropna(subset=[Cols.USER_ID])
        item_features = item_features.dropna(subset=[Cols.ITEM_ID])
        user_features[Cols.USER_ID] = user_features[Cols.USER_ID].astype(int)
        item_features[Cols.ITEM_ID] = item_features[Cols.ITEM_ID].astype(int)

        return FeatureStore(
            interactions=interactions,
            user_features=user_features,
            item_features=item_features,
            user_id_map=user_id_map,
            item_id_map=item_id_map,
            n_users=len(user_id_map),
            n_items=len(item_id_map),
        )

    def temporal_split(
        self,
        interactions: pd.DataFrame,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
    ) -> DataSplit:
        """
        Time-based split — avoids leaking future interactions into training.
        Splits globally on timestamp percentile.
        """
        interactions = interactions.sort_values(Cols.TIMESTAMP).reset_index(drop=True)
        n = len(interactions)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        return DataSplit(
            train=interactions.iloc[:train_end].copy(),
            val=interactions.iloc[train_end:val_end].copy(),
            test=interactions.iloc[val_end:].copy(),
        )

    # ------------------------------------------------------------------
    # Helpers available to subclasses
    # ------------------------------------------------------------------

    @staticmethod
    def _rename(df: pd.DataFrame, column_map: dict) -> pd.DataFrame:
        """Rename columns using a source→canonical map; ignore missing cols."""
        rename = {src: dst for src, dst in column_map.items() if src in df.columns}
        return df.rename(columns=rename)

    @staticmethod
    def _require_cols(df: pd.DataFrame, required: list, source: str) -> None:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{source}: missing required columns {missing}")
