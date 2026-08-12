"""
KuaiRec data loader.

Downloads: https://kuairec.com/
Expected files in datastore/raw/kuairec/:
    big_matrix.csv          (or small_matrix.csv for dev)
    item_categories.csv
    item_daily_features.csv
    user_features.csv

KuaiRec paper: https://arxiv.org/abs/2202.10842
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data.base import BaseDataLoader
from data.schema import Cols


class KuaiRecLoader(BaseDataLoader):

    # KuaiRec column → canonical column
    _INTERACTION_MAP = {
        "video_id": Cols.ITEM_ID,
        "is_like": Cols.LIKE,
        "is_comment": Cols.COMMENT,
        "is_share": Cols.SHARE,
        "is_follow": Cols.FOLLOW,
    }

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        kr_cfg = cfg["data"]["kuairec"]
        self.raw_dir = Path(kr_cfg["raw_dir"])
        self.col_map = kr_cfg.get("column_map", {})
        self.clip = kr_cfg.get("clip_watch_ratio", True)
        self.min_user = kr_cfg.get("min_interactions_per_user", 5)
        self.min_item = kr_cfg.get("min_interactions_per_item", 5)
        self.interaction_file = kr_cfg.get("interaction_file", "big_matrix.csv")

    # ------------------------------------------------------------------
    # Interface implementation
    # ------------------------------------------------------------------

    def load_interactions(self) -> pd.DataFrame:
        path = self.raw_dir / self.interaction_file
        self._check_file(path)

        df = pd.read_csv(path)
        df = self._rename(df, self._INTERACTION_MAP)

        # Ensure canonical columns exist with sensible defaults
        for col in [Cols.LIKE, Cols.COMMENT, Cols.SHARE, Cols.FOLLOW]:
            if col not in df.columns:
                df[col] = 0

        if Cols.TIMESTAMP not in df.columns:
            # KuaiRec big_matrix has 'date' as YYYYMMDD integer; derive timestamp
            if "date" in df.columns:
                df[Cols.TIMESTAMP] = pd.to_datetime(
                    df["date"].astype(str), format="%Y%m%d"
                ).astype(np.int64) // 10**9
            else:
                df[Cols.TIMESTAMP] = 0

        if Cols.WATCH_RATIO not in df.columns:
            raise ValueError("KuaiRec interaction file missing 'watch_ratio' column")

        if self.clip:
            df[Cols.WATCH_RATIO] = df[Cols.WATCH_RATIO].clip(0.0, 1.0)

        df = self._filter_cold_start(df)

        required = [Cols.USER_ID, Cols.ITEM_ID, Cols.TIMESTAMP, Cols.WATCH_RATIO]
        self._require_cols(df, required, "KuaiRec interactions")
        return df.reset_index(drop=True)

    def load_user_features(self) -> pd.DataFrame:
        path = self.raw_dir / "user_features.csv"
        if not path.exists():
            # Return a minimal stub so the pipeline still runs
            return pd.DataFrame({Cols.USER_ID: []})

        df = pd.read_csv(path)
        if "user_id" not in df.columns:
            raise ValueError("user_features.csv missing 'user_id' column")
        df = self._encode_categoricals(df, exclude=[Cols.USER_ID])
        return df.fillna(0)

    def load_item_features(self) -> pd.DataFrame:
        path = self.raw_dir / "item_categories.csv"
        self._check_file(path)

        df = pd.read_csv(path)
        if "video_id" in df.columns:
            df = df.rename(columns={"video_id": Cols.ITEM_ID})

        # item_daily_features: aggregate to single row per item
        daily_path = self.raw_dir / "item_daily_features.csv"
        if daily_path.exists():
            daily = pd.read_csv(daily_path)
            if "video_id" in daily.columns:
                daily = daily.rename(columns={"video_id": Cols.ITEM_ID})
            numeric_cols = daily.select_dtypes(include="number").columns.tolist()
            agg_cols = [c for c in numeric_cols if c != Cols.ITEM_ID]
            if agg_cols:
                daily_agg = (
                    daily.groupby(Cols.ITEM_ID)[agg_cols].mean().reset_index()
                )
                df = df.merge(daily_agg, on=Cols.ITEM_ID, how="left")

        # Explode category list into one-hot dummies
        if "feat" in df.columns:
            df = self._expand_category_feat(df)

        df = self._encode_categoricals(df, exclude=[Cols.ITEM_ID])
        return df.fillna(0)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _filter_cold_start(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove users/items with too few interactions."""
        for _ in range(3):  # iterate to handle cascades
            user_counts = df[Cols.USER_ID].value_counts()
            item_counts = df[Cols.ITEM_ID].value_counts()
            df = df[
                df[Cols.USER_ID].isin(user_counts[user_counts >= self.min_user].index)
                & df[Cols.ITEM_ID].isin(item_counts[item_counts >= self.min_item].index)
            ]
        return df

    @staticmethod
    def _expand_category_feat(df: pd.DataFrame) -> pd.DataFrame:
        """
        KuaiRec item_categories has a 'feat' column like '[3, 17, 42]'.
        Explode into binary multi-hot columns cat_3, cat_17, ...
        """
        import ast

        def safe_parse(v):
            try:
                return ast.literal_eval(v) if isinstance(v, str) else v
            except Exception:
                return []

        df = df.copy()
        df["_feat_list"] = df["feat"].apply(safe_parse)
        all_cats = sorted({c for lst in df["_feat_list"] for c in lst})
        for cat in all_cats:
            df[f"cat_{cat}"] = df["_feat_list"].apply(lambda lst: int(cat in lst))
        return df.drop(columns=["feat", "_feat_list"])

    @staticmethod
    def _encode_categoricals(df: pd.DataFrame, exclude: list) -> pd.DataFrame:
        """Label-encode any remaining string/object columns."""
        for col in df.select_dtypes(include=["object", "category"]).columns:
            if col in exclude:
                continue
            df[col] = pd.Categorical(df[col]).codes
        return df

    @staticmethod
    def _check_file(path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(
                f"KuaiRec file not found: {path}\n"
                "Run: python scripts/download_kuairec.py"
            )
