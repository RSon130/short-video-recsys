"""
Feature engineering pipeline.

Loads raw data via the loader, applies temporal split, saves parquet files.

Usage:
    python features/engineer.py                      # uses config/kuairec.yaml
    python features/engineer.py --config config/kuairec.yaml
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from config_loader import load_config
from data import get_loader
from data.schema import Cols


def normalize_dense_features(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """
    Put dense side features on a comparable scale to the learned embeddings.

    Why this exists: the raw KuaiRec side features span roughly twelve orders of
    magnitude — item columns reach 2.6e11 (play/show counts), user columns reach
    ~2e3 — while the tower embeddings are L2-normalised to about 0.1. Feeding
    both into the ranker's first Linear layer saturated its sigmoid, and the
    model collapsed to predicting 1.0 for every input. That failure is easy to
    misread as "the ranker adds nothing": the loss simply sits at
    var + (1 - mean)^2 and never moves.

    Two steps:
      1. signed log1p — compresses count-like columns whose distribution spans
         many orders of magnitude, without breaking on zeros or negatives.
      2. z-score — centres and scales each column to roughly unit variance.

    Columns with zero variance carry no information; they are scaled to zero
    rather than dividing by zero.

    Note on leakage: these are static user/item attributes rather than
    time-varying interaction data, so statistics are fitted over the whole
    feature table. Anything derived from interactions would have to be fitted on
    the training split alone.
    """
    ids = df[id_col]
    values = df.drop(columns=[id_col]).astype("float64")

    values = np.sign(values) * np.log1p(np.abs(values))

    std = values.std()
    normalized = (values - values.mean()) / std.replace(0.0, 1.0)
    normalized[std[std == 0.0].index] = 0.0

    return pd.concat([ids, normalized.astype("float32")], axis=1)


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    """
    Write a DataFrame to a Parquet file, creating parent directories as needed.

    Parquet is chosen over CSV for downstream ML work because:
    - Columnar storage makes feature-column reads ~10× faster.
    - Schema (dtypes) is preserved — no silent string/int coercion on reload.
    - ~3–5× smaller file size via built-in compression.

    Args:
        df:   DataFrame to write.
        path: Destination file path (any depth; directories are created).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def run(cfg: dict) -> None:
    """
    Execute the full feature engineering pipeline.

    Steps:
        1. Instantiate the appropriate data loader (via get_loader).
        2. Call build_feature_store() — loads raw CSVs, remaps user/item IDs
           to contiguous integers, returns a FeatureStore.
        3. Apply temporal_split() — splits interactions by timestamp percentile
           to guarantee no future data leaks into training.
        4. Persist all outputs to datastore/processed/:
               interactions/train.parquet
               interactions/val.parquet
               interactions/test.parquet
               user_features.parquet
               item_features.parquet
               id_maps.pkl  ← dict with user_id_map, item_id_map, n_users, n_items

    The id_maps.pkl file is critical: it maps original raw IDs (e.g. KuaiRec
    video_id) to the integer indices used by embedding layers.  Without it the
    model cannot translate API user_ids back to embedding rows at serving time.

    Args:
        cfg: Merged config dict (from load_config).
    """
    project_root = Path(__file__).parent.parent

    loader = get_loader(cfg)
    fs = loader.build_feature_store()
    split = loader.temporal_split(
        fs.interactions,
        cfg["data"]["train_ratio"],
        cfg["data"]["val_ratio"],
    )

    # Read the destination from config rather than hardcoding it. cfg defined
    # processed_dir all along and nothing consumed it, so the path here was
    # assembled from literal parts — which meant it silently kept writing into
    # the data/ *package* directory after storage moved to datastore/.
    processed_dir = project_root / cfg["data"]["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    interactions_dir = processed_dir / "interactions"
    save_parquet(split.train, interactions_dir / "train.parquet")
    save_parquet(split.val, interactions_dir / "val.parquet")
    save_parquet(split.test, interactions_dir / "test.parquet")

    # Normalised before persisting so every consumer — ranker training,
    # evaluation, serving — reads the same scaled values.
    user_features = normalize_dense_features(fs.user_features, Cols.USER_ID)
    item_features = normalize_dense_features(fs.item_features, Cols.ITEM_ID)
    save_parquet(user_features, processed_dir / "user_features.parquet")
    save_parquet(item_features, processed_dir / "item_features.parquet")

    user_dense_dim = user_features.shape[1] - 1  # exclude user_id column
    item_dense_dim = item_features.shape[1] - 1  # exclude item_id column

    id_maps = {
        "user_id_map": fs.user_id_map,
        "item_id_map": fs.item_id_map,
        "n_users": fs.n_users,
        "n_items": fs.n_items,
        "user_dense_dim": user_dense_dim,
        "item_dense_dim": item_dense_dim,
    }
    with open(processed_dir / "id_maps.pkl", "wb") as f:
        pickle.dump(id_maps, f)

    print(f"train rows: {len(split.train)}")
    print(f"val rows:   {len(split.val)}")
    print(f"test rows:  {len(split.test)}")
    print(f"user_features rows: {len(user_features)}")
    print(f"item_features rows: {len(item_features)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/kuairec.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run(cfg)
