"""
MLP ranker training loop (MSE loss on watch_ratio).

Usage:
    python training/train_ranking.py
    python training/train_ranking.py --config config/kuairec.yaml
"""
import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from config_loader import load_config
from features.dense_features import DenseFeatureStore
from models.ranker import build_ranker
from data.schema import Cols


class RankingDataset(Dataset):
    """
    PyTorch Dataset for MSE-based ranker training.

    Constructs a feature vector for each observed interaction by concatenating:
        [user_embedding || item_embedding || user_dense_features || item_dense_features]

    The label is the observed watch_ratio ∈ [0, 1], which is the regression
    target for the MLP ranker.

    Design note — O(1) feature lookup:
        User and item side features are pre-indexed into dicts keyed by integer
        ID in __init__, so __getitem__ never scans the DataFrame — crucial for
        large datasets where a pandas .loc per sample would be prohibitively slow.

    Args:
        interactions:  DataFrame with columns user_id, item_id, watch_ratio.
        user_embs:     np.ndarray of shape (n_users, embedding_dim).
        item_embs:     np.ndarray of shape (n_items, embedding_dim).
        user_features: DataFrame with column user_id plus numeric feature columns.
        item_features: DataFrame with column item_id plus numeric feature columns.
    """
    def __init__(self, interactions, user_embs, item_embs, features):
        self.uids = interactions[Cols.USER_ID].to_numpy(dtype=np.int64)
        self.iids = interactions[Cols.ITEM_ID].to_numpy(dtype=np.int64)
        self.labels = interactions[Cols.WATCH_RATIO].to_numpy(dtype=np.float32)
        self.user_embs = user_embs
        self.item_embs = item_embs

        # Shared with evaluation and serving — see features/dense_features.py for
        # why the input layout lives in one place.
        self.features = features
        self.user_dense_dim = features.user_dense_dim
        self.item_dense_dim = features.item_dense_dim

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Returns the raw index only; the feature vector is assembled per batch
        # by collate(). Building one 241-dim row at a time in Python dominated
        # the epoch, and there are millions of rows per epoch.
        return idx

    def collate(self, indices):
        """Assemble one batch of ranker inputs with vectorised lookups."""
        indices = np.asarray(indices, dtype=np.int64)
        uids = self.uids[indices]
        iids = self.iids[indices]

        x = self.features.build_matrix(
            self.user_embs[uids], self.item_embs[iids], uids, iids
        )
        return {
            "x": torch.from_numpy(x),
            "label": torch.from_numpy(self.labels[indices]),
        }


def get_device(cfg):
    """
    Resolve the compute device from config (same logic as in train_retrieval).
    Reads from cfg[training][ranking][device].
    """
    device_cfg = cfg["training"]["ranking"]["device"]
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def train(cfg):
    """
    Full MLP ranker training loop.

    Steps:
        1. Load processed interaction data, pre-computed user/item embeddings,
           and side-feature parquets.
        2. Build RankingDataset + DataLoader.
        3. Instantiate MLPRanker and Adam optimiser.
        4. Train with MSELoss for cfg[training][ranking][epochs] epochs,
           logging loss per epoch.
        5. Save ranker_model.pt to datastore/processed/ (loaded by serving API).
        6. Save checkpoint to checkpoint_dir.

    Why train the ranker *after* the retrieval model?
        The ranker uses embeddings produced by the trained two-tower model as
        input features.  Training order: retrieval → export embeddings → ranking.

    Args:
        cfg: Merged config dict.
    """
    train_df = pd.read_parquet("datastore/processed/interactions/train.parquet")
    val_df = pd.read_parquet("datastore/processed/interactions/val.parquet")
    user_embs = np.load("datastore/processed/user_embeddings.npy")
    item_embs = np.load("datastore/processed/item_embeddings.npy")
    features = DenseFeatureStore.load()

    batch_size = cfg["training"]["ranking"]["batch_size"]
    dataset = RankingDataset(train_df, user_embs, item_embs, features)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=0, collate_fn=dataset.collate)

    val_dataset = RankingDataset(val_df, user_embs, item_embs, features)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, collate_fn=val_dataset.collate)
    print(f"Ranker training rows: {len(dataset):,} | validation rows: {len(val_dataset):,}")

    device = get_device(cfg)
    model = build_ranker(cfg, dataset.user_dense_dim, dataset.item_dense_dim).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["ranking"]["lr"],
        weight_decay=cfg["training"]["ranking"]["weight_decay"],
    )
    criterion = nn.MSELoss()

    epochs = cfg["training"]["ranking"]["epochs"]
    patience = cfg["training"]["ranking"]["early_stopping_patience"]

    def run_epoch(data_loader, train_mode):
        """One pass over data_loader. Shared so the two paths cannot diverge."""
        model.train() if train_mode else model.eval()
        total = 0.0
        with torch.set_grad_enabled(train_mode):
            for batch in data_loader:
                preds = model(batch["x"].to(device)).squeeze(1)
                loss = criterion(preds, batch["label"].to(device))
                if train_mode:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                total += loss.item()
        return total / len(data_loader)

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_mse = run_epoch(loader, True)
        val_mse = run_epoch(val_loader, False)
        elapsed = time.time() - t0

        marker = ""
        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_without_improvement = 0
            marker = "  <- best"
        else:
            epochs_without_improvement += 1

        print(f"Epoch {epoch}/{epochs} — train mse: {train_mse:.6f} — "
              f"val mse: {val_mse:.6f} — {elapsed:.1f}s{marker}")

        if epochs_without_improvement >= patience:
            print(f"Early stopping: no validation improvement in {patience} epochs.")
            break

    model.load_state_dict(best_state)
    print(f"Restored best ranker from epoch {best_epoch} (val mse {best_val:.6f})")

    torch.save(model.state_dict(), "datastore/processed/ranker_model.pt")

    ckpt_dir = Path(cfg["training"]["ranking"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / f"ranker_best_epoch_{best_epoch}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/kuairec.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg)
