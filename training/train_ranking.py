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

from features.engineer import load_config
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
    def __init__(self, interactions, user_embs, item_embs, user_features, item_features):
        self.records = interactions[[Cols.USER_ID, Cols.ITEM_ID, Cols.WATCH_RATIO]].values
        self.user_embs = user_embs
        self.item_embs = item_embs

        self.user_feat_map = {
            int(row[Cols.USER_ID]): row.drop(Cols.USER_ID).values.astype(np.float32)
            for _, row in user_features.iterrows()
        }
        self.item_feat_map = {
            int(row[Cols.ITEM_ID]): row.drop(Cols.ITEM_ID).values.astype(np.float32)
            for _, row in item_features.iterrows()
        }
        self.user_dense_dim = user_features.shape[1] - 1
        self.item_dense_dim = item_features.shape[1] - 1

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        uid, iid, label = self.records[idx]
        uid, iid = int(uid), int(iid)

        u_emb = self.user_embs[uid]
        i_emb = self.item_embs[iid]

        u_row = self.user_feat_map.get(uid, np.zeros(self.user_dense_dim, dtype=np.float32))
        i_row = self.item_feat_map.get(iid, np.zeros(self.item_dense_dim, dtype=np.float32))

        x = np.concatenate([u_emb, i_emb, u_row, i_row]).astype(np.float32)
        return {
            "x": torch.from_numpy(x),
            "label": torch.tensor(label, dtype=torch.float32),
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
        5. Save ranker_model.pt to data/processed/ (loaded by serving API).
        6. Save checkpoint to checkpoint_dir.

    Why train the ranker *after* the retrieval model?
        The ranker uses embeddings produced by the trained two-tower model as
        input features.  Training order: retrieval → export embeddings → ranking.

    Args:
        cfg: Merged config dict.
    """
    train_df = pd.read_parquet("data/processed/interactions/train.parquet")
    user_embs = np.load("data/processed/user_embeddings.npy")
    item_embs = np.load("data/processed/item_embeddings.npy")
    user_features = pd.read_parquet("data/processed/user_features.parquet")
    item_features = pd.read_parquet("data/processed/item_features.parquet")

    dataset = RankingDataset(train_df, user_embs, item_embs, user_features, item_features)
    loader = DataLoader(dataset, batch_size=cfg["training"]["ranking"]["batch_size"], shuffle=True, num_workers=0)

    device = get_device(cfg)
    model = build_ranker(cfg, dataset.user_dense_dim, dataset.item_dense_dim).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["ranking"]["lr"],
        weight_decay=cfg["training"]["ranking"]["weight_decay"],
    )
    criterion = nn.MSELoss()

    epochs = cfg["training"]["ranking"]["epochs"]

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        for batch in loader:
            preds = model(batch["x"].to(device)).squeeze(1)
            loss = criterion(preds, batch["label"].to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        elapsed = time.time() - t0
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{epochs} — mse: {avg_loss:.6f} — {elapsed:.1f}s")

    # Save final model
    torch.save(model.state_dict(), "data/processed/ranker_model.pt")

    # Save checkpoint
    ckpt_dir = Path(cfg["training"]["ranking"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / f"ranker_epoch_{epochs}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/kuairec.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg)
