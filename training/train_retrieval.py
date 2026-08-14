"""
Two-tower retrieval training loop (BPR loss).

Usage:
    python training/train_retrieval.py
    python training/train_retrieval.py --config config/kuairec.yaml
"""
import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from config_loader import load_config
from models.two_tower import build_model
from data.schema import Cols


class InteractionDataset(Dataset):
    """
    PyTorch Dataset for BPR (Bayesian Personalised Ranking) training.

    Each sample contains one positive (user, item) pair from the observed
    interaction log, plus `num_neg` randomly sampled negative items that the
    user has NOT interacted with.  BPR loss then pushes the user embedding
    closer to the positive item and away from the negatives.

    Negative sampling strategy: uniform random — fast and works well in
    practice.  Known limitation: popular items are over-represented as
    negatives, which can cause popularity bias.  Tier 2 improvement: use
    in-batch negatives or popularity-corrected sampling.

    Args:
        df:      Interaction DataFrame with columns user_id and item_id.
        n_items: Total item vocabulary size — upper bound for negative sampling.
        num_neg: Number of negative samples per positive interaction.
        seed:    RNG seed for reproducibility.
    """
    def __init__(self, df, n_items, num_neg, seed=42):
        self.df = df
        self.n_items = n_items
        self.num_neg = num_neg
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        user_id = int(row[Cols.USER_ID])
        pos_item = int(row[Cols.ITEM_ID])

        neg_items = []
        while len(neg_items) < self.num_neg:
            candidate = int(self.rng.integers(0, self.n_items))
            if candidate != pos_item:
                neg_items.append(candidate)

        return {
            "user_id": torch.tensor(user_id, dtype=torch.int64),
            "pos_item": torch.tensor(pos_item, dtype=torch.int64),
            "neg_items": torch.tensor(neg_items, dtype=torch.int64),
        }


def bpr_loss(user_emb, pos_item_emb, neg_item_emb):
    """
    Bayesian Personalised Ranking loss.

    Objective: maximise the margin between a user's score for a positive
    item and their score for sampled negative items.

    Formula:
        loss = -mean( log σ(score_pos - score_neg_avg) )

    Where scores are dot products of L2-normalised embeddings (= cosine sim).
    Using logsigmoid instead of log(sigmoid(x)) is numerically more stable.

    Args:
        user_emb:      (batch, dim) — L2-normalised user embeddings.
        pos_item_emb:  (batch, dim) — embeddings of observed positive items.
        neg_item_emb:  (batch, num_neg, dim) — embeddings of sampled negatives.

    Returns:
        Scalar loss tensor.  Gradients flow through all three inputs.
    """
    pos_score = (user_emb * pos_item_emb).sum(dim=1)
    neg_scores = torch.bmm(neg_item_emb, user_emb.unsqueeze(2)).squeeze(2)
    neg_score = neg_scores.mean(dim=1)
    return -F.logsigmoid(pos_score - neg_score).mean()


def get_device(cfg):
    """
    Resolve the compute device from config.

    'auto' selects CUDA if available, otherwise CPU.  This lets the same
    config file work on a developer laptop (CPU) and a cloud GPU instance
    without any manual changes.
    """
    device_cfg = cfg["training"]["retrieval"]["device"]
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def train(cfg):
    """
    Full two-tower training loop.

    Steps:
        1. Load processed interaction data and ID maps.
        2. Build InteractionDataset + DataLoader for batched BPR training.
        3. Instantiate TwoTowerModel and Adam optimiser.
        4. Train for cfg[training][retrieval][epochs] epochs, logging loss each epoch.
        5. Save the final model checkpoint.
        6. Export item_embeddings.npy — all item vectors from the trained item tower.
           These are loaded by build_index.py to populate the FAISS index.
        7. Export user_embeddings.npy — all user vectors, used by the serving API
           to compute the query vector for each incoming recommend request.

    Dense features are zero-padded in Tier 1 (real features added in Tier 2).

    Args:
        cfg: Merged config dict.
    """
    with open("datastore/processed/id_maps.pkl", "rb") as f:
        id_maps = pickle.load(f)

    train_df = pd.read_parquet("datastore/processed/interactions/train.parquet")

    n_items = id_maps["n_items"]
    n_users = id_maps["n_users"]
    num_neg = cfg["features"]["num_neg_samples"]

    dataset = InteractionDataset(train_df, n_items, num_neg, seed=cfg["project"]["seed"])
    loader = DataLoader(dataset, batch_size=cfg["training"]["retrieval"]["batch_size"], shuffle=True, num_workers=0)

    device = get_device(cfg)
    model = build_model(cfg, n_users, n_items).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["retrieval"]["lr"],
        weight_decay=cfg["training"]["retrieval"]["weight_decay"],
    )

    epochs = cfg["training"]["retrieval"]["epochs"]
    user_dense_dim = cfg["features"]["user_dense_dim"]
    item_dense_dim = cfg["features"]["item_dense_dim"]

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        for batch in loader:
            batch_size = batch["user_id"].shape[0]

            user_dense = torch.zeros(batch_size, user_dense_dim)
            item_dense_pos = torch.zeros(batch_size, item_dense_dim)

            user_emb, pos_emb = model(
                batch["user_id"].to(device),
                user_dense.to(device),
                batch["pos_item"].to(device),
                item_dense_pos.to(device),
            )

            neg_items_flat = batch["neg_items"].view(-1)
            item_dense_neg = torch.zeros(neg_items_flat.shape[0], item_dense_dim)
            neg_emb_flat = model.item_tower(neg_items_flat.to(device), item_dense_neg.to(device))
            neg_emb = neg_emb_flat.view(batch_size, num_neg, -1)

            loss = bpr_loss(user_emb, pos_emb, neg_emb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        elapsed = time.time() - t0
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{epochs} — loss: {avg_loss:.4f} — {elapsed:.1f}s")

    # Save last epoch checkpoint
    ckpt_dir = Path(cfg["training"]["retrieval"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / f"two_tower_epoch_{epochs}.pt")

    # Export item embeddings
    model.eval()
    with torch.no_grad():
        all_item_ids = torch.arange(n_items).to(device)
        item_dense_all = torch.zeros(n_items, item_dense_dim).to(device)
        item_embs = model.item_tower(all_item_ids, item_dense_all).cpu().numpy()

    item_emb_path = "datastore/processed/item_embeddings.npy"
    np.save(item_emb_path, item_embs)
    print(f"Saved item embeddings: {item_emb_path}")

    # Export user embeddings
    with torch.no_grad():
        all_user_ids = torch.arange(n_users).to(device)
        user_dense_all = torch.zeros(n_users, user_dense_dim).to(device)
        user_embs = model.user_tower(all_user_ids, user_dense_all).cpu().numpy()

    user_emb_path = "datastore/processed/user_embeddings.npy"
    np.save(user_emb_path, user_embs)
    print(f"Saved user embeddings: {user_emb_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/kuairec.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg)
