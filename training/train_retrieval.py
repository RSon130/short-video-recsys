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

    Labelling strategy — why this dataset does not sample random negatives
    ------------------------------------------------------------------------
    The obvious implicit-feedback recipe is "observed item = positive, random
    unobserved item = negative". That recipe assumes a *sparse* interaction
    matrix, where an unobserved pair is a reasonable guess at disinterest.

    KuaiRec is the opposite: it is fully observed, measured here at **99.6%
    density**. Practically every (user, item) pair carries a real watch_ratio,
    so "an item the user has not interacted with" is an almost empty set. The
    previous implementation drew negatives uniformly from the whole item
    vocabulary, which meant negatives were observed interactions with the same
    watch_ratio distribution as the positives (mean 0.702 either way).
    Positives and negatives were statistically identical, there was no ranking
    signal to learn, and training flatlined at loss ~0.597 against a
    random-init baseline of log(2) ~ 0.693.

    Since the matrix is fully observed, engagement is *known* rather than
    inferred:

        positive : watch_ratio >= pos_threshold   (the user watched it through)
        negative : watch_ratio <= neg_threshold   (the user bailed out early)

    Pairs in between are dropped from training rather than forced into a class
    they don't clearly belong to. Negatives are drawn from the *same user's*
    low-engagement items, so the loss contrasts two things that user actually
    saw — which is the comparison BPR is meant to model.

    Performance note: users, items, and per-user negative pools are converted
    to numpy up front. The previous version called df.iloc[idx] per sample;
    that single pandas lookup dominated the epoch, and removing it is most of
    the speedup.

    Args:
        df:             Interaction DataFrame with user_id, item_id, watch_ratio.
        num_neg:        Negatives per positive.
        pos_threshold:  Minimum watch_ratio for a positive.
        neg_threshold:  Maximum watch_ratio for a negative.
        seed:           RNG seed for reproducibility.
    """
    def __init__(self, df, num_neg, pos_threshold=0.7, neg_threshold=0.3, seed=42):
        self.num_neg = num_neg
        self.rng = np.random.default_rng(seed)

        positives = df.loc[df[Cols.WATCH_RATIO] >= pos_threshold,
                           [Cols.USER_ID, Cols.ITEM_ID]].to_numpy(dtype=np.int64)
        if len(positives) == 0:
            raise ValueError(
                f"No interactions with watch_ratio >= {pos_threshold}; "
                f"nothing to train on."
            )
        self.users = positives[:, 0]
        self.pos_items = positives[:, 1]

        negatives = df.loc[df[Cols.WATCH_RATIO] <= neg_threshold,
                           [Cols.USER_ID, Cols.ITEM_ID]]
        self.neg_pools = {
            int(uid): group.to_numpy(dtype=np.int64)
            for uid, group in negatives.groupby(Cols.USER_ID)[Cols.ITEM_ID]
        }
        # Users with no low-engagement item of their own fall back to the
        # global pool of items that rate poorly across the population.
        self.global_neg_pool = negatives[Cols.ITEM_ID].to_numpy(dtype=np.int64)
        if len(self.global_neg_pool) == 0:
            raise ValueError(
                f"No interactions with watch_ratio <= {neg_threshold}; "
                f"no negatives available."
            )

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user_id = self.users[idx]
        pos_item = self.pos_items[idx]

        pool = self.neg_pools.get(int(user_id), self.global_neg_pool)
        neg_items = pool[self.rng.integers(0, len(pool), self.num_neg)]

        return {
            "user_id": torch.from_numpy(np.asarray(user_id, dtype=np.int64)),
            "pos_item": torch.from_numpy(np.asarray(pos_item, dtype=np.int64)),
            "neg_items": torch.from_numpy(neg_items),
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


def run_epoch(model, loader, cfg, num_neg, device, optimizer=None):
    """
    Run one pass over `loader`, training if an optimizer is given.

    Shared by the training and validation passes so the two cannot compute the
    loss differently — the same class of drift that produced the train/serve
    skew in the ranker.

    Returns:
        Mean BPR loss over the epoch.
    """
    training = optimizer is not None
    model.train() if training else model.eval()

    user_dense_dim = cfg["features"]["user_dense_dim"]
    item_dense_dim = cfg["features"]["item_dense_dim"]
    total_loss = 0.0

    with torch.set_grad_enabled(training):
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

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

    return total_loss / len(loader)


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

    dataset = InteractionDataset(
        train_df,
        num_neg,
        pos_threshold=cfg["features"]["positive_watch_ratio"],
        neg_threshold=cfg["features"]["negative_watch_ratio"],
        seed=cfg["project"]["seed"],
    )
    print(
        f"Training pairs: {len(dataset):,} positives "
        f"(watch_ratio >= {cfg['features']['positive_watch_ratio']}) "
        f"from {len(train_df):,} interactions"
    )
    loader = DataLoader(dataset, batch_size=cfg["training"]["retrieval"]["batch_size"], shuffle=True, num_workers=0)

    device = get_device(cfg)
    model = build_model(cfg, n_users, n_items).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["retrieval"]["lr"],
        weight_decay=cfg["training"]["retrieval"]["weight_decay"],
    )

    epochs = cfg["training"]["retrieval"]["epochs"]
    patience = cfg["training"]["retrieval"]["early_stopping_patience"]
    user_dense_dim = cfg["features"]["user_dense_dim"]
    item_dense_dim = cfg["features"]["item_dense_dim"]

    # Validation exists to answer "how many epochs?" with evidence instead of a
    # guess. The first full run showed why it matters: training loss bottomed at
    # epoch 4 and drifted upward for the remaining 16, so the exported
    # embeddings came from a model measurably worse than the best one seen.
    val_df = pd.read_parquet("datastore/processed/interactions/val.parquet")
    val_dataset = InteractionDataset(
        val_df,
        num_neg,
        pos_threshold=cfg["features"]["positive_watch_ratio"],
        neg_threshold=cfg["features"]["negative_watch_ratio"],
        seed=cfg["project"]["seed"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"]["retrieval"]["batch_size"],
        shuffle=False,
        num_workers=0,
    )
    print(f"Validation pairs: {len(val_dataset):,}")

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = run_epoch(model, loader, cfg, num_neg, device, optimizer)
        val_loss = run_epoch(model, val_loader, cfg, num_neg, device)
        elapsed = time.time() - t0

        marker = ""
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_without_improvement = 0
            marker = "  <- best"
        else:
            epochs_without_improvement += 1

        print(f"Epoch {epoch}/{epochs} — train: {train_loss:.4f} — "
              f"val: {val_loss:.4f} — {elapsed:.1f}s{marker}")

        if epochs_without_improvement >= patience:
            print(f"Early stopping: no validation improvement in {patience} epochs.")
            break

    # Everything downstream must come from the best model, not the last one.
    model.load_state_dict(best_state)
    print(f"Restored best model from epoch {best_epoch} (val loss {best_val:.4f})")

    ckpt_dir = Path(cfg["training"]["retrieval"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / f"two_tower_best_epoch_{best_epoch}.pt")

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
