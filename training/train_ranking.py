"""
MLP ranker training loop.

Objective is set by config (ranking.objective):
    listwise    softmax over one positive and N negatives — optimises ordering
                against several alternatives at once (sampled softmax / InfoNCE)
    pairwise    BPR over a single (positive, negative) pair — the N=1 case
    regression  MSE against observed watch_ratio — optimises calibration

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
import torch.nn.functional as F
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


class PairwiseRankingDataset(Dataset):
    """
    Emits (user, positive item, negative item) triples for a ranking objective.

    Why the ranker moved off MSE
    ----------------------------
    The ranker was trained to regress watch_ratio with MSE. That optimises
    *calibration* — predicting how much of a video someone will watch — and it
    did so well, lifting watch-time AUC from 0.482 to 0.714 on big_matrix. But
    the metric that matters for a feed is recall@K: are the items the user
    actually wants in the top few slots? A regression head minimising squared
    error is pulled toward the conditional mean, and on the sparse big_matrix
    split that reordering made a good shortlist worse — the full pipeline
    scored below retrieval alone at every cutoff under 20 (recall@5 0.0031 vs
    0.0191).

    Retrieval was trained with a ranking loss (BPR) and behaved correctly at
    that scale. The ranker now uses the same framing and the same labels: a
    positive is watch_ratio >= pos_threshold, a negative is the *same user's*
    item at <= neg_threshold, and the loss only ever compares two items the
    same user saw.

    Args:
        interactions:  DataFrame with user_id, item_id, watch_ratio.
        user_embs:     (n_users, emb_dim) array.
        item_embs:     (n_items, emb_dim) array.
        features:      DenseFeatureStore, shared with evaluation and serving.
        pos_threshold: Minimum watch_ratio for a positive.
        neg_threshold: Maximum watch_ratio for a negative.
        seed:          RNG seed for negative sampling.
    """
    def __init__(self, interactions, user_embs, item_embs, features,
                 pos_threshold=0.7, neg_threshold=0.3, seed=42, num_neg=1):
        self.num_neg = num_neg
        self.user_embs = user_embs
        self.item_embs = item_embs
        self.features = features
        self.user_dense_dim = features.user_dense_dim
        self.item_dense_dim = features.item_dense_dim
        self.rng = np.random.default_rng(seed)

        positives = interactions.loc[
            interactions[Cols.WATCH_RATIO] >= pos_threshold,
            [Cols.USER_ID, Cols.ITEM_ID],
        ].to_numpy(dtype=np.int64)
        if len(positives) == 0:
            raise ValueError(
                f"No interactions with watch_ratio >= {pos_threshold}; "
                f"nothing to rank."
            )
        self.uids = positives[:, 0]
        self.pos_iids = positives[:, 1]

        negatives = interactions.loc[
            interactions[Cols.WATCH_RATIO] <= neg_threshold,
            [Cols.USER_ID, Cols.ITEM_ID],
        ]
        self.neg_pools = {
            int(uid): group.to_numpy(dtype=np.int64)
            for uid, group in negatives.groupby(Cols.USER_ID)[Cols.ITEM_ID]
        }
        self.global_neg_pool = negatives[Cols.ITEM_ID].to_numpy(dtype=np.int64)
        if len(self.global_neg_pool) == 0:
            raise ValueError(
                f"No interactions with watch_ratio <= {neg_threshold}; "
                f"no negatives available."
            )

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        return idx

    def sample_negatives(self, uids, num_neg):
        """Draw num_neg negatives per user from that user's own low-watch items."""
        neg_iids = np.empty((len(uids), num_neg), dtype=np.int64)
        for i, uid in enumerate(uids):
            pool = self.neg_pools.get(int(uid), self.global_neg_pool)
            neg_iids[i] = pool[self.rng.integers(0, len(pool), num_neg)]
        return neg_iids

    def build_rows(self, uids, iids):
        """Feature matrix for paired id arrays, via the shared layout."""
        return self.features.build_matrix(
            self.user_embs[uids], self.item_embs[iids], uids, iids
        )

    def collate(self, indices):
        """Build the positive and negative feature matrices for one batch."""
        indices = np.asarray(indices, dtype=np.int64)
        uids = self.uids[indices]
        pos_iids = self.pos_iids[indices]
        neg_iids = self.sample_negatives(uids, self.num_neg)[:, 0]

        return {
            "x_pos": torch.from_numpy(self.build_rows(uids, pos_iids)),
            "x_neg": torch.from_numpy(self.build_rows(uids, neg_iids)),
        }


class ListwiseRankingDataset(PairwiseRankingDataset):
    """
    Emits one positive against `num_neg` negatives, scored as a single list.

    Pairwise BPR asks "does the user prefer this positive to this one negative?".
    Listwise asks "does the positive rank first among these num_neg + 1 items?",
    which is the question recall@K actually poses. Each update sees num_neg
    contrasts instead of one, so the gradient is better conditioned — this is why
    large-scale rankers use a sampled softmax rather than a single pair.

    The loss is softmax cross-entropy over the candidate scores with the positive
    at index 0 (the same object as InfoNCE / sampled softmax). Negatives are
    drawn exactly as in the pairwise case: from the same user's own low-engagement
    items, so no cross-user popularity signal leaks in.

    Note that pairwise is the num_neg=1 special case of this framing, which is why
    the sampling and feature assembly are inherited rather than reimplemented.
    """
    def collate(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        uids = self.uids[indices]
        pos_iids = self.pos_iids[indices]
        neg_iids = self.sample_negatives(uids, self.num_neg)

        # Candidate 0 is the positive; the rest are negatives.
        candidates = np.concatenate([pos_iids[:, None], neg_iids], axis=1)
        n_candidates = candidates.shape[1]

        flat_uids = np.repeat(uids, n_candidates)
        flat_iids = candidates.ravel()

        return {
            "x": torch.from_numpy(self.build_rows(flat_uids, flat_iids)),
            "n_candidates": n_candidates,
        }


def listwise_loss(scores, n_candidates):
    """
    Softmax cross-entropy over one positive and n_candidates-1 negatives.

    scores arrives flat, one row per (user, candidate) pair, grouped so that each
    block of n_candidates belongs to one user with the positive first. Reshaping
    to (batch, n_candidates) and taking cross-entropy against target 0 maximises
    the probability that the positive ranks above every sampled negative at once,
    rather than beating one of them at a time.
    """
    scores = scores.view(-1, n_candidates)
    target = torch.zeros(scores.shape[0], dtype=torch.long, device=scores.device)
    return F.cross_entropy(scores, target)


def pairwise_loss(pos_scores, neg_scores):
    """
    BPR loss over ranker scores: -log sigma(score_pos - score_neg).

    Identical in form to the retrieval loss, applied to ranker outputs. Only
    the score difference matters, so the ranker is free to place scores
    anywhere on the real line as long as the ordering is right — which is
    exactly what recall@K rewards.
    """
    return -F.logsigmoid(pos_scores - neg_scores).mean()


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
    objective = cfg["ranking"]["objective"]
    if objective not in ("listwise", "pairwise", "regression"):
        raise ValueError(f"Unknown ranking.objective: {objective!r}")

    def make_dataset(df):
        if objective in ("listwise", "pairwise"):
            cls = ListwiseRankingDataset if objective == "listwise" else PairwiseRankingDataset
            return cls(
                df, user_embs, item_embs, features,
                pos_threshold=cfg["features"]["positive_watch_ratio"],
                neg_threshold=cfg["features"]["negative_watch_ratio"],
                seed=cfg["project"]["seed"],
                num_neg=cfg["features"]["num_neg_samples"] if objective == "listwise" else 1,
            )
        return RankingDataset(df, user_embs, item_embs, features)

    dataset = make_dataset(train_df)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=0, collate_fn=dataset.collate)

    val_dataset = make_dataset(val_df)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, collate_fn=val_dataset.collate)
    print(f"Ranker objective: {objective}")
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

    def batch_loss(batch):
        """Loss for one batch under the configured objective."""
        if objective == "listwise":
            scores = model(batch["x"].to(device)).squeeze(1)
            return listwise_loss(scores, batch["n_candidates"])
        if objective == "pairwise":
            pos = model(batch["x_pos"].to(device)).squeeze(1)
            neg = model(batch["x_neg"].to(device)).squeeze(1)
            return pairwise_loss(pos, neg)
        # Regression compares against watch_ratio in [0, 1], so it needs the
        # calibrated head rather than the raw score.
        preds = model.predict(batch["x"].to(device)).squeeze(1)
        return criterion(preds, batch["label"].to(device))

    def run_epoch(data_loader, train_mode):
        """One pass over data_loader. Shared so the two paths cannot diverge."""
        model.train() if train_mode else model.eval()
        total = 0.0
        with torch.set_grad_enabled(train_mode):
            for batch in data_loader:
                loss = batch_loss(batch)
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
        train_loss = run_epoch(loader, True)
        val_loss = run_epoch(val_loader, False)
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

        print(f"Epoch {epoch}/{epochs} — train {objective}: {train_loss:.6f} — "
              f"val {objective}: {val_loss:.6f} — {elapsed:.1f}s{marker}")

        if epochs_without_improvement >= patience:
            print(f"Early stopping: no validation improvement in {patience} epochs.")
            break

    model.load_state_dict(best_state)
    print(f"Restored best ranker from epoch {best_epoch} (val {best_val:.6f})")

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
