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
from features.dense_features import DenseFeatureStore
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


class TowerFeatures:
    """
    Supplies side features to the retrieval towers, or nothing at all.

    When disabled the towers are ID-only: dense width is reported as 0, so
    build_model constructs an input layer that expects only the ID embedding
    and the batches produced here are (batch, 0) tensors that concatenate
    cleanly. That is deliberately different from feeding zero vectors through
    dense weights the model can never use — the earlier behaviour, which made
    the towers ID-only in effect while the architecture and the design doc both
    claimed otherwise.
    """
    def __init__(self, store, enabled: bool):
        self.store = store
        self.enabled = enabled
        self.user_dense_dim = store.user_dense_dim if enabled else 0
        self.item_dense_dim = store.item_dense_dim if enabled else 0

    def user_batch(self, uids):
        if not self.enabled:
            return torch.zeros(len(uids), 0)
        return self.store.user_batch(uids)

    def item_batch(self, iids):
        if not self.enabled:
            return torch.zeros(len(iids), 0)
        return self.store.item_batch(iids)


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


def sampled_softmax_loss(user_emb, pos_item_emb, temperature=0.07, false_neg_mask=None):
    """
    In-batch sampled softmax (InfoNCE) — cross-entropy over the batch.

    Why this exists, and why BPR was not enough
    -------------------------------------------
    BPR gives every user an *independent* objective: point toward whatever
    direction separates that user's positives from their negatives. Nothing in
    it relates one user to another, so there is no force keeping two users'
    embeddings apart.

    On this data that is fatal. Item quality alone explains 29.5% of watch_ratio
    variance and is by far the strongest single direction, so it acts as an
    attractor that every user slides into. Training collapsed all 7,176 user
    embeddings onto one vector — measured per-dimension std 4e-06, one unique
    row — while still beating a popularity baseline, because a global ordering
    is enough to do that.

    The signal being abandoned is real, not absent: users' top-50 lists overlap
    at Jaccard 0.045 against 0.003 expected by chance, and roughly 59% of
    watch_ratio variance is user-item interaction rather than item or user main
    effects.

    In-batch softmax supplies the missing term. Each user's positive is scored
    against *every other user's* positive in the batch, so two users with the
    same embedding cannot both pick out their own item and the loss penalises
    them for it. That is cross-user competition, which BPR has none of, and it
    is why production two-tower systems use this rather than pairwise loss.

    Args:
        user_emb:     (B, dim) L2-normalised user embeddings.
        pos_item_emb: (B, dim) embeddings of each user's positive item.
        temperature:  softmax temperature; lower sharpens the distribution.

    Returns:
        Scalar loss. Chance level is log(B).
    """
    logits = (user_emb @ pos_item_emb.T) / temperature

    # Mask false negatives. In-batch softmax assumes another user's positive is
    # unlikely to be yours — true on sparse data, false here. Measured on this
    # split: 13.5% of in-batch negatives are items the user actually likes (an
    # undercount from a 2M-positive sample; the true rate is higher). Without
    # masking, a quarter or more of the gradient pushes away items it should
    # pull closer, the signal cancels, and training sits at chance — log(1024)
    # = 6.9315, observed 6.9308.
    #
    # This is the same failure as the original uniform negative sampling bug:
    # the standard recipe assumes a sparse matrix and this one is not.
    if false_neg_mask is not None:
        logits = logits.masked_fill(false_neg_mask, float("-inf"))

    targets = torch.arange(len(user_emb), device=user_emb.device)
    return F.cross_entropy(logits, targets)


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


def build_optimizer(model, lr, weight_decay):
    """
    Adam with weight decay applied only to matrix weights.

    Why this is not the default: PyTorch applies weight_decay to *every*
    parameter, including LayerNorm gains and biases. A LayerNorm gain that
    decays toward zero zeroes that layer's output; the gradient path behind it
    dies, and decay then grinds the remaining weights to zero as well. It is a
    death spiral with no error message.

    That is exactly what happened here. The user tower ended up with every
    weight at 0.000000 except the final bias, so the tower computed
    normalize(bias) — one constant unit vector — and all 7,176 users received an
    identical embedding. Retrieval returned the same ranking for everybody while
    still beating the popularity baseline, because a global item ordering is
    enough to do that. Nothing crashed.

    Excluding 1-D parameters (LayerNorm gains, all biases) from decay is the
    standard remedy and is what every transformer implementation does.
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (no_decay if param.ndim <= 1 else decay).append(param)

    return torch.optim.Adam(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
    )


def assert_not_collapsed(embeddings, name, min_std=1e-4):
    """
    Fail loudly if an embedding table degenerated to a single vector.

    A collapsed table is not a crash, it is a silent loss of personalisation:
    every row identical means every user gets the same recommendations. This
    check exists because that shipped once, undetected, through training,
    evaluation, and a cloud deployment.
    """
    spread = float(embeddings.std(axis=0).mean())
    unique_rows = len(np.unique(np.round(embeddings, 5), axis=0))
    print(f"  {name}: per-dim std {spread:.6f}, {unique_rows:,} unique rows "
          f"of {len(embeddings):,}")
    if spread < min_std or unique_rows < 2:
        raise SystemExit(
            f"{name} collapsed: per-dimension std {spread:.2e} (min {min_std:.0e}), "
            f"{unique_rows} unique rows. Every entity would receive identical "
            f"recommendations. Refusing to export."
        )


def run_epoch(model, loader, features, num_neg, device, optimizer=None,
              objective="bpr", temperature=0.07, positives=None):
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

    total_loss = 0.0

    with torch.set_grad_enabled(training):
        for batch in loader:
            batch_size = batch["user_id"].shape[0]

            user_ids = batch["user_id"]
            pos_items = batch["pos_item"]
            neg_items_flat = batch["neg_items"].view(-1)

            user_emb, pos_emb = model(
                user_ids.to(device),
                features.user_batch(user_ids).to(device),
                pos_items.to(device),
                features.item_batch(pos_items).to(device),
            )

            neg_emb_flat = model.item_tower(
                neg_items_flat.to(device),
                features.item_batch(neg_items_flat).to(device),
            )
            neg_emb = neg_emb_flat.view(batch_size, num_neg, -1)

            if objective == "infonce":
                mask = None
                if positives is not None:
                    u = user_ids.numpy()
                    i = pos_items.numpy()
                    # rows = users in the batch, cols = their positive items
                    dense = positives[u][:, i].toarray().astype(bool)
                    np.fill_diagonal(dense, False)   # the true target must survive
                    mask = torch.from_numpy(dense).to(device)
                loss = sampled_softmax_loss(user_emb, pos_emb, temperature, mask)
            else:
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

    # Whether the towers see side features is a measured choice — see the
    # ablation table in config/base.yaml. When disabled the towers are genuinely
    # ID-only (dense width 0) rather than fed zero vectors through unused
    # weights, so the architecture matches what is actually being trained.
    features = TowerFeatures(
        DenseFeatureStore.load(),
        enabled=cfg["two_tower"]["use_dense_features"],
    )
    print(f"Tower side features: "
          f"{'enabled' if features.enabled else 'disabled (ID-only towers)'}")

    device = get_device(cfg)
    torch.manual_seed(cfg["project"]["seed"])
    np.random.seed(cfg["project"]["seed"])
    model = build_model(
        cfg, n_users, n_items,
        user_dense_dim=features.user_dense_dim,
        item_dense_dim=features.item_dense_dim,
    ).to(device)

    optimizer = build_optimizer(
        model,
        lr=cfg["training"]["retrieval"]["lr"],
        weight_decay=cfg["training"]["retrieval"]["weight_decay"],
    )

    epochs = cfg["training"]["retrieval"]["epochs"]
    patience = cfg["training"]["retrieval"]["early_stopping_patience"]
    objective = cfg["two_tower"].get("objective", "bpr")
    temperature = cfg["two_tower"].get("temperature", 0.07)
    print(f"Retrieval objective: {objective}"
          + (f" (temperature {temperature})" if objective == "infonce" else ""))

    # Sparse user x item matrix of known positives, used to mask false negatives
    # in the in-batch softmax. Built once; each batch pulls a 1024x1024 submatrix.
    positives = None
    if objective == "infonce":
        from scipy.sparse import csr_matrix
        pos_rows = train_df.loc[
            train_df[Cols.WATCH_RATIO] >= cfg["features"]["positive_watch_ratio"]
        ]
        positives = csr_matrix(
            (np.ones(len(pos_rows), dtype=np.int8),
             (pos_rows[Cols.USER_ID].to_numpy(), pos_rows[Cols.ITEM_ID].to_numpy())),
            shape=(n_users, n_items),
        )
        print(f"False-negative mask: {positives.nnz:,} known positives")

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
        train_loss = run_epoch(model, loader, features, num_neg, device, optimizer,
                               objective, temperature, positives)
        val_loss = run_epoch(model, val_loader, features, num_neg, device,
                             optimizer=None, objective=objective,
                             temperature=temperature, positives=positives)
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

    # Export with the same features the towers were trained on. Exporting with
    # zeros here while training on real features would be train/serve skew in
    # the retrieval stage.
    model.eval()
    with torch.no_grad():
        all_item_ids = torch.arange(n_items)
        item_embs = model.item_tower(
            all_item_ids.to(device),
            features.item_batch(all_item_ids).to(device),
        ).cpu().numpy()

    assert_not_collapsed(item_embs, "item embeddings")
    item_emb_path = "datastore/processed/item_embeddings.npy"
    np.save(item_emb_path, item_embs)
    print(f"Saved item embeddings: {item_emb_path}")

    with torch.no_grad():
        all_user_ids = torch.arange(n_users)
        user_embs = model.user_tower(
            all_user_ids.to(device),
            features.user_batch(all_user_ids).to(device),
        ).cpu().numpy()

    assert_not_collapsed(user_embs, "user embeddings")
    user_emb_path = "datastore/processed/user_embeddings.npy"
    np.save(user_emb_path, user_embs)
    print(f"Saved user embeddings: {user_emb_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/kuairec.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg)
