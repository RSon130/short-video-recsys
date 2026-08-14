"""
Two-tower retrieval model.

UserTower and ItemTower each produce L2-normalised embeddings.
Dot product of two normalised vectors equals cosine similarity.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class UserTower(nn.Module):
    """
    Encodes a user into a fixed-size L2-normalised embedding vector.

    Architecture:
        1. ID embedding lookup  — maps user_id (int) → dense vector of size emb_dim.
           Each user gets its own trainable embedding, capturing collaborative
           filtering signals (users with similar history cluster together).
        2. Concatenation        — joins the ID embedding with side-feature vector
           (age, activity level, etc.) to form the full user representation.
        3. MLP with LayerNorm   — projects the concat into the shared retrieval
           space. LayerNorm stabilises training by normalising each hidden layer's
           pre-activation, reducing internal covariate shift.
        4. L2 normalisation     — final step ensures ‖output‖₂ = 1, so inner
           product with item embeddings equals cosine similarity directly.

    Args:
        n_users:     Vocabulary size — total number of unique users after ID mapping.
        emb_dim:     Dimension of the learnable ID embedding.
        dense_dim:   Dimension of the side-feature vector concatenated to the embedding.
        hidden_dims: List of hidden layer widths, e.g. [256, 128].
        output_dim:  Final embedding dimension (shared with ItemTower).
        dropout:     Dropout probability applied after each hidden activation.
    """
    def __init__(self, n_users, emb_dim, dense_dim, hidden_dims, output_dim, dropout):
        super().__init__()
        self.id_emb = nn.Embedding(n_users, emb_dim)
        layers = []
        prev = emb_dim + dense_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, user_ids, user_features):
        """
        Args:
            user_ids:      Long tensor of shape (batch,) — internal integer IDs.
            user_features: Float tensor of shape (batch, dense_dim) — side features.

        Returns:
            L2-normalised user embeddings, shape (batch, output_dim).
        """
        x = torch.cat([self.id_emb(user_ids), user_features], dim=1)
        return F.normalize(self.mlp(x), dim=1)


class ItemTower(nn.Module):
    """
    Encodes an item into a fixed-size L2-normalised embedding vector.

    Mirrors UserTower exactly — same MLP + LayerNorm + L2-norm pattern.
    Both towers project into the *same* embedding space so that
    user–item relevance can be measured with a single dot product at
    retrieval time (no learned similarity layer needed).

    Args:
        n_items:     Vocabulary size — total number of unique items.
        item_emb_dim: Dimension of the learnable item ID embedding.
        dense_dim:   Dimension of item side features (category, play count, etc.).
        hidden_dims: MLP hidden layer widths.
        output_dim:  Final embedding dimension (must match UserTower).
        dropout:     Dropout probability.
    """
    def __init__(self, n_items, item_emb_dim, dense_dim, hidden_dims, output_dim, dropout):
        super().__init__()
        self.id_emb = nn.Embedding(n_items, item_emb_dim)
        layers = []
        prev = item_emb_dim + dense_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, item_ids, item_features):
        """
        Args:
            item_ids:      Long tensor of shape (batch,).
            item_features: Float tensor of shape (batch, dense_dim).

        Returns:
            L2-normalised item embeddings, shape (batch, output_dim).
        """
        x = torch.cat([self.id_emb(item_ids), item_features], dim=1)
        return F.normalize(self.mlp(x), dim=1)


class TwoTowerModel(nn.Module):
    """
    Thin wrapper that runs both towers in a single forward pass.

    The two-tower architecture is the industry standard for large-scale
    retrieval (used at YouTube, Pinterest, Twitter).  The key insight is
    that encoding users and items *independently* allows item embeddings
    to be pre-computed offline and stored in a FAISS index.  At serving
    time only the user tower runs online — O(1) index lookup replaces an
    O(n) dot-product scan over all items.

    Training objective: BPR (Bayesian Personalised Ranking) — pushes the
    user embedding closer to positive items and away from sampled negatives.
    """
    def __init__(self, user_tower, item_tower):
        super().__init__()
        self.user_tower = user_tower
        self.item_tower = item_tower

    def forward(self, user_ids, user_features, item_ids, item_features):
        """
        Returns:
            (user_emb, item_emb) — both L2-normalised, shape (batch, embedding_dim).
            Their dot product is cosine similarity ∈ [-1, 1].
        """
        user_emb = self.user_tower(user_ids, user_features)
        item_emb = self.item_tower(item_ids, item_features)
        return user_emb, item_emb


def build_model(cfg, n_users, n_items,
                user_dense_dim=None, item_dense_dim=None) -> TwoTowerModel:
    """
    Construct a TwoTowerModel from a config dict.

    Reads all hyperparameters from cfg so callers never hard-code dimensions.
    This is the single entry point used by both the training script and tests.

    Args:
        cfg:            Merged config dict (base.yaml + dataset override).
        n_users:        Total unique users — ID embedding vocabulary size.
        n_items:        Total unique items — ID embedding vocabulary size.
        user_dense_dim: Actual user side-feature width. Falls back to the config
                        placeholder when None; pass the real value, which is
                        only known once the processed features are loaded.
        item_dense_dim: Same, for items.

    Returns:
        An untrained TwoTowerModel.
    """
    user_emb_dim = cfg["features"]["user_emb_dim"]
    item_emb_dim = cfg["features"]["item_emb_dim"]
    if user_dense_dim is None:
        user_dense_dim = cfg["features"]["user_dense_dim"]
    if item_dense_dim is None:
        item_dense_dim = cfg["features"]["item_dense_dim"]
    tower_hidden = cfg["two_tower"]["tower_hidden"]
    embedding_dim = cfg["two_tower"]["embedding_dim"]
    dropout = cfg["two_tower"]["dropout"]

    user_tower = UserTower(n_users, user_emb_dim, user_dense_dim, tower_hidden, embedding_dim, dropout)
    item_tower = ItemTower(n_items, item_emb_dim, item_dense_dim, tower_hidden, embedding_dim, dropout)
    return TwoTowerModel(user_tower, item_tower)
