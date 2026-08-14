"""
MLP ranker — predicts watch_ratio from a concatenated feature vector.

Input: [user_emb || item_emb || user_dense_features || item_dense_features]
Output: scalar in [0, 1] (sigmoid-activated)
"""
import torch.nn as nn


class MLPRanker(nn.Module):
    """
    Multi-layer perceptron that re-ranks retrieval candidates.

    Role in the pipeline:
        The two-tower model retrieves the top-K candidates using fast ANN
        search (cosine similarity only).  MLPRanker scores each candidate
        more precisely by combining *both* user and item representations with
        richer features — enabling it to model interaction effects that the
        dot-product similarity cannot capture.

    Input construction (done outside this class):
        x = concat([user_emb, item_emb, user_dense_features, item_dense_features])
        shape: (batch, 2*embedding_dim + user_dense_dim + item_dense_dim)

    Output:
        Predicted watch_ratio ∈ [0, 1], shape (batch, 1).
        Sigmoid activation ensures the output is always a valid probability.
        Trained with MSE loss against the observed watch_ratio.

    Args:
        input_dim:   Total dimension of the concatenated feature vector.
        hidden_dims: List of hidden layer widths, e.g. [256, 128].
        dropout:     Dropout probability applied after each hidden activation.
    """
    def __init__(self, input_dim, hidden_dims, dropout):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: Float tensor of shape (batch, input_dim).

        Returns:
            Predicted watch_ratio, shape (batch, 1), values in [0, 1].
        """
        return self.net(x)


def build_ranker(cfg, user_dense_dim: int = None, item_dense_dim: int = None) -> MLPRanker:
    """
    Construct an MLPRanker from a config dict.

    input_dim = 2 * embedding_dim + user_dense_dim + item_dense_dim

    Pass user_dense_dim and item_dense_dim explicitly when the actual feature
    dimensions differ from the config defaults (e.g. after loading KuaiRec data
    with expanded one-hot category columns).

    Args:
        cfg:             Merged config dict.
        user_dense_dim:  Actual user dense feature dimension; falls back to
                         cfg["features"]["user_dense_dim"] if None.
        item_dense_dim:  Actual item dense feature dimension; same fallback.

    Returns:
        An untrained MLPRanker.
    """
    emb_dim = cfg["two_tower"]["embedding_dim"]
    if user_dense_dim is None:
        user_dense_dim = cfg["features"]["user_dense_dim"]
    if item_dense_dim is None:
        item_dense_dim = cfg["features"]["item_dense_dim"]
    input_dim = 2 * emb_dim + user_dense_dim + item_dense_dim
    hidden_dims = cfg["ranking"]["hidden_dims"]
    dropout = cfg["ranking"]["dropout"]
    return MLPRanker(input_dim, hidden_dims, dropout)
