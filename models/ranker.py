"""
MLP ranker — predicts watch_ratio from a concatenated feature vector.

Input: [user_emb || item_emb || user_dense_features || item_dense_features]
Output: scalar in [0, 1] (sigmoid-activated)
"""
import torch
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
        Raw score, shape (batch, 1) — see forward(). predict() applies a sigmoid
        for a calibrated watch_ratio in [0, 1].

    Training objective is selected by config (ranking.objective):
        pairwise    BPR over (positive, negative) pairs — optimises ordering
        regression  MSE against observed watch_ratio — optimises calibration

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
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        Score candidates as raw logits.

        The final Sigmoid used to live inside the network. It was moved out so
        the pairwise objective can operate on unbounded scores: a sigmoid
        squashes every score into [0, 1], which compresses the margin between a
        positive and a negative and flattens the gradient exactly where the
        ranking loss needs signal. Sigmoid is monotonic, so ordering — and
        therefore every ranking metric — is identical either way; only the
        gradients differ.

        Use predict() when a calibrated watch_ratio in [0, 1] is wanted.

        Args:
            x: Float tensor of shape (batch, input_dim).

        Returns:
            Unbounded scores, shape (batch, 1). Higher means more relevant.
        """
        return self.net(x)

    def predict(self, x):
        """
        Calibrated watch_ratio prediction in [0, 1].

        Ranking by predict() and ranking by forward() produce the same order.
        This exists for the serving contract, which reports a watch_ratio, and
        for the regression objective, whose MSE target lives in [0, 1].
        """
        return torch.sigmoid(self.net(x))


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
