"""
Grouped AUC and user-bootstrap helpers for the KuaiRand evaluation.

GAUC here is per-user AUC averaged with users weighted equally, over users who
have both classes for the label. Ties get average ranks, so a constant score
gives exactly 0.5.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def per_user_auc(users: np.ndarray, labels: np.ndarray, scores: np.ndarray) -> pd.Series:
    """AUC per user (index = user id); users with one class are omitted."""
    df = pd.DataFrame({"u": np.asarray(users), "y": np.asarray(labels).astype(bool),
                       "s": np.asarray(scores, dtype=float)})
    df["r"] = df.groupby("u")["s"].rank(method="average")
    g = df.groupby("u")
    n = g["y"].size()
    n_pos = g["y"].sum()
    rank_pos = df["r"].where(df["y"], 0.0).groupby(df["u"]).sum()
    n_neg = n - n_pos
    ok = (n_pos > 0) & (n_neg > 0)
    return ((rank_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))[ok]


def users_with_both_classes(users: np.ndarray, labels: np.ndarray) -> int:
    s = pd.Series(np.asarray(labels).astype(bool)).groupby(np.asarray(users))
    return int((s.any() & ~s.all()).sum())


def _boot_means(values: np.ndarray, n_boot: int, seed: int, chunk: int = 200) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(values)
    out = np.empty(n_boot)
    for start in range(0, n_boot, chunk):
        k = min(chunk, n_boot - start)
        out[start:start + k] = values[rng.integers(0, n, size=(k, n))].mean(axis=1)
    return out


def bootstrap_mean(values, n_boot: int = 2000, seed: int = 42) -> dict:
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    means = _boot_means(v, n_boot, seed)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {"n": int(len(v)), "mean": float(v.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "half_width": float((hi - lo) / 2)}


def paired_bootstrap(a: pd.Series, b: pd.Series, n_boot: int = 2000, seed: int = 42) -> dict:
    """User bootstrap of mean(b - a) over users present in both; two-sided p."""
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    d = (j["b"] - j["a"]).to_numpy()
    means = _boot_means(d, n_boot, seed)
    lo, hi = np.quantile(means, [0.025, 0.975])
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return {"n": int(len(d)), "mean_a": float(j["a"].mean()), "mean_b": float(j["b"].mean()),
            "diff": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "p": float(min(1.0, max(p, 1.0 / n_boot)))}
